"""Wikidata dump import — import people/orgs/locations from Wikidata JSON dump."""

import queue
import sys
import threading
from pathlib import Path
from typing import NamedTuple, Optional

import click

from ..models import RoleRecord
from ._common import _configure_logging, _resolve_db_path


class ImportBatch(NamedTuple):
    """A batch of records ready for embedding, produced by the reader thread."""
    record_type: str          # "people", "org", or "location"
    records: list             # PersonRecord, CompanyRecord, or LocationRecord list
    embedding_texts: list[str]  # empty for locations (no embeddings needed)
    last_entity_index: int
    last_entity_id: str
    people_count: int         # cumulative people yielded so far
    orgs_count: int           # cumulative orgs yielded so far
    locations_count: int = 0  # cumulative locations yielded so far
    roles_count: int = 0      # cumulative roles yielded so far


def _reader_thread(
    importer,
    *,
    people: bool,
    orgs: bool,
    locations: bool,
    limit: Optional[int],
    require_enwiki: bool,
    skip_updates: bool,
    existing_people_ids: set[str],
    existing_org_ids: set[str],
    existing_location_ids: set[str],
    start_index: int,
    batch_size: int,
    embed_queue: queue.Queue,
    location_queue: queue.Queue,
    role_queue: queue.Queue,
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Reader thread: iterates the dump, accumulates batches, puts them on embed_queue.

    Location and role records are put on their own queues directly (no embedding needed).
    """
    import logging
    logger = logging.getLogger(__name__)

    people_records: list = []
    org_records: list = []
    location_records: list = []
    role_records: list = []
    last_entity_index = start_index
    last_entity_id = ""
    people_yielded = 0
    orgs_yielded = 0
    locations_yielded = 0
    def progress_callback(entity_index: int, entity_id: str, _ppl_count: int, _org_count: int) -> None:
        nonlocal last_entity_index, last_entity_id
        last_entity_index = entity_index
        last_entity_id = entity_id

    def flush_people() -> None:
        nonlocal people_records, people_yielded
        if people_records and not shutdown_event.is_set():
            texts = [r.get_embedding_text() for r in people_records]
            people_yielded += len(people_records)
            batch = ImportBatch(
                record_type="people",
                records=list(people_records),
                embedding_texts=texts,
                last_entity_index=last_entity_index,
                last_entity_id=last_entity_id,
                people_count=people_yielded,
                orgs_count=orgs_yielded,
                locations_count=locations_yielded,
            )
            embed_queue.put(batch)
            people_records = []

    def flush_orgs() -> None:
        nonlocal org_records, orgs_yielded
        if org_records and not shutdown_event.is_set():
            texts = [r.name for r in org_records]
            orgs_yielded += len(org_records)
            batch = ImportBatch(
                record_type="org",
                records=list(org_records),
                embedding_texts=texts,
                last_entity_index=last_entity_index,
                last_entity_id=last_entity_id,
                people_count=people_yielded,
                orgs_count=orgs_yielded,
                locations_count=locations_yielded,
            )
            embed_queue.put(batch)
            org_records = []

    def flush_locations() -> None:
        nonlocal location_records, locations_yielded
        if location_records and not shutdown_event.is_set():
            locations_yielded += len(location_records)
            location_queue.put(list(location_records))
            location_records = []

    def flush_roles() -> None:
        nonlocal role_records
        if role_records and not shutdown_event.is_set():
            role_queue.put(list(role_records))
            role_records = []

    try:
        for record_type, record in importer.import_all(
            people_limit=limit if people else 0,
            orgs_limit=limit if orgs else 0,
            locations_limit=limit if locations else 0,
            import_people=people,
            import_orgs=orgs,
            import_locations=locations,
            require_enwiki=require_enwiki,
            skip_people_ids=existing_people_ids if skip_updates else None,
            skip_org_ids=existing_org_ids if skip_updates else None,
            skip_location_ids=existing_location_ids if skip_updates else None,
            start_index=start_index,
            progress_callback=progress_callback,
        ):
            if shutdown_event.is_set():
                logger.info("Reader thread: shutdown requested, stopping")
                break

            if record_type == "person":
                people_records.append(record)
                if len(people_records) >= batch_size:
                    flush_people()
            elif record_type == "org":
                org_records.append(record)
                if len(org_records) >= batch_size:
                    flush_orgs()
            elif record_type == "location":
                location_records.append(record)
                if len(location_records) >= batch_size:
                    flush_locations()
            elif record_type == "role":
                role_records.append(record)
                if len(role_records) >= batch_size:
                    flush_roles()

        # Flush remaining partial batches
        flush_people()
        flush_orgs()
        flush_locations()
        flush_roles()

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: reader is done
        embed_queue.put(None)
        location_queue.put(None)
        role_queue.put(None)


def _run_fk_only(db_path: Path, *, people: bool, orgs: bool, locations: bool) -> None:
    """Resolve FK relations by reading FK data from the record JSON column in the database.

    No dump re-read needed — all FK QIDs (country_qid, org_qid, parent_qids) are stored
    in the record column during pass 1.
    """
    import json as _json
    from corp_entity_db.store import get_person_database, get_database, get_locations_database

    click.echo("\n=== FK-only mode: Resolving FK relations from database ===", err=True)

    person_database = get_person_database(db_path=db_path, readonly=False)
    org_database = get_database(db_path=db_path, readonly=False) if orgs else None
    locations_database = get_locations_database(db_path=db_path, readonly=False) if locations else None

    conn = person_database._connect()

    # Location FKs: read parent_qids/country_qids from record column
    if locations and locations_database:
        click.echo("  Reading location FK data from database...", err=True)
        location_fks: dict[int, dict] = {}
        cursor = conn.execute("SELECT qid, record FROM locations WHERE qid IS NOT NULL AND source_id = 4")
        for row in cursor:
            rec = _json.loads(row[1]) if row[1] else {}
            parent_qids = rec.get("parent_qids", [])
            country_qids = rec.get("country_qids", [])
            if parent_qids or country_qids:
                location_fks[row[0]] = {"parent_qids": parent_qids, "country_qids": country_qids}
        click.echo(f"  Loaded {len(location_fks):,} location FK records", err=True)
        resolved = locations_database.resolve_parent_ids(location_fks)
        click.echo(f"  Resolved {resolved:,} location parent_ids", err=True)

    # Org FKs: read country_qid from record column
    if orgs and org_database:
        click.echo("  Reading org FK data from database...", err=True)
        org_fks: dict[int, dict] = {}
        cursor = conn.execute("SELECT qid, record FROM organizations WHERE qid IS NOT NULL")
        for row in cursor:
            rec = _json.loads(row[1]) if row[1] else {}
            country_qid = rec.get("country_qid", "")
            if country_qid:
                org_fks[row[0]] = {"country_qid": country_qid}
        click.echo(f"  Loaded {len(org_fks):,} org FK records", err=True)
        resolved = org_database.resolve_fks(org_fks)
        click.echo(f"  Resolved {resolved:,} org region_ids", err=True)

    # Backfill missing role records from people's P106 occupation QIDs
    if people:
        from corp_entity_db.store import get_roles_database
        from corp_entity_db.importers.wikidata_dump import WikidataDumpImporter
        roles_database = get_roles_database(db_path=db_path, readonly=False)
        roles_conn = roles_database._connect()
        existing_role_qids = {row[0] for row in roles_conn.execute("SELECT qid FROM roles WHERE qid IS NOT NULL")}

        # Collect all unique occupation QIDs from people's record JSON
        click.echo("  Scanning people records for occupation QIDs...", err=True)
        needed_qids: set[int] = set()
        cursor = conn.execute(
            "SELECT record FROM people WHERE source_id = 4 AND record IS NOT NULL"
        )
        for row in cursor:
            rec = _json.loads(row[0]) if row[0] else {}
            for occ in rec.get("occupations", []):
                if occ.startswith("Q") and occ[1:].isdigit():
                    needed_qids.add(int(occ[1:]))
            role_qid = rec.get("role_qid", "")
            if role_qid and role_qid.startswith("Q") and role_qid[1:].isdigit():
                needed_qids.add(int(role_qid[1:]))
            for pos_qid in rec.get("positions", []):
                if isinstance(pos_qid, str) and pos_qid.startswith("Q") and pos_qid[1:].isdigit():
                    needed_qids.add(int(pos_qid[1:]))

        missing_qids = needed_qids - existing_role_qids
        if missing_qids:
            with click.progressbar(
                length=len(missing_qids),
                label=f"  Fetching {len(missing_qids):,} role labels",
                file=sys.stderr,
            ) as bar:
                api_labels = WikidataDumpImporter.fetch_qid_labels(
                    missing_qids,
                    progress_callback=lambda done, total: bar.update(done - bar.pos),
                )
            role_records = []
            for qid_int, label in api_labels.items():
                role_records.append(RoleRecord(
                    name=label,
                    source="wikidata",
                    source_id=f"Q{qid_int}",
                    qid=qid_int,
                    record={"wikidata_id": f"Q{qid_int}", "label": label, "backfilled": True},
                ))
            if role_records:
                inserted = roles_database.insert_batch(role_records)
                click.echo(f"  Backfilled {inserted:,} missing role records", err=True)
        else:
            click.echo("  All occupation/position QIDs already have role records", err=True)
        roles_database.close()

    # Insert discovered organizations: scan people records for org QIDs not in organizations table
    if people and orgs and org_database:
        click.echo("  Scanning people records for org QIDs...", err=True)
        needed_org_qids: set[int] = set()
        cursor = conn.execute(
            "SELECT record FROM people WHERE source_id = 4 AND record IS NOT NULL"
        )
        for row in cursor:
            rec = _json.loads(row[0]) if row[0] else {}
            org_qid = rec.get("org_qid", "")
            if org_qid and org_qid.startswith("Q") and org_qid[1:].isdigit():
                needed_org_qids.add(int(org_qid[1:]))

        if needed_org_qids:
            org_conn = org_database._connect()
            existing_org_qids = {row[0] for row in org_conn.execute(
                "SELECT qid FROM organizations WHERE qid IS NOT NULL"
            )}
            missing_org_qids = needed_org_qids - existing_org_qids

            if missing_org_qids:
                click.echo(f"  Found {len(missing_org_qids):,} org QIDs not in organizations table", err=True)
                with click.progressbar(
                    length=len(missing_org_qids),
                    label=f"  Fetching {len(missing_org_qids):,} org labels",
                    file=sys.stderr,
                ) as bar:
                    api_labels = WikidataDumpImporter.fetch_qid_labels(
                        missing_org_qids,
                        progress_callback=lambda done, total: bar.update(done - bar.pos),
                    )
                from corp_entity_db.models import CompanyRecord, EntityType
                org_records = []
                for qid_int, label in api_labels.items():
                    org_records.append(CompanyRecord(
                        name=label,
                        source="wikipedia",
                        source_id=f"Q{qid_int}",
                        entity_type=EntityType.UNKNOWN,
                        record={"wikidata_id": f"Q{qid_int}", "label": label, "discovered_from": "people_import"},
                    ))
                if org_records:
                    inserted = org_database.insert_batch(org_records)
                    click.echo(f"  Inserted {inserted:,} discovered organizations", err=True)
            else:
                click.echo("  All org QIDs already in organizations table", err=True)

    # People FKs: resolve_fks reads each row's record JSON directly
    # Resolves country_id, known_for_org_id, and known_for_role_id
    if people:
        resolved = person_database.resolve_fks()
        click.echo(f"  Resolved {resolved:,} person FKs (country_id, known_for_org, known_for_role)", err=True)

    person_database.close()
    if org_database:
        org_database.close()
    if locations_database:
        locations_database.close()

    click.echo("\nFK resolution complete!", err=True)


@click.command("import-wikidata-dump")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata JSON dump file (.bz2 or .gz)")
@click.option("--download", is_flag=True, help="Download latest dump first (~100GB)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--no-aria2", is_flag=True, help="Don't use aria2c even if available (slower)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--people/--no-people", default=True, help="Import people (default: yes)")
@click.option("--orgs/--no-orgs", default=True, help="Import organizations (default: yes)")
@click.option("--locations/--no-locations", default=True, help="Import locations (default: yes)")
@click.option("--require-enwiki", is_flag=True, help="Only import orgs with English Wikipedia articles")
@click.option("--resume", is_flag=True, help="Resume from last position in dump file (tracks entity index)")
@click.option("--skip-updates", is_flag=True, help="Skip Q codes already in database (no updates)")
@click.option("--limit", type=int, help="Max records per type (people and/or orgs)")
@click.option("--batch-size", type=int, default=10000, help="Batch size for commits (default: 10000)")
@click.option("--fk-only", is_flag=True, help="Skip import, only resolve FK relations (pass 2)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_wikidata_dump(
    dump_path: Optional[str],
    download: bool,
    force: bool,
    no_aria2: bool,
    db_path: Optional[str],
    people: bool,
    orgs: bool,
    locations: bool,
    require_enwiki: bool,
    resume: bool,
    skip_updates: bool,
    limit: Optional[int],
    batch_size: int,
    fk_only: bool,
    verbose: bool,
):
    """
    Import people, organizations, and locations from Wikidata JSON dump.

    This uses the full Wikidata JSON dump (~100GB compressed) to import
    all humans and organizations with English Wikipedia articles. This
    avoids SPARQL query timeouts that occur with large result sets.

    The dump is streamed line-by-line to minimize memory usage.

    \b
    Features:
    - No timeouts (processes locally)
    - Complete coverage (all notable people/orgs)
    - Resumable with --resume (tracks position in dump file)
    - Skip existing with --skip-updates (loads existing Q codes)
    - People like Andy Burnham are captured via occupation (P106)
    - Locations (countries, cities, regions) with parent hierarchy

    \b
    Resume options:
    - --resume: Resume from where the dump processing left off (tracks entity index).
                Progress is saved after each batch. Use this if import was interrupted.
    - --skip-updates: Skip Q codes already in database (no updates to existing records).
                      Use this to add new records without re-processing existing ones.

    \b
    Examples:
        corp-entity-db import-wikidata-dump --dump /path/to/dump.json.bz2 --limit 10000
        corp-entity-db import-wikidata-dump --download --people --no-orgs --limit 50000
        corp-entity-db import-wikidata-dump --dump dump.json.bz2 --orgs --no-people
        corp-entity-db import-wikidata-dump --dump dump.json.bz2 --locations --no-people --no-orgs  # Locations only
        corp-entity-db import-wikidata-dump --dump dump.json.bz2 --resume  # Resume interrupted import
        corp-entity-db import-wikidata-dump --dump dump.json.bz2 --skip-updates  # Skip existing Q codes
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database, get_locations_database
    from corp_entity_db.importers.wikidata_dump import WikidataDumpImporter, DumpProgress

    if not fk_only and not dump_path and not download:
        raise click.UsageError("Either --dump path or --download is required")

    if not people and not orgs and not locations:
        raise click.UsageError("Must import at least one of --people, --orgs, or --locations")

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Wikidata dump to {db_path_obj}...", err=True)

    # === FK-only mode: skip import, just resolve FK relations ===
    # Reads FK data from the record JSON column in the database (no dump re-read needed).
    if fk_only:
        _run_fk_only(db_path_obj, people=people, orgs=orgs, locations=locations)
        return

    # Initialize importer
    importer = WikidataDumpImporter(dump_path=dump_path)

    # Download if requested
    if download:
        import shutil
        dump_target = importer.get_dump_path()
        click.echo(f"Downloading Wikidata dump (~100GB) to:", err=True)
        click.echo(f"  {dump_target}", err=True)

        # Check for aria2c
        has_aria2 = shutil.which("aria2c") is not None
        use_aria2 = has_aria2 and not no_aria2

        if use_aria2:
            click.echo("  Using aria2c for fast parallel download (16 connections)", err=True)
            dump_file = importer.download_dump(force=force, use_aria2=True)
            click.echo(f"\nUsing dump: {dump_file}", err=True)
        else:
            if not has_aria2:
                click.echo("", err=True)
                click.echo("  TIP: Install aria2c for 10-20x faster downloads:", err=True)
                click.echo("       brew install aria2  (macOS)", err=True)
                click.echo("       apt install aria2   (Ubuntu/Debian)", err=True)
                click.echo("", err=True)

            # Use urllib to get content length first
            import urllib.request
            req = urllib.request.Request(
                "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2",
                headers={"User-Agent": "corp-entity-db/1.0"},
                method="HEAD"
            )
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("content-length", 0))

            if total_size:
                total_gb = total_size / (1024 ** 3)
                click.echo(f"  Size: {total_gb:.1f} GB", err=True)

            # Download with progress bar
            progress_bar = None

            def update_progress(downloaded: int, total: int) -> None:
                nonlocal progress_bar
                if progress_bar is None and total > 0:
                    progress_bar = click.progressbar(
                        length=total,
                        label="Downloading",
                        show_percent=True,
                        show_pos=True,
                        item_show_func=lambda x: f"{(x or 0) / (1024**3):.1f} GB" if x else "",
                    )
                    progress_bar.__enter__()
                if progress_bar:
                    # Update to absolute position
                    progress_bar.update(downloaded - progress_bar.pos)

            try:
                dump_file = importer.download_dump(force=force, use_aria2=False, progress_callback=update_progress)
            finally:
                if progress_bar:
                    progress_bar.__exit__(None, None, None)

            click.echo(f"\nUsing dump: {dump_file}", err=True)
    elif dump_path:
        click.echo(f"Using dump: {dump_path}", err=True)


    database = get_person_database(db_path=db_path_obj, readonly=False)

    # Load existing source_ids for skip_updates mode
    existing_people_ids: set[str] = set()
    existing_org_ids: set[str] = set()
    existing_location_ids: set[str] = set()
    if skip_updates:
        click.echo("Loading existing records for --skip-updates...", err=True)
        if people:
            existing_people_ids = database.get_all_source_ids(source="wikidata")
            click.echo(f"  Found {len(existing_people_ids):,} existing people Q codes", err=True)
        if orgs:
            # readonly=False because we also write later via insert_batch
            org_database = get_database(db_path=db_path_obj, readonly=False)
            existing_org_ids = org_database.get_all_source_ids(source="wikipedia")
            click.echo(f"  Found {len(existing_org_ids):,} existing org Q codes", err=True)
        if locations:
            locations_database = get_locations_database(db_path=db_path_obj, readonly=False)
            existing_location_ids = locations_database.get_all_source_ids(source="wikidata")
            click.echo(f"  Found {len(existing_location_ids):,} existing location Q codes", err=True)

    # Load progress for resume mode (position-based resume)
    progress: Optional[DumpProgress] = None
    start_index = 0
    if resume:
        progress = DumpProgress.load()
        if progress:
            # Verify the progress is for the same dump file
            actual_dump_path = importer._dump_path or Path(dump_path) if dump_path else importer.get_dump_path()
            if progress.matches_dump(actual_dump_path):
                start_index = progress.entity_index
                click.echo(f"Resuming from entity index {start_index:,}", err=True)
                click.echo(f"  Last entity: {progress.last_entity_id}", err=True)
                click.echo(f"  Last updated: {progress.last_updated}", err=True)
            else:
                click.echo("Warning: Progress file is for a different dump, starting from beginning", err=True)
                progress = None
        else:
            click.echo("No progress file found, starting from beginning", err=True)

    # Initialize progress tracking
    if progress is None:
        actual_dump_path = importer._dump_path or Path(dump_path) if dump_path else importer.get_dump_path()
        progress = DumpProgress(
            dump_path=str(actual_dump_path),
            dump_size=actual_dump_path.stat().st_size if actual_dump_path.exists() else 0,
        )

    # ========================================
    # Location-only import (separate pass)
    # ========================================
    if locations and not people and not orgs:
        from corp_entity_db.store import get_locations_database

        click.echo("\n=== Location Import ===", err=True)
        click.echo(f"  Locations: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if require_enwiki:
            click.echo("    Filter: only locations with English Wikipedia articles", err=True)

        # Initialize locations database (readonly=False for import operations)
        locations_database = get_locations_database(db_path=db_path_obj, readonly=False)

        # Load existing location Q codes for skip_updates mode
        existing_location_ids: set[str] = set()
        if skip_updates:
            existing_location_ids = locations_database.get_all_source_ids(source="wikidata")
            click.echo(f"    Skip updates: {len(existing_location_ids):,} existing Q codes", err=True)

        if start_index > 0:
            click.echo(f"  Resuming from entity index {start_index:,}", err=True)

        location_records: list = []
        locations_count = 0
        last_entity_index = start_index
        last_entity_id = ""

        def location_progress_callback(entity_index: int, entity_id: str, _loc_count: int) -> None:
            nonlocal last_entity_index, last_entity_id
            last_entity_index = entity_index
            last_entity_id = entity_id

        def save_location_progress() -> None:
            if progress:
                progress.entity_index = last_entity_index
                progress.last_entity_id = last_entity_id
                progress.save()

        def flush_location_batch() -> None:
            nonlocal location_records, locations_count
            if location_records:
                inserted = locations_database.insert_batch(location_records)
                locations_count += inserted
                location_records = []

        click.echo("Starting dump iteration...", err=True)
        sys.stderr.flush()

        try:
            if limit:
                # Use progress bar when we have limits
                with click.progressbar(
                    length=limit,
                    label="Processing dump",
                    show_percent=True,
                    show_pos=True,
                ) as pbar:
                    for record in importer.import_locations(
                        limit=limit,
                        require_enwiki=require_enwiki,
                        skip_ids=existing_location_ids if skip_updates else None,
                        start_index=start_index,
                        progress_callback=location_progress_callback,
                    ):
                        pbar.update(1)
                        location_records.append(record)
                        if len(location_records) >= batch_size:
                            flush_location_batch()

                            save_location_progress()
            else:
                # No limit - show counter updates
                for record in importer.import_locations(
                    limit=None,
                    require_enwiki=require_enwiki,
                    skip_ids=existing_location_ids if skip_updates else None,
                    start_index=start_index,
                    progress_callback=location_progress_callback,
                ):
                    location_records.append(record)
                    if len(location_records) >= batch_size:
                        flush_location_batch()

                        save_location_progress()
                        click.echo(f"\r  Progress: {locations_count:,} locations...", nl=False, err=True)
                        sys.stderr.flush()

                click.echo("", err=True)  # Newline after counter

            # Final batches
            flush_location_batch()
            save_location_progress()

        finally:
            # Ensure we save progress even on interrupt
            save_location_progress()

        click.echo(f"\nLocation import complete: {locations_count:,} locations", err=True)

        # === Pass 2: Resolve FK relations (location parent_ids) ===
        click.echo("\n=== Pass 2: Resolving FK relations ===", err=True)
        sys.stderr.flush()

        location_fks: dict[int, dict] = {}
        for entity_type, qid, fk_data in importer.import_fk_relations():
            if entity_type == "location" and qid.startswith("Q") and qid[1:].isdigit():
                location_fks[int(qid[1:])] = fk_data

        click.echo(f"  Extracted FK data for {len(location_fks):,} locations", err=True)

        resolved_locs = locations_database.resolve_parent_ids(location_fks)
        click.echo(f"  Resolved {resolved_locs:,} location parent_ids", err=True)

        locations_database.close()
        database.close()
        click.echo("\nWikidata dump import complete!", err=True)
        return

    # Combined import - single pass through the dump for people, orgs, and locations
    click.echo("\n=== Combined Import (single dump pass) ===", err=True)
    sys.stderr.flush()  # Ensure output is visible immediately
    if people:
        click.echo(f"  People: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if skip_updates and existing_people_ids:
            click.echo(f"    Skip updates: {len(existing_people_ids):,} existing Q codes", err=True)
    if orgs:
        click.echo(f"  Orgs: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if require_enwiki:
            click.echo("    Filter: only orgs with English Wikipedia articles", err=True)
        if skip_updates and existing_org_ids:
            click.echo(f"    Skip updates: {len(existing_org_ids):,} existing Q codes", err=True)
    if locations:
        click.echo(f"  Locations: {'up to ' + str(limit) + ' records' if limit else 'unlimited'}", err=True)
        if skip_updates and existing_location_ids:
            click.echo(f"    Skip updates: {len(existing_location_ids):,} existing Q codes", err=True)
    if start_index > 0:
        click.echo(f"  Resuming from entity index {start_index:,}", err=True)

    # Initialize databases (readonly=False for import operations)
    from corp_entity_db.store import get_roles_database
    person_database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False) if orgs else None
    locations_database = get_locations_database(db_path=db_path_obj, readonly=False) if locations else None
    roles_database = get_roles_database(db_path=db_path_obj, readonly=False)

    # Pipeline: reader thread → write_queue → main thread (DB writes)
    # Embeddings are deferred to post-import (needs FK resolution for rich text)
    # Locations and roles bypass the write_queue and go directly via their own queues
    write_queue: queue.Queue = queue.Queue(maxsize=4)
    location_queue: queue.Queue = queue.Queue(maxsize=4)
    role_queue: queue.Queue = queue.Queue(maxsize=4)
    shutdown_event = threading.Event()
    thread_errors: list[Exception] = []

    people_count = 0
    orgs_count = 0
    locations_count = 0
    roles_count = 0
    last_entity_index = start_index
    last_entity_id = ""

    def save_progress() -> None:
        if progress:
            progress.entity_index = last_entity_index
            progress.last_entity_id = last_entity_id
            progress.people_yielded = people_count
            progress.orgs_yielded = orgs_count
            progress.save()

    def drain_location_queue() -> None:
        """Drain all pending location batches from the queue."""
        nonlocal locations_count
        while True:
            try:
                loc_batch = location_queue.get_nowait()
                if loc_batch is None:
                    break
                if locations_database:
                    inserted = locations_database.insert_batch(loc_batch)
                    locations_count += inserted
            except queue.Empty:
                break

    def drain_role_queue() -> None:
        """Drain all pending role batches from the queue."""
        nonlocal roles_count
        while True:
            try:
                role_batch = role_queue.get_nowait()
                if role_batch is None:
                    break
                inserted = roles_database.insert_batch(role_batch)
                roles_count += inserted
            except queue.Empty:
                break

    click.echo("Starting pipeline (reader → writer, embeddings deferred to post-import)...", err=True)
    sys.stderr.flush()

    reader = threading.Thread(
        target=_reader_thread,
        args=(importer,),
        kwargs=dict(
            people=people,
            orgs=orgs,
            locations=locations,
            limit=limit,
            require_enwiki=require_enwiki,
            skip_updates=skip_updates,
            existing_people_ids=existing_people_ids,
            existing_org_ids=existing_org_ids,
            existing_location_ids=existing_location_ids,
            start_index=start_index,
            batch_size=batch_size,
            embed_queue=write_queue,
            location_queue=location_queue,
            role_queue=role_queue,
            shutdown_event=shutdown_event,
            thread_errors=thread_errors,
        ),
        daemon=True,
        name="wikidata-reader",
    )

    reader.start()

    try:
        while True:
            # Check for thread errors before blocking
            if thread_errors:
                raise thread_errors[0]

            # Drain location and role queues (non-blocking) while waiting for record batches
            drain_location_queue()
            drain_role_queue()

            try:
                batch = write_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if batch is None:
                # Reader is done — all batches sent
                break

            # Insert into database without embeddings (main thread owns SQLite writes)
            if batch.record_type == "people":
                person_database.insert_batch(batch.records)
            elif batch.record_type == "org" and org_database:
                org_database.insert_batch(batch.records)

            # Update progress from batch metadata
            people_count = batch.people_count
            orgs_count = batch.orgs_count
            last_entity_index = batch.last_entity_index
            last_entity_id = batch.last_entity_id

            save_progress()

            click.echo(
                f"\r  Progress: {people_count:,} people, {orgs_count:,} orgs, "
                f"{locations_count:,} locations, {roles_count:,} roles...",
                nl=False, err=True,
            )
            sys.stderr.flush()

        # Drain any remaining location and role batches
        drain_location_queue()
        drain_role_queue()

        click.echo("", err=True)  # Newline after counter

        # Check for any errors that occurred after the sentinel
        if thread_errors:
            raise thread_errors[0]

    except KeyboardInterrupt:
        click.echo("\n  Interrupted! Saving progress...", err=True)
        shutdown_event.set()
        # Drain any completed batches from write_queue so we don't lose work
        while True:
            try:
                batch = write_queue.get_nowait()
                if batch is None:
                    break
                if batch.record_type == "people":
                    person_database.insert_batch(batch.records)
                elif batch.record_type == "org" and org_database:
                    org_database.insert_batch(batch.records)
                people_count = batch.people_count
                orgs_count = batch.orgs_count
                last_entity_index = batch.last_entity_index
                last_entity_id = batch.last_entity_id
            except queue.Empty:
                break
        drain_location_queue()
        drain_role_queue()
        save_progress()
        click.echo(f"  Saved: {people_count:,} people, {orgs_count:,} orgs, {locations_count:,} locations, {roles_count:,} roles", err=True)
        raise
    finally:
        save_progress()
        reader.join(timeout=5)

    click.echo(f"Import complete: {people_count:,} people, {orgs_count:,} orgs, {locations_count:,} locations, {roles_count:,} roles", err=True)

    # === Backfill missing role records ===
    # Some occupation/position QIDs may have been missed by _is_role_entity during the dump scan.
    # Fetch their labels from the Wikidata API and create role records before FK resolution.
    click.echo("\n=== Backfilling missing role records ===", err=True)
    sys.stderr.flush()
    conn = roles_database._connect()
    existing_role_qids = {row[0] for row in conn.execute("SELECT qid FROM roles WHERE qid IS NOT NULL")}
    missing_qids = importer.get_missing_role_qids(existing_role_qids)
    if missing_qids:
        # First check _role_labels cache (populated during dump scan for detected roles)
        cached_labels = {
            int(qid_str[1:]): label
            for qid_str, label in importer._role_labels.items()
            if qid_str.startswith("Q") and qid_str[1:].isdigit() and int(qid_str[1:]) in missing_qids
        }
        still_missing = missing_qids - set(cached_labels.keys())

        if still_missing:
            with click.progressbar(
                length=len(still_missing),
                label=f"  Fetching {len(still_missing):,} role labels",
                file=sys.stderr,
            ) as bar:
                api_labels = WikidataDumpImporter.fetch_qid_labels(
                    still_missing,
                    progress_callback=lambda done, total: bar.update(done - bar.pos),
                )
            cached_labels.update(api_labels)

        # Create role records for all resolved labels
        role_records = []
        for qid_int, label in cached_labels.items():
            role_records.append(RoleRecord(
                name=label,
                source="wikidata",
                source_id=f"Q{qid_int}",
                qid=qid_int,
                record={"wikidata_id": f"Q{qid_int}", "label": label, "backfilled": True},
            ))
        if role_records:
            inserted = roles_database.insert_batch(role_records)
            roles_count += inserted
            click.echo(f"  Backfilled {inserted:,} missing role records", err=True)
        else:
            click.echo("  No role records to backfill", err=True)
    else:
        click.echo("  All occupation/position QIDs already have role records", err=True)

    # === Insert discovered organizations ===
    # People referenced org QIDs (P108 employer, P39 qualifiers, etc.) that may not be
    # in the organizations table. Insert them as stubs with labels from the Wikidata API.
    if people and orgs and org_database:
        discovered_qids = importer.get_discovered_org_qids()
        if discovered_qids:
            org_conn = org_database._connect()
            existing_org_qids = {row[0] for row in org_conn.execute(
                "SELECT qid FROM organizations WHERE qid IS NOT NULL"
            )}
            missing_qids = set()
            for q in discovered_qids:
                if q.startswith("Q") and q[1:].isdigit():
                    qid_int = int(q[1:])
                    if qid_int not in existing_org_qids:
                        missing_qids.add(qid_int)

            if missing_qids:
                click.echo(f"\n=== Inserting {len(missing_qids):,} discovered organizations ===", err=True)
                sys.stderr.flush()
                with click.progressbar(
                    length=len(missing_qids),
                    label=f"  Fetching {len(missing_qids):,} org labels",
                    file=sys.stderr,
                ) as bar:
                    api_labels = WikidataDumpImporter.fetch_qid_labels(
                        missing_qids,
                        progress_callback=lambda done, total: bar.update(done - bar.pos),
                    )
                from ..models import CompanyRecord, EntityType
                org_records = []
                for qid_int, label in api_labels.items():
                    org_records.append(CompanyRecord(
                        name=label,
                        source="wikipedia",
                        source_id=f"Q{qid_int}",
                        entity_type=EntityType.UNKNOWN,
                        record={"wikidata_id": f"Q{qid_int}", "label": label, "discovered_from": "people_import"},
                    ))
                if org_records:
                    inserted = org_database.insert_batch(org_records)
                    click.echo(f"  Inserted {inserted:,} discovered organizations", err=True)
                else:
                    click.echo("  No labels resolved for discovered organizations", err=True)
            else:
                click.echo("  All discovered org QIDs already in organizations table", err=True)

    # === Pass 2: Resolve FK relations ===
    click.echo("\n=== Pass 2: Resolving FK relations ===", err=True)
    sys.stderr.flush()

    person_fks: dict[int, dict] = {}
    org_fks: dict[int, dict] = {}
    location_fks: dict[int, dict] = {}
    fk_count = 0

    for entity_type, qid, fk_data in importer.import_fk_relations():
        if not qid.startswith("Q") or not qid[1:].isdigit():
            continue
        qid_int = int(qid[1:])
        if entity_type == "person":
            person_fks[qid_int] = fk_data
        elif entity_type == "org":
            org_fks[qid_int] = fk_data
        elif entity_type == "location":
            location_fks[qid_int] = fk_data
        fk_count += 1
        if fk_count % 500_000 == 0:
            click.echo(
                f"\r  Reading FKs: {len(person_fks):,} people, {len(org_fks):,} orgs, "
                f"{len(location_fks):,} locations...",
                nl=False, err=True,
            )
            sys.stderr.flush()

    click.echo(
        f"\n  Extracted FK data: {len(person_fks):,} people, {len(org_fks):,} orgs, "
        f"{len(location_fks):,} locations",
        err=True,
    )

    # Resolve in order: locations first (others reference them), then orgs, then people
    if locations and locations_database and location_fks:
        resolved = locations_database.resolve_parent_ids(location_fks)
        click.echo(f"  Resolved {resolved:,} location parent_ids", err=True)

    if orgs and org_database and org_fks:
        # Re-open org_database if it was closed
        if org_database._conn is None:
            org_database = get_database(db_path=db_path_obj, readonly=False)
        resolved = org_database.resolve_fks(org_fks)
        click.echo(f"  Resolved {resolved:,} org region_ids", err=True)

    if people:
        resolved = person_database.resolve_fks(person_fks if person_fks else None)
        click.echo(f"  Resolved {resolved:,} person FKs (country_id, known_for_org, known_for_role)", err=True)

    # Backfill known_for_org_id from reverse org→person mappings
    # (rebuilt during pass 2: P169 CEO, P488 chairperson, P112 founder, P1037 director, P3320 board member)
    reverse_map = importer.get_reverse_person_orgs()
    if reverse_map and people:
        # Pass org QIDs directly — backfill_known_for_org resolves to org FKs
        backfilled = person_database.backfill_known_for_org(reverse_map)
        click.echo(
            f"Backfilled known_for_org from org executive properties: "
            f"{backfilled:,} updated ({len(reverse_map):,} reverse mappings)",
            err=True,
        )

    # Free FK data and reverse map memory
    del person_fks, org_fks, location_fks, reverse_map

    # Keep references for final label resolution
    database = person_database
    if org_database:
        org_database.close()
    if locations_database:
        locations_database.close()
    roles_database.close()

    # Run canonicalization to link equivalent records across sources
    click.echo("\n=== Canonicalization ===", err=True)
    if people:
        people_result = person_database.canonicalize()
        click.echo(
            f"  People: {people_result['canonical_groups']:,} groups, "
            f"{people_result['matched_by_org']:,} by org, "
            f"{people_result['matched_by_date']:,} by date",
            err=True,
        )
    if orgs:
        canon_org_database = get_database(db_path=db_path_obj, readonly=False)
        org_result = canon_org_database.canonicalize()
        click.echo(
            f"  Orgs: {org_result.get('groups_found', 0):,} groups",
            err=True,
        )
        canon_org_database.close()

    database.close()

    click.echo("\nWikidata dump import complete!", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)


@click.command("backfill-aliases")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata dump file")
@click.option("--download", is_flag=True, help="Download the dump if not present")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_backfill_aliases(dump_path: Optional[str], download: bool, db_path: Optional[str], verbose: bool):
    """
    Backfill Wikidata aliases into existing org record JSON from the dump.

    Scans the Wikidata dump for entities matching existing orgs and writes
    their English aliases into the record JSON. Follow up with:

    \b
        corp-entity-db populate-aliases
        corp-entity-db build-index --no-people

    \b
    Examples:
        corp-entity-db backfill-aliases --download
        corp-entity-db backfill-aliases --dump /path/to/latest-all.json.bz2
    """
    import sqlite3

    _configure_logging(verbose)

    from ..importers.wikidata_dump import WikidataDumpImporter

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    importer = WikidataDumpImporter(dump_path=dump_path)

    if download:
        click.echo("Downloading Wikidata dump (this may take a while)...", err=True)
        importer.download_dump()

    if importer._dump_path is None:
        # Try default cache location
        default_dump = Path.home() / ".cache" / "corp-extractor" / "wikidata-latest-all.json.bz2"
        if default_dump.exists():
            importer._dump_path = default_dump
        else:
            raise click.ClickException(
                "No dump file found. Use --dump /path/to/dump or --download"
            )

    click.echo(f"Database: {db_path_obj}", err=True)
    click.echo(f"Dump: {importer._dump_path}", err=True)

    conn = sqlite3.connect(str(db_path_obj))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-500000")

    def progress(entity_index: int, entity_id: str) -> None:
        if entity_index % 1_000_000 == 0:
            click.echo(f"  Scanned {entity_index:,} entities...", err=True)

    updated = importer.backfill_aliases(conn, progress_callback=progress)
    conn.close()

    click.echo(f"\nBackfilled aliases for {updated:,} organizations.", err=True)
    click.echo("Now run:", err=True)
    click.echo("  corp-entity-db populate-aliases", err=True)
    click.echo("  corp-entity-db build-index --no-people", err=True)
