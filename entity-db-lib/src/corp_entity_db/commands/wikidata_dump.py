"""Wikidata dump import — import people/orgs/locations from Wikidata JSON dump."""

import queue
import sys
import threading
from pathlib import Path
from typing import NamedTuple, Optional

import click

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
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Reader thread: iterates the dump, accumulates batches, puts them on embed_queue.

    Location records are put on location_queue directly (no embedding needed).
    """
    import logging
    logger = logging.getLogger(__name__)

    people_records: list = []
    org_records: list = []
    location_records: list = []
    last_entity_index = start_index
    last_entity_id = ""
    people_yielded = 0
    orgs_yielded = 0
    locations_yielded = 0

    def progress_callback(entity_index: int, entity_id: str, ppl_count: int, org_count: int) -> None:
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

        # Flush remaining partial batches
        flush_people()
        flush_orgs()
        flush_locations()

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: reader is done
        embed_queue.put(None)
        location_queue.put(None)


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

    if not dump_path and not download:
        raise click.UsageError("Either --dump path or --download is required")

    if not people and not orgs and not locations:
        raise click.UsageError("Must import at least one of --people, --orgs, or --locations")

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Wikidata dump to {db_path_obj}...", err=True)

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

    # === FK-only mode: skip import, just resolve FK relations ===
    if fk_only:
        click.echo("\n=== FK-only mode: Resolving FK relations ===", err=True)
        sys.stderr.flush()

        person_database = get_person_database(db_path=db_path_obj, readonly=False)
        org_database = get_database(db_path=db_path_obj, readonly=False) if orgs else None
        locations_database = get_locations_database(db_path=db_path_obj, readonly=False) if locations else None

        person_fks: dict[int, dict] = {}
        org_fks: dict[int, dict] = {}
        location_fks: dict[int, dict] = {}
        fk_count = 0

        click.echo("Reading dump for FK data...", err=True)
        for entity_type, qid, fk_data in importer.import_fk_relations():
            if not qid.startswith("Q") or not qid[1:].isdigit():
                continue
            qid_int = int(qid[1:])
            if entity_type == "person" and people:
                person_fks[qid_int] = fk_data
            elif entity_type == "org" and orgs:
                org_fks[qid_int] = fk_data
            elif entity_type == "location" and locations:
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

        # Resolve in order: locations first, then orgs, then people
        if locations_database and location_fks:
            resolved = locations_database.resolve_parent_ids(location_fks)
            click.echo(f"  Resolved {resolved:,} location parent_ids", err=True)

        if org_database and org_fks:
            resolved = org_database.resolve_fks(org_fks)
            click.echo(f"  Resolved {resolved:,} org region_ids", err=True)

        if people and person_fks:
            resolved = person_database.resolve_fks(person_fks)
            click.echo(f"  Resolved {resolved:,} person FKs (country_id, known_for_org)", err=True)

        # Backfill known_for_org from reverse mappings rebuilt during pass 2
        reverse_map = importer.get_reverse_person_orgs()
        if reverse_map and people:
            backfilled = person_database.backfill_known_for_org(reverse_map)
            click.echo(
                f"  Backfilled known_for_org: {backfilled:,} updated ({len(reverse_map):,} reverse mappings)",
                err=True,
            )

        person_database.close()
        if org_database:
            org_database.close()
        if locations_database:
            locations_database.close()

        click.echo("\nFK resolution complete!", err=True)
        return

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

        def location_progress_callback(entity_index: int, entity_id: str, loc_count: int) -> None:
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
    person_database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False) if orgs else None
    locations_database = get_locations_database(db_path=db_path_obj, readonly=False) if locations else None

    # Pipeline: reader thread → write_queue → main thread (DB writes)
    # Embeddings are deferred to post-import (needs FK resolution for rich text)
    # Locations bypass the write_queue and go directly via location_queue
    write_queue: queue.Queue = queue.Queue(maxsize=4)
    location_queue: queue.Queue = queue.Queue(maxsize=4)
    shutdown_event = threading.Event()
    thread_errors: list[Exception] = []

    people_count = 0
    orgs_count = 0
    locations_count = 0
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

            # Drain location queue (non-blocking) while waiting for record batches
            drain_location_queue()

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
                f"{locations_count:,} locations...",
                nl=False, err=True,
            )
            sys.stderr.flush()

        # Drain any remaining location batches
        drain_location_queue()

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
        save_progress()
        click.echo(f"  Saved: {people_count:,} people, {orgs_count:,} orgs, {locations_count:,} locations", err=True)
        raise
    finally:
        save_progress()
        reader.join(timeout=5)

    click.echo(f"Import complete: {people_count:,} people, {orgs_count:,} orgs, {locations_count:,} locations", err=True)

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

    if people and person_fks:
        resolved = person_database.resolve_fks(person_fks)
        click.echo(f"  Resolved {resolved:,} person FKs (country_id, known_for_org)", err=True)

    # Backfill known_for_org_id/known_for_org_location_id from reverse org→person mappings
    # (rebuilt during pass 2: P169 CEO, P488 chairperson, P112 founder, P1037 director, P3320 board member)
    reverse_map = importer.get_reverse_person_orgs()
    if reverse_map and people:
        # Pass org QIDs directly — backfill_known_for_org resolves to org/location FKs
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
