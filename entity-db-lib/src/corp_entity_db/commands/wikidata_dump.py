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
    record_type: str          # "people" or "org"
    records: list             # PersonRecord or CompanyRecord list
    embedding_texts: list[str]
    last_entity_index: int
    last_entity_id: str
    people_count: int         # cumulative people yielded so far
    orgs_count: int           # cumulative orgs yielded so far


def _reader_thread(
    importer,
    *,
    people: bool,
    orgs: bool,
    limit: Optional[int],
    require_enwiki: bool,
    skip_updates: bool,
    existing_people_ids: set[str],
    existing_org_ids: set[str],
    start_index: int,
    batch_size: int,
    embed_queue: queue.Queue,
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Reader thread: iterates the dump, accumulates batches, puts them on embed_queue."""
    import logging
    logger = logging.getLogger(__name__)

    people_records: list = []
    org_records: list = []
    last_entity_index = start_index
    last_entity_id = ""
    people_yielded = 0
    orgs_yielded = 0

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
            )
            embed_queue.put(batch)
            org_records = []

    try:
        for record_type, record in importer.import_all(
            people_limit=limit if people else 0,
            orgs_limit=limit if orgs else 0,
            import_people=people,
            import_orgs=orgs,
            require_enwiki=require_enwiki,
            skip_people_ids=existing_people_ids if skip_updates else None,
            skip_org_ids=existing_org_ids if skip_updates else None,
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
            else:  # org
                org_records.append(record)
                if len(org_records) >= batch_size:
                    flush_orgs()

        # Flush remaining partial batches
        flush_people()
        flush_orgs()

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: reader is done
        embed_queue.put(None)


def _embedder_thread(
    embedder,
    *,
    embed_queue: queue.Queue,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    thread_errors: list[Exception],
) -> None:
    """Embedder thread: consumes ImportBatch, produces (batch, embeddings, scalar_embeddings)."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        while True:
            batch = embed_queue.get()
            if batch is None:
                # Reader is done
                break
            if shutdown_event.is_set():
                logger.info("Embedder thread: shutdown requested, stopping")
                break

            embeddings, scalar_embeddings = embedder.embed_batch_and_quantize(batch.embedding_texts)
            result_queue.put((batch, embeddings, scalar_embeddings))

    except Exception as e:
        thread_errors.append(e)
    finally:
        # Sentinel: embedder is done
        result_queue.put(None)


@click.command("import-wikidata-dump")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata JSON dump file (.bz2 or .gz)")
@click.option("--download", is_flag=True, help="Download latest dump first (~100GB)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--no-aria2", is_flag=True, help="Don't use aria2c even if available (slower)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--people/--no-people", default=True, help="Import people (default: yes)")
@click.option("--orgs/--no-orgs", default=True, help="Import organizations (default: yes)")
@click.option("--locations/--no-locations", default=False, help="Import locations (default: no)")
@click.option("--require-enwiki", is_flag=True, help="Only import orgs with English Wikipedia articles")
@click.option("--resume", is_flag=True, help="Resume from last position in dump file (tracks entity index)")
@click.option("--skip-updates", is_flag=True, help="Skip Q codes already in database (no updates)")
@click.option("--limit", type=int, help="Max records per type (people and/or orgs)")
@click.option("--batch-size", type=int, default=10000, help="Batch size for commits (default: 10000)")
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
    from corp_entity_db.embeddings import CompanyEmbedder
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

    # Initialize embedder (loads model, may take time on first run)
    click.echo("Loading embedding model...", err=True)
    sys.stderr.flush()
    embedder = CompanyEmbedder()
    click.echo("Embedding model loaded.", err=True)
    sys.stderr.flush()

    # Load existing QID labels from database and seed the importer's cache
    # readonly=False because we also write new labels via persist_new_labels()
    database = get_person_database(db_path=db_path_obj, readonly=False)
    existing_labels = database.get_all_qid_labels()
    if existing_labels:
        click.echo(f"Loaded {len(existing_labels):,} existing QID labels from DB", err=True)
        importer.set_label_cache(existing_labels)
    del existing_labels  # Free memory — labels are now in the importer's cache

    # Load existing source_ids for skip_updates mode
    existing_people_ids: set[str] = set()
    existing_org_ids: set[str] = set()
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

    # Helper to persist new labels after each batch
    def persist_new_labels() -> int:
        new_labels = importer.get_new_labels_since()
        if new_labels:
            database.insert_qid_labels(new_labels)
            importer.flush_new_labels()
            return len(new_labels)
        return 0

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
                            persist_new_labels()
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
                        persist_new_labels()
                        save_location_progress()
                        click.echo(f"\r  Progress: {locations_count:,} locations...", nl=False, err=True)
                        sys.stderr.flush()

                click.echo("", err=True)  # Newline after counter

            # Final batches
            flush_location_batch()
            persist_new_labels()
            save_location_progress()

        finally:
            # Ensure we save progress even on interrupt
            save_location_progress()

        click.echo(f"\nLocation import complete: {locations_count:,} locations", err=True)

        # Final label resolution
        click.echo("\n=== Final QID Label Resolution ===", err=True)
        all_labels = importer.get_label_cache(copy=False)
        click.echo(f"  Total labels in cache: {len(all_labels):,}", err=True)

        # Final stats
        final_label_count = database.get_qid_labels_count()
        click.echo(f"  Total labels in DB: {final_label_count:,}", err=True)

        locations_database.close()
        database.close()
        click.echo("\nWikidata dump import complete!", err=True)
        return

    # Combined import - single pass through the dump for both people and orgs
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
    if start_index > 0:
        click.echo(f"  Resuming from entity index {start_index:,}", err=True)

    # Initialize databases (readonly=False for import operations)
    person_database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False) if orgs else None

    # Pipeline: reader thread → embed_queue → embedder thread → result_queue → main thread (DB writes)
    embed_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queue: queue.Queue = queue.Queue(maxsize=2)
    shutdown_event = threading.Event()
    thread_errors: list[Exception] = []

    people_count = 0
    orgs_count = 0
    last_entity_index = start_index
    last_entity_id = ""

    def save_progress() -> None:
        if progress:
            progress.entity_index = last_entity_index
            progress.last_entity_id = last_entity_id
            progress.people_yielded = people_count
            progress.orgs_yielded = orgs_count
            progress.save()

    click.echo("Starting parallel pipeline (reader → embedder → writer)...", err=True)
    sys.stderr.flush()

    reader = threading.Thread(
        target=_reader_thread,
        args=(importer,),
        kwargs=dict(
            people=people,
            orgs=orgs,
            limit=limit,
            require_enwiki=require_enwiki,
            skip_updates=skip_updates,
            existing_people_ids=existing_people_ids,
            existing_org_ids=existing_org_ids,
            start_index=start_index,
            batch_size=batch_size,
            embed_queue=embed_queue,
            shutdown_event=shutdown_event,
            thread_errors=thread_errors,
        ),
        daemon=True,
        name="wikidata-reader",
    )
    embedder_th = threading.Thread(
        target=_embedder_thread,
        args=(embedder,),
        kwargs=dict(
            embed_queue=embed_queue,
            result_queue=result_queue,
            shutdown_event=shutdown_event,
            thread_errors=thread_errors,
        ),
        daemon=True,
        name="wikidata-embedder",
    )

    reader.start()
    embedder_th.start()

    try:
        while True:
            # Check for thread errors before blocking
            if thread_errors:
                raise thread_errors[0]

            result = result_queue.get()
            if result is None:
                # Embedder is done — all batches processed
                break

            batch, embeddings, scalar_embeddings = result

            # Insert into database (main thread owns SQLite writes)
            if batch.record_type == "people":
                person_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
            elif batch.record_type == "org" and org_database:
                org_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)

            # Update progress from batch metadata
            people_count = batch.people_count
            orgs_count = batch.orgs_count
            last_entity_index = batch.last_entity_index
            last_entity_id = batch.last_entity_id

            persist_new_labels()
            save_progress()

            click.echo(f"\r  Progress: {people_count:,} people, {orgs_count:,} orgs...", nl=False, err=True)
            sys.stderr.flush()

        click.echo("", err=True)  # Newline after counter

        # Check for any errors that occurred after the sentinel
        if thread_errors:
            raise thread_errors[0]

    except KeyboardInterrupt:
        click.echo("\n  Interrupted! Saving progress...", err=True)
        shutdown_event.set()
        # Drain any completed batches from result_queue so we don't lose work
        while True:
            try:
                result = result_queue.get_nowait()
                if result is None:
                    break
                batch, embeddings, scalar_embeddings = result
                if batch.record_type == "people":
                    person_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
                elif batch.record_type == "org" and org_database:
                    org_database.insert_batch(batch.records, embeddings, scalar_embeddings=scalar_embeddings)
                people_count = batch.people_count
                orgs_count = batch.orgs_count
                last_entity_index = batch.last_entity_index
                last_entity_id = batch.last_entity_id
            except queue.Empty:
                break
        persist_new_labels()
        save_progress()
        click.echo(f"  Saved: {people_count:,} people, {orgs_count:,} orgs", err=True)
        raise
    finally:
        save_progress()
        reader.join(timeout=5)
        embedder_th.join(timeout=5)

    click.echo(f"Import complete: {people_count:,} people, {orgs_count:,} orgs", err=True)

    # Backfill known_for_org from reverse org→person mappings
    # (P169 CEO, P488 chairperson, P112 founder, P1037 director, P3320 board member)
    reverse_map = importer.get_reverse_person_orgs()
    if reverse_map and people:
        # Resolve org QIDs to labels before backfilling (threads are joined, safe to read directly)
        label_cache = importer.get_label_cache(copy=False)
        resolved_map: dict[str, list[tuple[str, str, str | None, str | None]]] = {}
        for person_qid, entries in reverse_map.items():
            resolved_entries: list[tuple[str, str, str | None, str | None]] = []
            for org_qid, role, start_date, end_date in entries:
                org_label = label_cache.get(org_qid, "")
                if org_label:
                    resolved_entries.append((org_label, role, start_date, end_date))
            if resolved_entries:
                resolved_map[person_qid] = resolved_entries
        del label_cache  # Release reference
        if resolved_map:
            backfilled = person_database.backfill_known_for_org(resolved_map)
            click.echo(
                f"Backfilled known_for_org from org executive properties: "
                f"{backfilled:,} updated ({len(resolved_map):,} reverse mappings)",
                err=True,
            )
        del resolved_map

    # Free reverse map memory — no longer needed
    del reverse_map

    # Keep references for final label resolution
    database = person_database
    if org_database:
        org_database.close()

    # Final label resolution pass for any remaining unresolved QIDs
    click.echo("\n=== Final QID Label Resolution ===", err=True)

    # Use the live label cache (threads are joined, single-threaded from here)
    all_labels = importer.get_label_cache(copy=False)
    click.echo(f"  Total labels in cache: {len(all_labels):,}", err=True)

    # Check for any remaining unresolved QIDs in the database
    people_unresolved = database.get_unresolved_qids()
    click.echo(f"  Unresolved QIDs in people: {len(people_unresolved):,}", err=True)

    org_unresolved: set[str] = set()
    if orgs:
        # readonly=False because we also call resolve_qid_labels later
        org_database = get_database(db_path=db_path_obj, readonly=False)
        org_unresolved = org_database.get_unresolved_qids()
        click.echo(f"  Unresolved QIDs in orgs: {len(org_unresolved):,}", err=True)

    all_unresolved = people_unresolved | org_unresolved
    need_sparql = all_unresolved - set(all_labels.keys())

    if need_sparql:
        click.echo(f"  Resolving {len(need_sparql):,} remaining QIDs via SPARQL...", err=True)
        sparql_resolved = importer.resolve_qids_via_sparql(need_sparql)
        all_labels.update(sparql_resolved)
        # Persist newly resolved labels
        if sparql_resolved:
            database.insert_qid_labels(sparql_resolved)
            click.echo(f"  SPARQL resolved and stored: {len(sparql_resolved):,}", err=True)

    # Update records with any newly resolved labels
    if all_labels:
        updates, deletes = database.resolve_qid_labels(all_labels)
        if updates or deletes:
            click.echo(f"  People: {updates:,} updated, {deletes:,} duplicates deleted", err=True)

        if orgs:
            org_database = get_database(db_path=db_path_obj, readonly=False)
            org_updates, org_deletes = org_database.resolve_qid_labels(all_labels)
            if org_updates or org_deletes:
                click.echo(f"  Orgs: {org_updates:,} updated, {org_deletes:,} duplicates deleted", err=True)
            org_database.close()

    # Final stats
    final_label_count = database.get_qid_labels_count()
    click.echo(f"  Total labels in DB: {final_label_count:,}", err=True)

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
