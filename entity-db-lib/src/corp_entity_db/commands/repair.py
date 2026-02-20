"""Database repair commands for fixing people records after resumed Wikidata imports."""

import json
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import click

from ._common import _configure_logging, _resolve_db_path


def _insert_discovered_org(
    conn,
    org_qid: str,
    org_label: str,
    discovered_from: str,
) -> None:
    """Insert a single discovered org via raw SQL (no embeddings needed).

    Embeddings will be generated later by ``db post-import``.
    """
    from corp_entity_db.seed_data import SOURCE_NAME_TO_ID, ORG_TYPE_NAME_TO_ID

    name_normalized = org_label.lower().strip()
    record_json = json.dumps({
        "wikidata_id": org_qid,
        "discovered_from": discovered_from,
        "needs_label_resolution": org_label == org_qid,
    })
    source_type_id = SOURCE_NAME_TO_ID.get("wikipedia", 4)
    entity_type_id = ORG_TYPE_NAME_TO_ID.get("business", 17)

    # Parse QID integer from org_qid (e.g. "Q312" -> 312)
    qid = None
    if org_qid.startswith("Q"):
        qid_str = org_qid[1:]
        if qid_str.isdigit():
            qid = int(qid_str)

    conn.execute("""
        INSERT OR IGNORE INTO organizations
        (name, name_normalized, source_id, source_identifier, qid, entity_type_id, from_date, to_date, record)
        VALUES (?, ?, ?, ?, ?, ?, '', '', ?)
    """, (org_label, name_normalized, source_type_id, org_qid, qid, entity_type_id, record_json))


@click.command("repair-resume")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_repair_resume(db_path: Optional[str], verbose: bool):
    """
    Repair people records after a resumed Wikidata dump import.

    When a dump import is interrupted and resumed, the in-memory org→person
    executive mappings are lost (orgs before the resume point aren't re-scanned).
    This command reconstructs those mappings from the 'executives' key stored in
    org record JSON and backfills people missing known_for_org.

    Requires org records imported AFTER the executives field was added. For older
    databases, use 'fix-resume --dump <path>' instead.

    \b
    Steps:
    1. Rebuild reverse_person_orgs from org record JSON (executives field)
    2. Resolve org QIDs to labels from DB
    3. Backfill people with missing known_for_org (idempotent)
    4. Insert discovered orgs referenced by people but not in orgs table

    \b
    Examples:
        corp-entity-db repair-resume
        corp-entity-db repair-resume --db /path/to/entities.db
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    org_database = get_database(db_path=db_path_obj, readonly=True)
    person_database = get_person_database(db_path=db_path_obj, readonly=False)

    # Step 1: Rebuild reverse_person_orgs from org record JSON
    click.echo("\n=== Step 1: Rebuild reverse_person_orgs from org records ===", err=True)
    reverse_person_orgs: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]] = {}
    orgs_with_execs = 0
    total_exec_entries = 0

    for org_record in org_database.iter_records(source="wikipedia"):
        record_data = org_record.record or {}
        executives = record_data.get("executives")
        if not executives:
            continue
        orgs_with_execs += 1
        org_qid = org_record.source_id
        for exec_entry in executives:
            person_qid = exec_entry.get("person_qid", "")
            if not person_qid:
                continue
            role = exec_entry.get("role", "")
            start_date = exec_entry.get("start_date")
            end_date = exec_entry.get("end_date")
            reverse_person_orgs.setdefault(person_qid, []).append(
                (org_qid, role, start_date, end_date)
            )
            total_exec_entries += 1

    click.echo(
        f"  Found {orgs_with_execs:,} orgs with executives, "
        f"{total_exec_entries:,} total exec entries, "
        f"{len(reverse_person_orgs):,} unique people",
        err=True,
    )

    if not reverse_person_orgs:
        click.echo("  No executives found in org records. Use 'fix-resume --dump <path>' for older databases.", err=True)
        org_database.close()
        person_database.close()
        return

    # Step 2: Backfill known_for_org_id
    click.echo("\n=== Step 2: Backfill known_for_org FKs ===", err=True)
    if reverse_person_orgs:
        # Pass org QIDs directly — backfill resolves to org FKs
        backfilled = person_database.backfill_known_for_org(reverse_person_orgs)
        click.echo(f"  Updated {backfilled:,} people records", err=True)
    else:
        click.echo("  No reverse mappings to backfill", err=True)

    # Step 4: Insert discovered orgs referenced by people but not in orgs table
    click.echo("\n=== Step 4: Check for missing discovered orgs ===", err=True)

    existing_org_ids = org_database.get_all_source_ids(source="wikipedia")
    org_database_rw = get_database(db_path=db_path_obj, readonly=False)
    conn = org_database_rw._connect()

    # Collect org QIDs from people records
    missing_orgs = 0
    for person_record in person_database.iter_records(source="wikidata"):
        record_data = person_record.record or {}
        org_qid = record_data.get("org_qid", "")
        if org_qid and org_qid not in existing_org_ids:
            _insert_discovered_org(conn, org_qid, org_qid, "repair_resume")
            existing_org_ids.add(org_qid)
            missing_orgs += 1
            if missing_orgs % 10000 == 0:
                conn.commit()

    conn.commit()
    click.echo(f"  Inserted {missing_orgs:,} missing discovered orgs", err=True)

    # Step 5: Resolve country_id and region_id FKs
    click.echo("\n=== Step 5: Resolve country/region FKs ===", err=True)
    _resolve_all_fks(db_path_obj)

    org_database.close()
    org_database_rw.close()
    person_database.close()
    click.echo("\nRepair complete!", err=True)


def _resolve_all_fks(db_path: Path) -> None:
    """Resolve country_id on people and region_id on orgs from record JSON."""
    import json as _json
    from corp_entity_db.store import get_person_database, get_database

    person_database = get_person_database(db_path=db_path, readonly=False)
    org_database = get_database(db_path=db_path, readonly=False)

    # Org FKs: read country_qid from record column, resolve to region_id
    conn = org_database._connect()
    org_fks: dict[int, dict] = {}
    cursor = conn.execute(
        "SELECT qid, record FROM organizations WHERE qid IS NOT NULL AND region_id IS NULL"
    )
    for row in cursor:
        rec = _json.loads(row[1]) if row[1] else {}
        country_qid = rec.get("country_qid", "")
        if country_qid:
            org_fks[row[0]] = {"country_qid": country_qid}
    if org_fks:
        click.echo(f"  Resolving region_id for {len(org_fks):,} orgs...", err=True)
        resolved = org_database.resolve_fks(org_fks)
        click.echo(f"  Resolved {resolved:,} org region_ids", err=True)
    else:
        click.echo("  No org region FKs to resolve", err=True)

    # People FKs: resolve country_id, known_for_org_id, and known_for_role_id from record column
    click.echo("  Resolving people FKs (country_id, known_for_org, known_for_role)...", err=True)
    resolved = person_database.resolve_fks()
    click.echo(f"  Resolved {resolved:,} person FKs", err=True)

    org_database.close()
    person_database.close()


@click.command("backfill-locations")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata JSON dump file (.bz2 or .gz)")
@click.option("--download", is_flag=True, help="Download latest dump first (~100GB)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Max location records to import")
@click.option("--batch-size", type=int, default=10000, help="Batch size for commits (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_backfill_locations(
    dump_path: Optional[str],
    download: bool,
    db_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import locations from Wikidata dump into an existing database.

    For databases imported before locations were included in phase 1, this
    command reads the full Wikidata dump and imports all location records
    (countries, cities, regions, etc.) that are not already present.

    After importing locations, resolves country_id and region_id FKs on
    people and org records.

    \b
    Steps:
    1. Stream Wikidata dump, importing all location entities (skipping existing)
    2. Resolve location parent_ids from record JSON
    3. Resolve country/region FKs on people and orgs

    \b
    Examples:
        corp-entity-db backfill-locations --dump /path/to/dump.json.bz2
        corp-entity-db backfill-locations --download
        corp-entity-db backfill-locations --dump dump.json.bz2 --limit 50000
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_locations_database
    from corp_entity_db.importers.wikidata_dump import WikidataDumpImporter

    if not dump_path and not download:
        raise click.UsageError("Either --dump path or --download is required")

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    importer = WikidataDumpImporter(dump_path=dump_path)

    if download:
        click.echo("Downloading Wikidata dump (~100GB)...", err=True)
        dump_file = importer.download_dump(force=False, use_aria2=True)
        click.echo(f"Using dump: {dump_file}", err=True)
    elif dump_path:
        click.echo(f"Using dump: {dump_path}", err=True)

    # Load existing location source IDs to skip
    locations_database = get_locations_database(db_path=db_path_obj, readonly=False)
    existing_ids = locations_database.get_all_source_ids(source="wikidata")
    click.echo(f"Existing locations: {len(existing_ids):,} (will skip)", err=True)

    # Step 1: Import locations from dump
    click.echo("\n=== Step 1: Import locations from dump ===", err=True)
    if limit:
        click.echo(f"  Limit: {limit:,} records", err=True)

    location_batch: list = []
    locations_inserted = 0

    for record in importer.import_locations(
        limit=limit,
        require_enwiki=False,
        skip_ids=existing_ids,
    ):
        location_batch.append(record)
        if len(location_batch) >= batch_size:
            inserted = locations_database.insert_batch(location_batch)
            locations_inserted += inserted
            location_batch = []
            click.echo(
                f"\r  Imported {locations_inserted:,} locations...",
                nl=False, err=True,
            )
            sys.stderr.flush()

    # Flush remaining
    if location_batch:
        inserted = locations_database.insert_batch(location_batch)
        locations_inserted += inserted

    click.echo(f"\n  Imported {locations_inserted:,} locations total", err=True)

    # Step 2: Resolve location parent_ids
    click.echo("\n=== Step 2: Resolve location parent_ids ===", err=True)
    loc_conn = locations_database._connect()
    location_fks: dict[int, dict] = {}
    cursor = loc_conn.execute(
        "SELECT qid, record FROM locations WHERE qid IS NOT NULL AND source_id = 4"
    )
    for row in cursor:
        rec = json.loads(row[1]) if row[1] else {}
        parent_qids = rec.get("parent_qids", [])
        country_qids = rec.get("country_qids", [])
        if parent_qids or country_qids:
            location_fks[row[0]] = {"parent_qids": parent_qids, "country_qids": country_qids}

    if location_fks:
        click.echo(f"  Resolving parent_ids for {len(location_fks):,} locations...", err=True)
        resolved = locations_database.resolve_parent_ids(location_fks)
        click.echo(f"  Resolved {resolved:,} location parent_ids", err=True)
    else:
        click.echo("  No location parent_ids to resolve", err=True)

    locations_database.close()

    # Step 3: Resolve country/region FKs on people and orgs
    click.echo("\n=== Step 3: Resolve country/region FKs ===", err=True)
    _resolve_all_fks(db_path_obj)

    click.echo("\nBackfill complete!", err=True)


@click.command("backfill-roles")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata JSON dump file (.bz2 or .gz)")
@click.option("--download", is_flag=True, help="Download latest dump first (~100GB)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Max role records to import")
@click.option("--batch-size", type=int, default=10000, help="Batch size for commits (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_backfill_roles(
    dump_path: Optional[str],
    download: bool,
    db_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import role/position entities from Wikidata dump into an existing database.

    For databases imported before roles were included in phase 1, this
    command reads the full Wikidata dump and imports all role/position entities
    (e.g., President, CEO, Minister) that are not already present.

    After importing roles, resolves known_for_role_id FKs on people records.

    \b
    Steps:
    1. Stream Wikidata dump, importing all role entities (skipping existing)
    2. Resolve known_for_role_id FKs on people records

    \b
    Examples:
        corp-entity-db backfill-roles --dump /path/to/dump.json.bz2
        corp-entity-db backfill-roles --download
        corp-entity-db backfill-roles --dump dump.json.bz2 --limit 50000
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_roles_database
    from corp_entity_db.importers.wikidata_dump import WikidataDumpImporter

    if not dump_path and not download:
        raise click.UsageError("Either --dump path or --download is required")

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    importer = WikidataDumpImporter(dump_path=dump_path)

    if download:
        click.echo("Downloading Wikidata dump (~100GB)...", err=True)
        dump_file = importer.download_dump(force=False, use_aria2=True)
        click.echo(f"Using dump: {dump_file}", err=True)
    elif dump_path:
        click.echo(f"Using dump: {dump_path}", err=True)

    # Load existing role source IDs to skip
    roles_database = get_roles_database(db_path=db_path_obj, readonly=False)
    existing_ids = roles_database.get_all_source_ids(source="wikidata")
    click.echo(f"Existing roles: {len(existing_ids):,} (will skip)", err=True)

    # Step 1: Import roles from dump
    click.echo("\n=== Step 1: Import roles from dump ===", err=True)
    if limit:
        click.echo(f"  Limit: {limit:,} records", err=True)

    role_batch: list = []
    roles_inserted = 0
    roles_scanned = 0

    for record_type, record in importer.import_all(
        import_people=False,
        import_orgs=False,
        import_locations=False,
    ):
        if record_type != "role":
            continue
        roles_scanned += 1
        if limit and roles_scanned > limit:
            break
        # Skip existing
        if record.source_id and record.source_id in existing_ids:
            continue
        role_batch.append(record)
        if len(role_batch) >= batch_size:
            inserted = roles_database.insert_batch(role_batch)
            roles_inserted += inserted
            role_batch = []
            click.echo(
                f"\r  Imported {roles_inserted:,} roles...",
                nl=False, err=True,
            )
            sys.stderr.flush()

    # Flush remaining
    if role_batch:
        inserted = roles_database.insert_batch(role_batch)
        roles_inserted += inserted

    click.echo(f"\n  Imported {roles_inserted:,} roles total", err=True)
    roles_database.close()

    # Step 2: Resolve known_for_role_id FKs on people records
    click.echo("\n=== Step 2: Resolve known_for_role_id FKs ===", err=True)
    _resolve_all_fks(db_path_obj)

    click.echo("\nBackfill complete!", err=True)


@click.command("fix-resume")
@click.option("--dump", "dump_path", type=click.Path(exists=True), help="Path to Wikidata dump file (required for steps 1-3)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--skip-record-update", is_flag=True, help="Skip updating org records with executives (just do people backfill)")
@click.option("--from-step", type=int, default=1, help="Start from this step (1-4). Use 4 to just insert discovered orgs.")
@click.option("--limit", type=int, default=0, help="Limit records processed per step (0 = unlimited, for testing)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_fix_resume(dump_path: Optional[str], db_path: Optional[str], skip_record_update: bool, from_step: int, limit: int, verbose: bool):
    """
    Fix people records by re-scanning a Wikidata dump file.

    Creates additional people records from org→person executive mappings
    (e.g. Microsoft P169→Bill Gates creates a "Bill Gates, CEO, Microsoft" record),
    and backfills organization references for people records.

    This scans the full dump (~100GB) — step 1 is JSON-only, steps 3b/3c need
    the embedder for new/updated records.

    Use --from-step 4 to skip the dump scan and just insert discovered orgs
    (reads org_qid from people record JSON — no dump needed).

    \b
    Steps:
    1. Scan dump: extract executive properties from orgs, org refs from people
    2. Optionally update org records with executives field (for future repair-resume)
    3. Resolve org QIDs to labels and backfill people missing known_for_org
    3b. Insert additional people records from reverse org→person mappings (with embeddings)
    4. Insert discovered orgs referenced by people but not in orgs table

    \b
    Examples:
        corp-entity-db fix-resume --dump /path/to/wikidata-latest-all.json.bz2
        corp-entity-db fix-resume --dump dump.json.bz2 --skip-record-update
        corp-entity-db fix-resume --dump dump.json.bz2 --db /path/to/entities.db
        corp-entity-db fix-resume --from-step 4       # Skip dump scan, just insert missing orgs
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database
    from corp_entity_db.importers.wikidata_dump import WikidataDumpImporter

    if from_step < 4 and not dump_path:
        raise click.UsageError("--dump is required for steps 1-3. Use --from-step 4 to skip the dump scan.")

    if from_step not in (1, 2, 3, 4):
        raise click.UsageError("--from-step must be 1, 2, 3, or 4")

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)
    if dump_path:
        click.echo(f"Dump: {dump_path}", err=True)
    if from_step > 1:
        click.echo(f"Starting from step {from_step}", err=True)
    if limit:
        click.echo(f"Limit: {limit:,} records per step", err=True)

    importer = WikidataDumpImporter(dump_path=dump_path) if dump_path else None

    # org_qid → list of executive dicts (populated in step 1)
    org_executives: dict[str, list[dict]] = {}
    # person_qid → [(org_qid, role, start_date, end_date)] (populated in step 1)
    reverse_person_orgs: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]] = {}
    # org_qid → label (discovered from people, populated in step 1)
    discovered_orgs: dict[str, str] = {}

    # ==================== Step 1: Scan dump ====================
    if from_step <= 1 and importer is not None:
        click.echo("\n=== Step 1: Scanning dump for executive properties ===", err=True)
        click.echo("  This scans the full dump (~100GB) — parsing JSON only, no embeddings.", err=True)
        click.echo("  Using prefetch thread to overlap decompression with processing.", err=True)
        sys.stderr.flush()

        entities_scanned = 0
        orgs_with_execs = 0
        people_with_orgs = 0

        executive_props = [
            ("P169", "chief executive officer"),
            ("P488", "chairperson"),
            ("P112", "founded by"),
            ("P1037", "director/manager"),
            ("P3320", "board member"),
        ]

        # Prefetch entities in background thread — bz2 decompression releases the GIL,
        # so decompression + JSON parsing overlap with property extraction in main thread
        entity_queue: queue.Queue = queue.Queue(maxsize=10_000)
        prefetch_error: list[Exception] = []
        stop_prefetch = threading.Event()

        def _prefetch_entities():
            try:
                for ent in importer.iter_entities():
                    if stop_prefetch.is_set():
                        return
                    entity_queue.put(ent)
            except Exception as e:
                prefetch_error.append(e)
            finally:
                entity_queue.put(None)

        prefetch_t = threading.Thread(target=_prefetch_entities, daemon=True, name="scan-prefetch")
        prefetch_t.start()

        while True:
            entity = entity_queue.get()
            if entity is None:
                break
            entities_scanned += 1
            if limit and entities_scanned >= limit:
                stop_prefetch.set()
                break
            if entities_scanned % 1_000_000 == 0:
                click.echo(
                    f"\r  Scanned {entities_scanned:,} entities "
                    f"({orgs_with_execs:,} orgs w/execs, {people_with_orgs:,} people w/orgs)...",
                    nl=False, err=True,
                )
                sys.stderr.flush()

            if entity.get("type") != "item":
                continue

            qid = entity.get("id", "")
            claims = entity.get("claims", {})

            # Check if org
            if importer._get_org_type(entity) is not None:
                executives_list: list[dict] = []
                for prop, role_desc in executive_props:
                    for claim in claims.get(prop, []):
                        mainsnak = claim.get("mainsnak", {})
                        person_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                        if not person_qid:
                            continue
                        qualifiers = claim.get("qualifiers", {})
                        start_date = importer._get_time_qualifier(qualifiers, "P580")
                        end_date = importer._get_time_qualifier(qualifiers, "P582")
                        reverse_person_orgs.setdefault(person_qid, []).append(
                            (qid, role_desc, start_date, end_date)
                        )
                        executives_list.append({
                            "person_qid": person_qid,
                            "role": role_desc,
                            "start_date": start_date,
                            "end_date": end_date,
                        })
                if executives_list:
                    org_executives[qid] = executives_list
                    orgs_with_execs += 1

            # Check if person (P31=Q5)
            elif importer._is_human(entity):
                # Extract org_qid from position held (P39) qualifiers
                for claim in claims.get("P39", []):
                    qualifiers = claim.get("qualifiers", {})
                    for q_claim in qualifiers.get("P642", []):
                        org_qid = q_claim.get("datavalue", {}).get("value", {}).get("id")
                        if org_qid:
                            discovered_orgs[org_qid] = discovered_orgs.get(org_qid, org_qid)
                            people_with_orgs += 1
                            break
                # Also check P108 employer
                for claim in claims.get("P108", []):
                    mainsnak = claim.get("mainsnak", {})
                    org_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                    if org_qid:
                        discovered_orgs[org_qid] = discovered_orgs.get(org_qid, org_qid)

            # Cache position item jurisdictions (P1001/P17) for all entities
            # Cheap check — only caches entities that have these properties
            importer._cache_position_jurisdiction(entity)

        # Drain queue so prefetch thread isn't stuck on a full put()
        if stop_prefetch.is_set():
            while not entity_queue.empty():
                try:
                    entity_queue.get_nowait()
                except queue.Empty:
                    break
        prefetch_t.join(timeout=5)
        if prefetch_error:
            raise prefetch_error[0]

        click.echo("", err=True)  # Newline after counter
        click.echo(
            f"  Scan complete: {entities_scanned:,} entities, "
            f"{orgs_with_execs:,} orgs with executives, "
            f"{len(reverse_person_orgs):,} unique people in reverse map, "
            f"{len(discovered_orgs):,} discovered orgs",
            err=True,
        )
    elif from_step <= 1:
        click.echo("\n=== Step 1: Skipped (no dump path) ===", err=True)

    # ==================== Step 2: Backfill org records ====================
    if from_step <= 2 and not skip_record_update and org_executives:
        click.echo(f"\n=== Step 2: Backfill executives into {len(org_executives):,} org records ===", err=True)
        org_database = get_database(db_path=db_path_obj, readonly=False)
        conn = org_database._connect()
        updated_orgs = 0

        for org_qid, exec_list in org_executives.items():
            # Read existing record
            org_rec = org_database.get_by_source_id("wikipedia", org_qid)
            if not org_rec:
                continue
            record_data = dict(org_rec.record or {})
            if "executives" in record_data:
                continue  # Already has executives, skip
            record_data["executives"] = exec_list
            # Update record JSON in DB
            from corp_entity_db.store import SOURCE_NAME_TO_ID
            source_type_id = SOURCE_NAME_TO_ID.get("wikipedia", 4)
            conn.execute(
                "UPDATE organizations SET record = ? WHERE source_id = ? AND source_identifier = ?",
                (json.dumps(record_data), source_type_id, org_qid),
            )
            updated_orgs += 1
            if updated_orgs % 10000 == 0:
                conn.commit()
                click.echo(f"\r  Updated {updated_orgs:,} org records...", nl=False, err=True)
                sys.stderr.flush()

        conn.commit()
        click.echo(f"\n  Updated {updated_orgs:,} org records with executives field", err=True)
        org_database.close()
    elif from_step <= 2 and skip_record_update:
        click.echo("\n=== Step 2: Skipped (--skip-record-update) ===", err=True)
    elif from_step <= 2:
        click.echo("\n=== Step 2: No org executives found to backfill ===", err=True)

    # ==================== Step 3: Backfill known_for_org FKs ====================
    if from_step <= 3 and reverse_person_orgs:
        click.echo("\n=== Step 3: Backfill known_for_org FKs ===", err=True)
        person_database = get_person_database(db_path=db_path_obj, readonly=False)

        # Pass org QIDs directly — backfill resolves to org/location FKs
        backfilled = person_database.backfill_known_for_org(reverse_person_orgs)
        click.echo(f"  Updated {backfilled:,} people records", err=True)

        person_database.close()
    elif from_step <= 3:
        click.echo("\n=== Step 3: No reverse mappings to backfill ===", err=True)

    # ==================== Step 3b: Insert additional records from reverse mappings ====================
    if from_step <= 3 and reverse_person_orgs:
        click.echo("\n=== Step 3b: Insert additional people records from reverse mappings ===", err=True)
        click.echo("  Creates new records for people who have org mappings they lack records for.", err=True)

        from corp_entity_db.store import get_person_database, get_database, get_roles_database
        from corp_entity_db.embeddings import CompanyEmbedder
        import numpy as np

        person_database = get_person_database(db_path=db_path_obj, readonly=False)
        org_database = get_database(db_path=db_path_obj, readonly=True)
        embedder = CompanyEmbedder()

        conn = person_database._connect()
        roles_db = get_roles_database(db_path=db_path_obj, readonly=False)

        # Phase 1: Collect all records that need inserting (no embedding yet — fast)
        click.echo("  Phase 1: Collecting records to insert...", err=True)
        pending: list[dict] = []
        skipped = 0
        people_checked = 0

        for person_qid, entries in reverse_person_orgs.items():
            people_checked += 1
            if people_checked % 50_000 == 0:
                click.echo(
                    f"\r  Checked {people_checked:,} people, {len(pending):,} pending...",
                    nl=False, err=True,
                )
                sys.stderr.flush()

            qid_int = int(person_qid.lstrip("Q")) if person_qid.startswith("Q") else None
            if qid_int is None:
                continue

            existing = conn.execute(
                "SELECT known_for_org_id FROM people WHERE qid = ?", (qid_int,)
            ).fetchall()
            if not existing:
                continue  # Person not in DB at all, skip

            # Track existing org FK values to avoid duplicates
            existing_fks = {row[0] for row in existing}
            base_row = conn.execute(
                "SELECT name, country_id, person_type_id, birth_date, death_date, record FROM people WHERE qid = ? LIMIT 1",
                (qid_int,),
            ).fetchone()
            if not base_row:
                continue

            base_name = base_row[0]
            base_country_id = base_row[1]
            base_person_type_id = base_row[2]
            base_birth_date = base_row[3] or ""
            base_death_date = base_row[4] or ""
            base_record_json = base_row[5] or "{}"

            for rev_org_qid, rev_role, rev_start, rev_end in entries:
                # Resolve org QID to org_id (wikidata→wikidata)
                resolved_org_id = org_database.get_id_by_source_id("wikipedia", rev_org_qid)

                if resolved_org_id is None:
                    skipped += 1
                    continue

                # Skip if this person already has a record with this org FK
                if resolved_org_id in existing_fks:
                    continue

                # Get org label for embedding text
                org_rec = org_database.get_by_source_id("wikipedia", rev_org_qid)
                org_label = org_rec.name if org_rec else ""
                if not org_label:
                    skipped += 1
                    continue

                embed_text = f"{base_name}, {rev_role}, {org_label}" if rev_role else f"{base_name}, {org_label}"
                role_id = roles_db.get_or_create(rev_role, source_id=4) if rev_role else None

                try:
                    record_data = json.loads(base_record_json) if isinstance(base_record_json, str) else {}
                except (json.JSONDecodeError, TypeError):
                    record_data = {}
                record_data["org_qid"] = rev_org_qid
                record_data["role_from_reverse"] = rev_role

                pending.append({
                    "qid_int": qid_int,
                    "name": base_name,
                    "name_normalized": base_name.lower().strip(),
                    "person_qid": person_qid,
                    "country_id": base_country_id,
                    "person_type_id": base_person_type_id,
                    "role_id": role_id,
                    "org_id": resolved_org_id,
                    "from_date": rev_start or "",
                    "to_date": rev_end or "",
                    "birth_date": base_birth_date,
                    "death_date": base_death_date,
                    "record_json": json.dumps(record_data),
                    "embed_text": embed_text,
                })
                existing_fks.add(resolved_org_id)

        click.echo(f"\n  Collected {len(pending):,} records to insert ({skipped:,} skipped, no org label)", err=True)

        # Phase 2: Batch embed (background thread) and insert (main thread) in parallel.
        # PyTorch releases the GIL during forward pass, so embedding overlaps with DB writes.
        if pending:
            click.echo("  Phase 2: Batch embedding and inserting (2 threads)...", err=True)
            EMBED_BATCH = 192
            new_records = 0

            embed_q: queue.Queue = queue.Queue(maxsize=2)
            result_q: queue.Queue = queue.Queue(maxsize=2)
            embed_errors: list[Exception] = []

            def _embed_worker_3b():
                try:
                    while True:
                        texts = embed_q.get()
                        if texts is None:
                            return
                        result_q.put(embedder.embed_batch(texts))
                except Exception as e:
                    embed_errors.append(e)
                    result_q.put(None)

            embed_t = threading.Thread(target=_embed_worker_3b, daemon=True, name="fix-embedder-3b")
            embed_t.start()

            batches = [pending[i:i + EMBED_BATCH] for i in range(0, len(pending), EMBED_BATCH)]

            # Submit first batch to start the pipeline
            embed_q.put([r["embed_text"] for r in batches[0]])

            for batch_idx, batch in enumerate(batches):
                # Get current batch embeddings (blocks until embedder finishes)
                fp32_embs = result_q.get()
                if fp32_embs is None:
                    raise embed_errors[0]

                # Submit NEXT batch immediately — embeds while we write to DB below
                if batch_idx + 1 < len(batches):
                    embed_q.put([r["embed_text"] for r in batches[batch_idx + 1]])

                # Write current batch to DB (overlaps with next batch embedding)
                for i, rec in enumerate(batch):
                    cursor = conn.execute(
                        """INSERT INTO people
                        (qid, name, name_normalized, source_id, source_identifier, country_id,
                         person_type_id, known_for_role_id, known_for_org_id,
                         from_date, to_date, birth_date, death_date, record, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            rec["qid_int"], rec["name"], rec["name_normalized"],
                            4, rec["person_qid"], rec["country_id"],
                            rec["person_type_id"], rec["role_id"],
                            rec.get("org_id"),
                            rec["from_date"], rec["to_date"],
                            rec["birth_date"], rec["death_date"],
                            rec["record_json"],
                            fp32_embs[i].astype(np.float32).tobytes(),
                        ),
                    )
                    new_records += 1

                conn.commit()
                click.echo(
                    f"\r  Embedded and inserted {new_records:,}/{len(pending):,} records...",
                    nl=False, err=True,
                )
                sys.stderr.flush()

            embed_q.put(None)  # shutdown embedder
            embed_t.join()

            click.echo(f"\n  Inserted {new_records:,} new people records", err=True)

        org_database.close()
        person_database.close()

    # ==================== Step 4: Insert discovered orgs ====================
    click.echo("\n=== Step 4: Insert missing discovered orgs ===", err=True)
    person_database = get_person_database(db_path=db_path_obj, readonly=True)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    conn = org_database._connect()

    existing_org_ids = org_database.get_all_source_ids(source="wikipedia")
    click.echo(f"  Existing orgs: {len(existing_org_ids):,}", err=True)

    # Collect org QIDs from people record JSON (works with or without dump scan)
    missing_orgs = 0
    people_scanned = 0
    for person_record in person_database.iter_records(source="wikidata"):
        record_data = person_record.record or {}
        org_qid = record_data.get("org_qid", "")
        if org_qid and org_qid not in existing_org_ids:
            org_label = org_qid
            _insert_discovered_org(conn, org_qid, org_label, "fix_resume")
            existing_org_ids.add(org_qid)
            missing_orgs += 1
            if missing_orgs % 10000 == 0:
                conn.commit()
        people_scanned += 1
        if people_scanned % 1_000_000 == 0:
            click.echo(f"\r  Scanned {people_scanned:,} people, {missing_orgs:,} new orgs...", nl=False, err=True)
            sys.stderr.flush()

    # Also insert from dump-discovered orgs (if step 1 was run)
    for org_qid in discovered_orgs:
        if org_qid not in existing_org_ids:
            org_label = org_qid
            _insert_discovered_org(conn, org_qid, org_label, "fix_resume")
            existing_org_ids.add(org_qid)
            missing_orgs += 1

    conn.commit()
    click.echo(f"\n  Inserted {missing_orgs:,} missing discovered orgs (scanned {people_scanned:,} people)", err=True)

    org_database.close()
    person_database.close()
    click.echo("\nFix complete!", err=True)
