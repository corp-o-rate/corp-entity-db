"""Database repair commands for fixing people records after resumed Wikidata imports."""

import json
import queue
import sys
import threading
from typing import Optional

import click

from ._common import _configure_logging, _resolve_db_path


def _insert_discovered_org(
    conn,
    is_v2: Optional[bool],
    org_qid: str,
    org_label: str,
    discovered_from: str,
) -> None:
    """Insert a single discovered org via raw SQL (no embeddings needed).

    Embeddings will be generated later by ``db post-import``.
    """
    name_normalized = org_label.lower().strip()
    record_json = json.dumps({
        "wikidata_id": org_qid,
        "discovered_from": discovered_from,
        "needs_label_resolution": org_label == org_qid,
    })
    if is_v2:
        from corp_entity_db.seed_data import SOURCE_NAME_TO_ID, ORG_TYPE_NAME_TO_ID
        source_type_id = SOURCE_NAME_TO_ID.get("wikipedia", 4)
        entity_type_id = ORG_TYPE_NAME_TO_ID.get("business", 17)
        conn.execute("""
            INSERT OR IGNORE INTO organizations
            (name, name_normalized, source_id, source_identifier, entity_type_id, from_date, to_date, record)
            VALUES (?, ?, ?, ?, ?, '', '', ?)
        """, (org_label, name_normalized, source_type_id, org_qid, entity_type_id, record_json))
    else:
        conn.execute("""
            INSERT OR IGNORE INTO organizations
            (name, name_normalized, source, source_id, region, entity_type, from_date, to_date, record)
            VALUES (?, ?, 'wikipedia', ?, '', 'business', '', '', ?)
        """, (org_label, name_normalized, org_qid, record_json))


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

    # Step 2: Resolve org QIDs to labels
    click.echo("\n=== Step 2: Resolve org QIDs to labels ===", err=True)
    all_labels = person_database.get_all_qid_labels()
    click.echo(f"  Loaded {len(all_labels):,} QID labels from DB", err=True)

    resolved_map: dict[str, list[tuple[str, str, str | None, str | None]]] = {}
    for person_qid, entries in reverse_person_orgs.items():
        resolved_entries: list[tuple[str, str, str | None, str | None]] = []
        for org_qid, role, start_date, end_date in entries:
            org_label = all_labels.get(org_qid, "")
            # Fallback: look up org name from organizations table
            if not org_label:
                org_rec = org_database.get_by_source_id("wikipedia", org_qid)
                if org_rec:
                    org_label = org_rec.name
            if org_label:
                resolved_entries.append((org_label, role, start_date, end_date))
        if resolved_entries:
            resolved_map[person_qid] = resolved_entries

    click.echo(f"  Resolved {len(resolved_map):,} people with org labels", err=True)

    # Step 3: Backfill people
    click.echo("\n=== Step 3: Backfill known_for_org ===", err=True)
    if resolved_map:
        backfilled = person_database.backfill_known_for_org(resolved_map)
        click.echo(f"  Updated {backfilled:,} people records", err=True)
    else:
        click.echo("  No resolved mappings to backfill", err=True)

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
            org_label = all_labels.get(org_qid, org_qid)
            _insert_discovered_org(conn, org_database_rw._is_v2, org_qid, org_label, "repair_resume")
            existing_org_ids.add(org_qid)
            missing_orgs += 1
            if missing_orgs % 10000 == 0:
                conn.commit()

    conn.commit()
    click.echo(f"  Inserted {missing_orgs:,} missing discovered orgs", err=True)

    org_database.close()
    org_database_rw.close()
    person_database.close()
    click.echo("\nRepair complete!", err=True)


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
    and backfills position jurisdictions (e.g. "President of the United States"
    gets known_for_org="United States").

    This scans the full dump (~100GB) — step 1 is JSON-only, steps 3b/3c need
    the embedder for new/updated records.

    Use --from-step 4 to skip the dump scan and just insert discovered orgs
    (reads org_qid from people record JSON — no dump needed).

    \b
    Steps:
    1. Scan dump: extract executive properties from orgs, position jurisdictions, org refs from people
    2. Optionally update org records with executives field (for future repair-resume)
    3. Resolve org QIDs to labels and backfill people missing known_for_org
    3b. Insert additional people records from reverse org→person mappings (with embeddings)
    3c. Backfill position jurisdictions for people with empty known_for_org (re-embeds)
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
    # position_qid → jurisdiction_qid (from position items with P1001/P17)
    position_jurisdictions: dict[str, str] = {}

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
        position_jurisdictions = importer._position_jurisdictions
        click.echo(
            f"  Scan complete: {entities_scanned:,} entities, "
            f"{orgs_with_execs:,} orgs with executives, "
            f"{len(reverse_person_orgs):,} unique people in reverse map, "
            f"{len(discovered_orgs):,} discovered orgs, "
            f"{len(position_jurisdictions):,} position jurisdictions cached",
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
            if org_database._is_v2:
                from corp_entity_db.store import SOURCE_NAME_TO_ID
                source_type_id = SOURCE_NAME_TO_ID.get("wikipedia", 4)
                conn.execute(
                    "UPDATE organizations SET record = ? WHERE source_id = ? AND source_identifier = ?",
                    (json.dumps(record_data), source_type_id, org_qid),
                )
            else:
                conn.execute(
                    "UPDATE organizations SET record = ? WHERE source = 'wikipedia' AND source_id = ?",
                    (json.dumps(record_data), org_qid),
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

    # ==================== Step 3: Resolve and backfill people ====================
    if from_step <= 3 and reverse_person_orgs:
        click.echo("\n=== Step 3: Resolve and backfill known_for_org ===", err=True)
        person_database = get_person_database(db_path=db_path_obj, readonly=False)
        org_database = get_database(db_path=db_path_obj, readonly=True)

        # Load labels from DB and importer cache
        all_labels = person_database.get_all_qid_labels()
        if importer:
            all_labels.update(importer.get_label_cache(copy=False))
        click.echo(f"  Loaded {len(all_labels):,} QID labels", err=True)

        resolved_map: dict[str, list[tuple[str, str, str | None, str | None]]] = {}
        for person_qid, entries in reverse_person_orgs.items():
            resolved_entries: list[tuple[str, str, str | None, str | None]] = []
            for org_qid, role, start_date, end_date in entries:
                org_label = all_labels.get(org_qid, "")
                if not org_label:
                    org_rec = org_database.get_by_source_id("wikipedia", org_qid)
                    if org_rec:
                        org_label = org_rec.name
                if org_label:
                    resolved_entries.append((org_label, role, start_date, end_date))
            if resolved_entries:
                resolved_map[person_qid] = resolved_entries

        click.echo(f"  Resolved {len(resolved_map):,} people with org labels", err=True)

        if resolved_map:
            backfilled = person_database.backfill_known_for_org(resolved_map)
            click.echo(f"  Updated {backfilled:,} people records", err=True)
        else:
            click.echo("  No resolved mappings to backfill", err=True)

        org_database.close()
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

        all_labels = person_database.get_all_qid_labels()
        if importer:
            all_labels.update(importer.get_label_cache(copy=False))

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
                "SELECT known_for_org FROM people WHERE qid = ?", (qid_int,)
            ).fetchall()
            if not existing:
                continue  # Person not in DB at all, skip

            existing_orgs = {row[0].lower() for row in existing if row[0]}
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
                org_label = all_labels.get(rev_org_qid, "")
                if not org_label:
                    org_rec = org_database.get_by_source_id("wikipedia", rev_org_qid)
                    if org_rec:
                        org_label = org_rec.name
                if not org_label:
                    skipped += 1
                    continue

                if org_label.lower() in existing_orgs:
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
                    "org_label": org_label,
                    "from_date": rev_start or "",
                    "to_date": rev_end or "",
                    "birth_date": base_birth_date,
                    "death_date": base_death_date,
                    "record_json": json.dumps(record_data),
                    "embed_text": embed_text,
                })
                existing_orgs.add(org_label.lower())

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
                        result_q.put(embedder.embed_batch_and_quantize(texts))
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
                emb_result = result_q.get()
                if emb_result is None:
                    raise embed_errors[0]
                fp32_embs, int8_embs = emb_result

                # Submit NEXT batch immediately — embeds while we write to DB below
                if batch_idx + 1 < len(batches):
                    embed_q.put([r["embed_text"] for r in batches[batch_idx + 1]])

                # Write current batch to DB (overlaps with next batch embedding)
                for i, rec in enumerate(batch):
                    cursor = conn.execute(
                        """INSERT INTO people
                        (qid, name, name_normalized, source_id, source_identifier, country_id,
                         person_type_id, known_for_role_id, known_for_org, from_date, to_date,
                         birth_date, death_date, record)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            rec["qid_int"], rec["name"], rec["name_normalized"],
                            4, rec["person_qid"], rec["country_id"],
                            rec["person_type_id"], rec["role_id"], rec["org_label"],
                            rec["from_date"], rec["to_date"],
                            rec["birth_date"], rec["death_date"],
                            rec["record_json"],
                        ),
                    )
                    row_id = cursor.lastrowid
                    assert row_id is not None

                    conn.execute(
                        "INSERT INTO person_embeddings (person_id, embedding) VALUES (?, ?)",
                        (row_id, fp32_embs[i].astype(np.float32).tobytes()),
                    )
                    conn.execute(
                        "INSERT INTO person_embeddings_scalar (person_id, embedding) VALUES (?, vec_int8(?))",
                        (row_id, int8_embs[i].astype(np.int8).tobytes()),
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

    # Also handle position jurisdictions: backfill P39 positions that lacked org qualifiers
    if from_step <= 3 and position_jurisdictions and importer:
        click.echo("\n=== Step 3c: Backfill people with position jurisdictions ===", err=True)
        click.echo(f"  Using {len(position_jurisdictions):,} cached position→jurisdiction mappings", err=True)

        from corp_entity_db.store import get_person_database, get_roles_database
        from corp_entity_db.embeddings import CompanyEmbedder
        import numpy as np

        person_database = get_person_database(db_path=db_path_obj, readonly=False)
        embedder = CompanyEmbedder()
        conn = person_database._connect()

        all_labels = person_database.get_all_qid_labels()
        if importer:
            all_labels.update(importer.get_label_cache(copy=False))

        rows = conn.execute(
            "SELECT id, qid, name, record FROM people WHERE (known_for_org = '' OR known_for_org IS NULL) AND source_id = 4"
        ).fetchall()
        click.echo(f"  Found {len(rows):,} people with empty known_for_org", err=True)

        # Phase 1: Collect updates (no embedding yet — fast)
        click.echo("  Phase 1: Resolving jurisdictions...", err=True)
        pending_updates: list[dict] = []
        roles_db = get_roles_database(db_path=db_path_obj, readonly=False)

        for row in rows:
            person_id, qid_int, name, record_json = row
            try:
                record_data = json.loads(record_json) if record_json else {}
            except (json.JSONDecodeError, TypeError):
                continue

            positions = record_data.get("positions", [])
            for pos_qid in positions:
                jur_qid = position_jurisdictions.get(pos_qid, "")
                if jur_qid:
                    jur_label = all_labels.get(jur_qid, "")
                    if jur_label:
                        pos_label = all_labels.get(pos_qid, "")
                        role_id = roles_db.get_or_create(pos_label, source_id=4) if pos_label else None
                        embed_text = f"{name}, {pos_label}, {jur_label}" if pos_label else f"{name}, {jur_label}"
                        pending_updates.append({
                            "person_id": person_id,
                            "jur_label": jur_label,
                            "role_id": role_id,
                            "embed_text": embed_text,
                        })
                        break  # Use first matching jurisdiction
        click.echo(f"  Collected {len(pending_updates):,} records to update", err=True)

        # Phase 2: Batch embed (background thread) and update (main thread) in parallel
        if pending_updates:
            click.echo("  Phase 2: Batch embedding and updating (2 threads)...", err=True)
            EMBED_BATCH = 192
            updated = 0

            embed_q_3c: queue.Queue = queue.Queue(maxsize=2)
            result_q_3c: queue.Queue = queue.Queue(maxsize=2)
            embed_errors_3c: list[Exception] = []

            def _embed_worker_3c():
                try:
                    while True:
                        texts = embed_q_3c.get()
                        if texts is None:
                            return
                        result_q_3c.put(embedder.embed_batch_and_quantize(texts))
                except Exception as e:
                    embed_errors_3c.append(e)
                    result_q_3c.put(None)

            embed_t_3c = threading.Thread(target=_embed_worker_3c, daemon=True, name="fix-embedder-3c")
            embed_t_3c.start()

            batches_3c = [pending_updates[i:i + EMBED_BATCH] for i in range(0, len(pending_updates), EMBED_BATCH)]

            # Submit first batch
            embed_q_3c.put([r["embed_text"] for r in batches_3c[0]])

            for batch_idx, batch in enumerate(batches_3c):
                emb_result = result_q_3c.get()
                if emb_result is None:
                    raise embed_errors_3c[0]
                fp32_embs, int8_embs = emb_result

                # Submit next batch immediately — embeds while we write to DB
                if batch_idx + 1 < len(batches_3c):
                    embed_q_3c.put([r["embed_text"] for r in batches_3c[batch_idx + 1]])

                for i, rec in enumerate(batch):
                    conn.execute(
                        """UPDATE people SET known_for_org = ?,
                           known_for_role_id = COALESCE(known_for_role_id, ?)
                        WHERE id = ?""",
                        (rec["jur_label"], rec["role_id"], rec["person_id"]),
                    )
                    conn.execute(
                        "UPDATE person_embeddings SET embedding = ? WHERE person_id = ?",
                        (fp32_embs[i].tobytes(), rec["person_id"]),
                    )
                    conn.execute(
                        "DELETE FROM person_embeddings_scalar WHERE person_id = ?",
                        (rec["person_id"],),
                    )
                    conn.execute(
                        "INSERT INTO person_embeddings_scalar (person_id, embedding) VALUES (?, vec_int8(?))",
                        (rec["person_id"], int8_embs[i].tobytes()),
                    )
                    updated += 1

                conn.commit()
                click.echo(
                    f"\r  Embedded and updated {updated:,}/{len(pending_updates):,} records...",
                    nl=False, err=True,
                )
                sys.stderr.flush()

            embed_q_3c.put(None)  # shutdown embedder
            embed_t_3c.join()

            click.echo(f"\n  Updated {updated:,} people records with position jurisdictions", err=True)

        roles_db.close()
        # Checkpoint WAL after heavy writes to prevent corruption on re-open
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        person_database.close()

    # ==================== Step 4: Insert discovered orgs ====================
    click.echo("\n=== Step 4: Insert missing discovered orgs ===", err=True)
    person_database = get_person_database(db_path=db_path_obj, readonly=True)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    conn = org_database._connect()

    # Load labels for resolving org names
    all_labels = person_database.get_all_qid_labels()
    click.echo(f"  Loaded {len(all_labels):,} QID labels", err=True)

    existing_org_ids = org_database.get_all_source_ids(source="wikipedia")
    click.echo(f"  Existing orgs: {len(existing_org_ids):,}", err=True)

    # Collect org QIDs from people record JSON (works with or without dump scan)
    missing_orgs = 0
    people_scanned = 0
    for person_record in person_database.iter_records(source="wikidata"):
        record_data = person_record.record or {}
        org_qid = record_data.get("org_qid", "")
        if org_qid and org_qid not in existing_org_ids:
            org_label = all_labels.get(org_qid, org_qid)
            _insert_discovered_org(conn, org_database._is_v2, org_qid, org_label, "fix_resume")
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
            org_label = all_labels.get(org_qid, org_qid)
            _insert_discovered_org(conn, org_database._is_v2, org_qid, org_label, "fix_resume")
            existing_org_ids.add(org_qid)
            missing_orgs += 1

    conn.commit()
    click.echo(f"\n  Inserted {missing_orgs:,} missing discovered orgs (scanned {people_scanned:,} people)", err=True)

    org_database.close()
    person_database.close()
    click.echo("\nFix complete!", err=True)
