"""Database import commands — GLEIF, SEC, Companies House, Wikidata, locations."""

from typing import Optional

import click

from ._common import _configure_logging, _resolve_db_path


@click.command("gleif-info")
def db_gleif_info():
    """
    Show information about the latest available GLEIF data file.

    \b
    Examples:
        corp-entity-db gleif-info
    """
    from corp_entity_db.importers import GleifImporter

    importer = GleifImporter()

    try:
        info = importer.get_latest_file_info()
        record_count = info.get('record_count')

        click.echo("\nLatest GLEIF Data File")
        click.echo("=" * 40)
        click.echo(f"File ID: {info['id']}")
        click.echo(f"Publish Date: {info['publish_date']}")
        click.echo(f"Record Count: {record_count:,}" if record_count else "Record Count: unknown")

        delta = info.get("delta_from_last_file", {})
        if delta:
            click.echo(f"\nChanges from previous file:")
            if delta.get('new'):
                click.echo(f"  New: {delta.get('new'):,}")
            if delta.get('updated'):
                click.echo(f"  Updated: {delta.get('updated'):,}")
            if delta.get('retired'):
                click.echo(f"  Retired: {delta.get('retired'):,}")

    except Exception as e:
        raise click.ClickException(f"Failed to get GLEIF info: {e}")


@click.command("import-gleif")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--download", is_flag=True, help="Download latest GLEIF file before importing")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--db", "db_path", type=click.Path(), help="Database path (default: ~/.cache/corp-entity-db/entities.db)")
@click.option("--limit", type=int, help="Limit number of records to import")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_gleif(file_path: Optional[str], download: bool, force: bool, db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import GLEIF LEI data into the entity database.

    If no file path is provided and --download is set, downloads the latest
    GLEIF data file automatically. Downloaded files are cached and reused
    unless --force is specified.

    \b
    Examples:
        corp-entity-db import-gleif /path/to/lei-records.xml
        corp-entity-db import-gleif --download
        corp-entity-db import-gleif --download --limit 10000
        corp-entity-db import-gleif --download --force  # Re-download
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder
    from corp_entity_db.importers import GleifImporter

    importer = GleifImporter()

    # Handle file path
    if file_path is None:
        if not download:
            raise click.UsageError("Either provide a file path or use --download to fetch the latest GLEIF data")
        click.echo("Downloading latest GLEIF data...", err=True)
        file_path = str(importer.download_latest(force=force))
    elif download:
        click.echo("Downloading latest GLEIF data (ignoring provided file path)...", err=True)
        file_path = str(importer.download_latest(force=force))

    click.echo(f"Importing GLEIF data from {file_path}...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_file(file_path, limit=limit):
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} GLEIF records successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
    database.close()


@click.command("import-sec")
@click.option("--download", is_flag=True, help="Download bulk submissions.zip (~500MB, ~100K+ filers)")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local file (submissions.zip or company_tickers.json)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=10000, help="Batch size (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_sec(download: bool, file_path: Optional[str], db_path: Optional[str], limit: Optional[int], batch_size: int, verbose: bool):
    """
    Import SEC Edgar data into the entity database.

    By default, downloads the bulk submissions.zip file which contains
    ALL SEC filers (~100K+), not just companies with ticker symbols (~10K).

    \b
    Examples:
        corp-entity-db import-sec --download
        corp-entity-db import-sec --download --limit 50000
        corp-entity-db import-sec --file /path/to/submissions.zip
        corp-entity-db import-sec --file /path/to/company_tickers.json  # legacy
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder
    from corp_entity_db.importers import SecEdgarImporter

    if not download and not file_path:
        raise click.UsageError("Either --download or --file is required")

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = SecEdgarImporter()

    # Get records
    if file_path:
        click.echo(f"Importing SEC Edgar data from {file_path}...", err=True)
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        click.echo("Downloading SEC submissions.zip (~500MB)...", err=True)
        record_iter = importer.import_from_url(limit=limit)

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} SEC Edgar records successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
    database.close()


@click.command("import-sec-officers")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--start-year", type=int, default=2020, help="Start year (default: 2020)")
@click.option("--end-year", type=int, help="End year (default: current year)")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--resume", is_flag=True, help="Resume from saved progress")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_sec_officers(db_path: Optional[str], start_year: int, end_year: Optional[int], limit: Optional[int], batch_size: int, resume: bool, skip_existing: bool, verbose: bool):
    """
    Import SEC Form 4 insider data into the people database.

    Downloads Form 4 filings from SEC EDGAR and extracts officers, directors,
    and significant investors (10%+ owners) from each company.

    Form 4 filings are submitted when insiders buy or sell company stock.
    They contain the person's name, role (officer/director), and company.

    Rate limited to 5 requests/second to comply with SEC guidelines.

    \b
    Examples:
        corp-entity-db import-sec-officers --limit 1000
        corp-entity-db import-sec-officers --start-year 2023
        corp-entity-db import-sec-officers --resume
        corp-entity-db import-sec-officers --skip-existing -v
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database
    from corp_entity_db.embeddings import CompanyEmbedder
    from corp_entity_db.importers.sec_form4 import SecForm4Importer

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing SEC Form 4 officers/directors to {db_path_obj}...", err=True)
    click.echo(f"Year range: {start_year} - {end_year or 'current'}", err=True)
    if resume:
        click.echo("Resuming from saved progress...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = SecForm4Importer()

    # Import records in batches
    records = []
    count = 0
    skipped_existing = 0

    def progress_callback(year: int, quarter: int, filing_idx: int, accession: str, total: int) -> None:
        if verbose and filing_idx % 100 == 0:
            click.echo(f"  {year} Q{quarter}: {filing_idx} filings, {total} records", err=True)

    for record in importer.import_range(
        start_year=start_year,
        end_year=end_year,
        limit=limit,
        resume=resume,
        progress_callback=progress_callback,
    ):
        # Skip existing records if flag is set
        if skip_existing:
            existing = database.get_by_source_id(record.source, record.source_id)
            if existing is not None:
                skipped_existing += 1
                continue

        # Look up org ID by CIK if available
        issuer_cik = record.record.get("issuer_cik", "")
        if issuer_cik:
            org_id = org_database.get_id_by_source_id("sec_edgar", issuer_cik.zfill(10))
            if org_id is not None:
                record.known_for_org_id = org_id

        records.append(record)

        if len(records) >= batch_size:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings = embedder.embed_batch(embedding_texts)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        embedding_texts = [r.get_embedding_text() for r in records]
        embeddings = embedder.embed_batch(embedding_texts)
        database.insert_batch(records, embeddings)
        count += len(records)

    if skip_existing and skipped_existing > 0:
        click.echo(f"\nImported {count} SEC officers/directors (skipped {skipped_existing} existing).", err=True)
    else:
        click.echo(f"\nImported {count} SEC officers/directors successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)

    org_database.close()
    database.close()


@click.command("import-ch-officers")
@click.option("--file", "file_path", type=click.Path(exists=True), required=True, help="Path to CH officers zip file (Prod195)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--resume", is_flag=True, help="Resume from saved progress")
@click.option("--include-resigned", is_flag=True, help="Include resigned officers (default: current only)")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_ch_officers(file_path: str, db_path: Optional[str], limit: Optional[int], batch_size: int, resume: bool, include_resigned: bool, skip_existing: bool, verbose: bool):
    """
    Import Companies House officers data into the people database.

    Requires the Prod195 bulk officers zip file from Companies House.
    Request access via BulkProducts@companieshouse.gov.uk.

    \b
    Examples:
        corp-entity-db import-ch-officers --file officers.zip --limit 10000
        corp-entity-db import-ch-officers --file officers.zip --resume
        corp-entity-db import-ch-officers --file officers.zip --include-resigned
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database
    from corp_entity_db.embeddings import CompanyEmbedder
    from corp_entity_db.importers.companies_house_officers import CompaniesHouseOfficersImporter

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Companies House officers to {db_path_obj}...", err=True)
    if resume:
        click.echo("Resuming from saved progress...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = CompaniesHouseOfficersImporter()

    # Import records in batches
    records = []
    count = 0
    skipped_existing = 0

    def progress_callback(file_idx: int, line_num: int, total: int) -> None:
        if verbose:
            click.echo(f"  File {file_idx}: line {line_num}, {total} records", err=True)

    for record in importer.import_from_zip(
        file_path,
        limit=limit,
        resume=resume,
        current_only=not include_resigned,
        progress_callback=progress_callback,
    ):
        # Skip existing records if flag is set
        if skip_existing:
            existing = database.get_by_source_id(record.source, record.source_id)
            if existing is not None:
                skipped_existing += 1
                continue

        # Look up org ID by company number if available
        company_number = record.record.get("company_number", "")
        if company_number:
            org_id = org_database.get_id_by_source_id("companies_house", company_number)
            if org_id is not None:
                record.known_for_org_id = org_id

        records.append(record)

        if len(records) >= batch_size:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings = embedder.embed_batch(embedding_texts)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        embedding_texts = [r.get_embedding_text() for r in records]
        embeddings = embedder.embed_batch(embedding_texts)
        database.insert_batch(records, embeddings)
        count += len(records)

    if skip_existing and skipped_existing > 0:
        click.echo(f"\nImported {count} CH officers (skipped {skipped_existing} existing).", err=True)
    else:
        click.echo(f"\nImported {count} CH officers successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)

    org_database.close()
    database.close()


@click.command("import-wikidata")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--type", "query_type", type=click.Choice(["lei", "ticker", "public", "business", "organization", "nonprofit", "government"]), default="lei",
              help="Query type to use for fetching data")
@click.option("--all", "import_all", is_flag=True, help="Run all query types sequentially")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_wikidata(db_path: Optional[str], limit: Optional[int], batch_size: int, query_type: str, import_all: bool, verbose: bool):
    """
    Import organization data from Wikidata via SPARQL.

    Uses simplified SPARQL queries that avoid timeouts on Wikidata's endpoint.
    Query types target different organization categories.

    \b
    Query types:
        lei          Companies with LEI codes (fastest, most reliable)
        ticker       Companies listed on stock exchanges
        public       Direct instances of "public company" (Q891723)
        business     Direct instances of "business enterprise" (Q4830453)
        organization All organizations (Q43229) - NGOs, associations, etc.
        nonprofit    Non-profit organizations (Q163740)
        government   Government agencies (Q327333)

    \b
    Examples:
        corp-entity-db import-wikidata --limit 10
        corp-entity-db import-wikidata --type organization --limit 1000
        corp-entity-db import-wikidata --type nonprofit --limit 5000
        corp-entity-db import-wikidata --all --limit 10000
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder
    from corp_entity_db.importers import WikidataImporter

    click.echo(f"Importing Wikidata organization data via SPARQL (type={query_type}, all={import_all})...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = WikidataImporter(batch_size=500)  # Smaller SPARQL batch size for reliability

    # Import records in batches
    records = []
    count = 0

    for record in importer.import_from_sparql(limit=limit, query_type=query_type, import_all=import_all):
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Wikidata records successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
    database.close()


@click.command("import-people")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=1000, help="Batch size for commits (default: 1000)")
@click.option("--type", "query_type", type=click.Choice([
    "executive", "politician", "athlete", "artist",
    "academic", "scientist", "journalist", "entrepreneur", "activist"
]), default="executive", help="Person type to import")
@click.option("--all", "import_all", is_flag=True, help="Run all person type queries sequentially")
@click.option("--enrich", is_flag=True, help="Query individual people to get role/org data (slower, resumable)")
@click.option("--enrich-only", is_flag=True, help="Only enrich existing people (skip bulk import)")
@click.option("--enrich-dates", is_flag=True, help="Query individual people to get start/end dates (slower)")
@click.option("--skip-existing", is_flag=True, help="Skip records that already exist (default: update them)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_people(db_path: Optional[str], limit: Optional[int], batch_size: int, query_type: str, import_all: bool, enrich: bool, enrich_only: bool, enrich_dates: bool, skip_existing: bool, verbose: bool):
    """
    Import notable people data from Wikidata via SPARQL.

    Uses a two-phase approach for reliability:
    1. Bulk import: Fast fetch of QID, name, country (no timeouts)
    2. Enrich (optional): Per-person queries for role/org/dates

    Imports people with English Wikipedia articles (ensures notability).

    \b
    Examples:
        corp-entity-db import-people --type executive --limit 5000
        corp-entity-db import-people --all --limit 10000
        corp-entity-db import-people --type executive --enrich
        corp-entity-db import-people --enrich-only --limit 100
        corp-entity-db import-people --type politician -v
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, get_database
    from corp_entity_db.embeddings import CompanyEmbedder
    from corp_entity_db.importers.wikidata_people import WikidataPeopleImporter

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Importing Wikidata people to {db_path_obj}...", err=True)

    # Initialize components (readonly=False for import operations)
    database = get_person_database(db_path=db_path_obj, readonly=False)
    org_database = get_database(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()
    importer = WikidataPeopleImporter(batch_size=batch_size)

    count = 0

    # Phase 1: Bulk import (fast, minimal data) - skip if --enrich-only
    if not enrich_only:
        records = []
        skipped_existing = 0

        click.echo("Phase 1: Bulk import (QID, name, country)...", err=True)

        for record in importer.import_from_sparql(limit=limit, query_type=query_type, import_all=import_all):
            # Skip existing records if flag is set
            if skip_existing:
                existing = database.get_by_source_id(record.source, record.source_id)
                if existing is not None:
                    skipped_existing += 1
                    continue

            records.append(record)

            if len(records) >= batch_size:
                embedding_texts = [r.get_embedding_text() for r in records]
                embeddings = embedder.embed_batch(embedding_texts)
                database.insert_batch(records, embeddings)
                count += len(records)

                click.echo(f"  Imported {count} people...", err=True)
                records = []

        # Final batch
        if records:
            embedding_texts = [r.get_embedding_text() for r in records]
            embeddings = embedder.embed_batch(embedding_texts)
            database.insert_batch(records, embeddings)
            count += len(records)

        if skip_existing and skipped_existing > 0:
            click.echo(f"\nPhase 1 complete: {count} people imported (skipped {skipped_existing} existing).", err=True)
        else:
            click.echo(f"\nPhase 1 complete: {count} people imported.", err=True)
    else:
        click.echo("Skipping Phase 1 (bulk import) - using existing database records.", err=True)
        # Enable enrich if enrich_only is set
        enrich = True

    # Phase 2: Enrich with role/org/dates (optional, slower but resumable)
    if enrich:
        click.echo("\nPhase 2: Enriching with role/org/dates (parallel queries)...", err=True)
        # Get all people without role/org
        people_to_enrich = []
        enriched_count = 0
        for record in database.iter_records():
            if not record.known_for_role and not record.known_for_org_name:
                people_to_enrich.append(record)
                enriched_count += 1
                # Apply limit if --enrich-only
                if enrich_only and limit and enriched_count >= limit:
                    break

        if people_to_enrich:
            click.echo(f"Found {len(people_to_enrich)} people to enrich...", err=True)
            importer.enrich_people_role_org_batch(people_to_enrich, delay_seconds=0.1, max_workers=5)

            # Persist the enriched data and re-generate embeddings
            updated = 0
            org_count = 0
            date_count = 0
            for person in people_to_enrich:
                if person.known_for_role or person.known_for_org_name:
                    # Look up org ID if we have org_qid (wikidata→wikidata only)
                    org_qid = person.record.get("org_qid", "")
                    if org_qid:
                        org_id = org_database.get_id_by_source_id("wikipedia", org_qid)
                        if org_id is not None:
                            person.known_for_org_id = org_id

                    # Update the record with new role/org/dates and re-embed
                    new_embedding_text = person.get_embedding_text()
                    new_embedding = embedder.embed(new_embedding_text)
                    if database.update_role_org(
                        person.source, person.source_id,
                        person.known_for_role,
                        person.known_for_org_id,
                        new_embedding,
                        person.from_date, person.to_date,
                    ):
                        updated += 1
                        if person.known_for_org_name:
                            org_count += 1
                        if person.from_date or person.to_date:
                            date_count += 1
                        if verbose:
                            date_str = ""
                            if person.from_date or person.to_date:
                                date_str = f" ({person.from_date or '?'} - {person.to_date or '?'})"
                            click.echo(f"  {person.name}: {person.known_for_role} at {person.known_for_org_name}{date_str}", err=True)

            click.echo(f"Updated {updated} people ({org_count} with orgs, {date_count} with dates).", err=True)

    # Phase 3: Enrich with dates (optional, even slower)
    if enrich_dates:
        click.echo("\nPhase 3: Enriching with dates...", err=True)
        # Get all people without dates but with role (dates are associated with positions)
        people_to_enrich = []
        for record in database.iter_records():
            if not record.from_date and not record.to_date and record.known_for_role:
                people_to_enrich.append(record)

        if people_to_enrich:
            click.echo(f"Found {len(people_to_enrich)} people to enrich with dates...", err=True)
            enriched = importer.enrich_people_batch(people_to_enrich, delay_seconds=0.3)

            # Persist the enriched dates
            updated = 0
            for person in people_to_enrich:
                if person.from_date or person.to_date:
                    if database.update_dates(person.source, person.source_id, person.from_date, person.to_date):
                        updated += 1
                        if verbose:
                            click.echo(f"  {person.name}: {person.from_date or '?'} - {person.to_date or '?'}", err=True)

            click.echo(f"Updated {updated} people with dates.", err=True)

    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
    org_database.close()
    database.close()


from typing import NamedTuple
import queue
import sys
import threading
from pathlib import Path


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

    database = get_person_database(db_path=db_path_obj, readonly=False)

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

    # Pipeline: reader thread → write_queue → main thread (DB writes)
    # Embeddings are deferred to post-import (needs FK resolution for rich text)
    write_queue: queue.Queue = queue.Queue(maxsize=4)
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

    click.echo("Starting pipeline (reader → writer, embeddings deferred to post-import)...", err=True)
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
            embed_queue=write_queue,
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

            click.echo(f"\r  Progress: {people_count:,} people, {orgs_count:,} orgs...", nl=False, err=True)
            sys.stderr.flush()

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
        save_progress()
        click.echo(f"  Saved: {people_count:,} people, {orgs_count:,} orgs", err=True)
        raise
    finally:
        save_progress()
        reader.join(timeout=5)

    click.echo(f"Import complete: {people_count:,} people, {orgs_count:,} orgs", err=True)

    # Backfill known_for_org_id from reverse org→person mappings
    # (P169 CEO, P488 chairperson, P112 founder, P1037 director, P3320 board member)
    reverse_map = importer.get_reverse_person_orgs()
    if reverse_map and people:
        # Pass org QIDs directly — backfill_known_for_org resolves to org/location FKs
        backfilled = person_database.backfill_known_for_org(reverse_map)
        click.echo(
            f"Backfilled known_for_org from org executive properties: "
            f"{backfilled:,} updated ({len(reverse_map):,} reverse mappings)",
            err=True,
        )

    # Free reverse map memory — no longer needed
    del reverse_map

    # Keep references for final label resolution
    database = person_database
    if org_database:
        org_database.close()

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


@click.command("import-companies-house")
@click.option("--download", is_flag=True, help="Download bulk data file (free, no API key needed)")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--file", "file_path", type=click.Path(exists=True), help="Local Companies House CSV/JSON file")
@click.option("--search", "search_terms", type=str, help="Comma-separated search terms (requires API key)")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", type=int, help="Limit number of records")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_companies_house(
    download: bool,
    force: bool,
    file_path: Optional[str],
    search_terms: Optional[str],
    db_path: Optional[str],
    limit: Optional[int],
    batch_size: int,
    verbose: bool,
):
    """
    Import UK Companies House data into the entity database.

    \b
    Options:
    --download    Download free bulk data (all UK companies, ~5M records)
    --file        Import from local CSV/JSON file
    --search      Search via API (requires COMPANIES_HOUSE_API_KEY)

    \b
    Examples:
        corp-entity-db import-companies-house --download
        corp-entity-db import-companies-house --download --limit 100000
        corp-entity-db import-companies-house --file /path/to/companies.csv
        corp-entity-db import-companies-house --search "bank,insurance"
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder
    from corp_entity_db.importers import CompaniesHouseImporter

    if not file_path and not search_terms and not download:
        raise click.UsageError("Either --download, --file, or --search is required")

    click.echo("Importing Companies House data...", err=True)

    # Initialize components (readonly=False for import operations)
    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj, embedding_dim=embedder.embedding_dim, readonly=False)
    importer = CompaniesHouseImporter()

    # Get records
    if download:
        # Download bulk data file
        csv_path = importer.download_bulk_data(force=force)
        click.echo(f"Using bulk data file: {csv_path}", err=True)
        record_iter = importer.import_from_file(csv_path, limit=limit)
    elif file_path:
        record_iter = importer.import_from_file(file_path, limit=limit)
    else:
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
        click.echo(f"Searching for: {terms}", err=True)
        record_iter = importer.import_from_search(
            search_terms=terms,
            limit_per_term=limit or 100,
            total_limit=limit,
        )

    # Import records in batches
    records = []
    count = 0

    for record in record_iter:
        records.append(record)

        if len(records) >= batch_size:
            names = [r.name for r in records]
            embeddings = embedder.embed_batch(names)
            database.insert_batch(records, embeddings)
            count += len(records)
            click.echo(f"Imported {count} records...", err=True)
            records = []

    # Final batch
    if records:
        names = [r.name for r in records]
        embeddings = embedder.embed_batch(names)
        database.insert_batch(records, embeddings)
        count += len(records)

    click.echo(f"\nImported {count} Companies House records successfully.", err=True)
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
    database.close()


@click.command("import-locations")
@click.option("--from-pycountry", is_flag=True, help="Import countries from pycountry")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_import_locations(from_pycountry: bool, db_path: Optional[str], verbose: bool):
    """
    Import locations into the database.

    \b
    Examples:
        corp-entity-db import-locations --from-pycountry
    """
    _configure_logging(verbose)

    if not from_pycountry:
        raise click.UsageError("Must specify --from-pycountry")

    from corp_entity_db.store import get_locations_database

    # readonly=False for import operations
    locations_db = get_locations_database(db_path, readonly=False)
    count = locations_db.import_from_pycountry()

    click.echo(f"Imported {count:,} locations from pycountry")
    click.echo("Run `corp-entity-db post-import` to update search indexes.", err=True)
