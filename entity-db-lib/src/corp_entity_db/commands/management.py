"""Database management commands — status, maintenance, migration."""

from pathlib import Path
from typing import Optional

import click

from ._common import _configure_logging, _resolve_db_path


@click.command("status")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--for-llm", is_flag=True, help="Output schema and type details for LLM documentation")
def db_status(db_path: Optional[str], for_llm: bool):
    """
    Show database status and statistics.

    \b
    Examples:
        corp-entity-db status
        corp-entity-db status --for-llm
        corp-entity-db status --db /path/to/entities.db
    """
    import sqlite3

    from corp_entity_db import OrganizationDatabase
    from corp_entity_db.hub import DEFAULT_DB_FILENAME, DEFAULT_DB_FULL_FILENAME, DEFAULT_DB_LITE_FILENAME
    from corp_entity_db.store import get_locations_database, get_person_database, get_roles_database

    db_path_obj = _resolve_db_path(db_path)

    try:
        database = OrganizationDatabase(db_path=db_path_obj)
        stats = database.get_stats()

        click.echo("\nEntity Database Status")
        click.echo("=" * 40)
        click.echo(f"Total organizations: {stats.total_records:,}")
        click.echo(f"Embedding dimension: {stats.embedding_dimension}")
        click.echo(f"Database size: {stats.database_size_bytes / 1024 / 1024:.2f} MB")

        # Get person stats
        person_db = get_person_database(db_path=db_path_obj)
        person_stats = person_db.get_stats()
        click.echo(f"Total people: {person_stats.get('total_records', 0):,}")

        if stats.by_source:
            click.echo("\n=== Organizations by Source ===")
            click.echo(f"{'Source':<20} {'Records':>15}")
            click.echo("-" * 36)
            for source, count in sorted(stats.by_source.items(), key=lambda x: -x[1]):
                click.echo(f"{source:<20} {count:>15,}")

        # People by source
        if person_stats.get("by_source"):
            click.echo("\n=== People by Source ===")
            click.echo(f"{'Source':<20} {'Records':>15}")
            click.echo("-" * 36)
            for source, count in sorted(person_stats["by_source"].items(), key=lambda x: -x[1]):
                click.echo(f"{source:<20} {count:>15,}")

        # Roles and Locations counts
        try:
            roles_db = get_roles_database(db_path=db_path_obj)
            roles_stats = roles_db.get_stats()
            locations_db = get_locations_database(db_path=db_path_obj)
            locations_stats = locations_db.get_stats()

            click.echo("\n=== Other Tables ===")
            click.echo(f"{'Table':<20} {'Records':>15}")
            click.echo("-" * 36)
            click.echo(f"{'roles':<20} {roles_stats['total_roles']:>15,}")
            click.echo(f"{'locations':<20} {locations_stats['total_locations']:>15,}")
        except Exception:
            pass  # Tables may not exist in older databases

        # For LLM mode: output enum tables and schema details
        if for_llm:
            db_file = str(db_path_obj)
            conn = sqlite3.connect(f"file:{db_file}?immutable=1", uri=True)
            conn.row_factory = sqlite3.Row

            click.echo("\n" + "=" * 60)
            click.echo("LLM DOCUMENTATION DETAILS")
            click.echo("=" * 60)

            # Database file variants
            click.echo("\n=== Database File Variants ===")
            click.echo(f"Default filename: {DEFAULT_DB_FILENAME}")
            click.echo(f"Full database: {DEFAULT_DB_FULL_FILENAME}")
            click.echo(f"Lite database: {DEFAULT_DB_LITE_FILENAME}")
            click.echo(f"Default path: {_resolve_db_path()}")

            # Source types
            click.echo("\n=== source_types (Data Sources) ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM source_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Organization types
            click.echo("\n=== organization_types (EntityType) ===")
            click.echo(f"{'ID':<5} {'Name':<25}")
            click.echo("-" * 30)
            cursor = conn.execute("SELECT id, name FROM organization_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<25}")

            # People types
            click.echo("\n=== people_types (PersonType) ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM people_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Simplified location types
            click.echo("\n=== simplified_location_types ===")
            click.echo(f"{'ID':<5} {'Name':<20}")
            click.echo("-" * 25)
            cursor = conn.execute("SELECT id, name FROM simplified_location_types ORDER BY id")
            for row in cursor:
                click.echo(f"{row['id']:<5} {row['name']:<20}")

            # Location types (sample)
            click.echo("\n=== location_types (Sample - Wikidata QID mappings) ===")
            click.echo(f"{'ID':<5} {'QID':<12} {'Name':<30} {'Simplified':<15}")
            click.echo("-" * 65)
            cursor = conn.execute("""
                SELECT lt.id, lt.qid, lt.name, slt.name as simplified
                FROM location_types lt
                JOIN simplified_location_types slt ON lt.simplified_id = slt.id
                ORDER BY lt.id
                LIMIT 20
            """)
            for row in cursor:
                qid = f"Q{row['qid']}" if row["qid"] else ""
                click.echo(f"{row['id']:<5} {qid:<12} {row['name']:<30} {row['simplified']:<15}")
            click.echo("... (showing first 20 of many)")

            # Table schemas
            click.echo("\n=== Table Schemas ===")
            for table in ["organizations", "people", "roles", "locations"]:
                click.echo(f"\n{table}:")
                cursor = conn.execute(f"PRAGMA table_info({table})")
                for row in cursor:
                    nullable = "" if row["notnull"] else "NULL"
                    pk = "PK" if row["pk"] else ""
                    click.echo(f"  {row['name']:<25} {row['type']:<15} {pk:<3} {nullable}")

            conn.close()

        database.close()

    except Exception as e:
        raise click.ClickException(f"Failed to read database: {e}")


@click.command("canonicalize")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10000, help="Batch size for updates (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_canonicalize(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Canonicalize organizations by linking equivalent records across sources.

    Records are considered equivalent if they share:
    - Same LEI (globally unique legal entity identifier)
    - Same ticker symbol
    - Same CIK (SEC identifier)
    - Same normalized name (after lowercasing, removing dots)
    - Same name with suffix expansion (Ltd -> Limited, etc.)

    For each group, the highest-priority source becomes canonical:
    gleif > sec_edgar > companies_house > wikipedia

    Canonicalization enables better search re-ranking by boosting results
    that have records from multiple authoritative sources.

    \b
    Examples:
        corp-entity-db canonicalize
        corp-entity-db canonicalize -v
        corp-entity-db canonicalize --db /path/to/entities.db
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase
    from corp_entity_db.store import get_person_database

    db_path_obj = _resolve_db_path(db_path)

    try:
        # Canonicalize organizations (readonly=False for write operations)
        database = OrganizationDatabase(db_path=db_path_obj, readonly=False)
        click.echo("Running organization canonicalization...", err=True)

        result = database.canonicalize(batch_size=batch_size)

        click.echo("\nOrganization Canonicalization Results")
        click.echo("=" * 40)
        click.echo(f"Total records processed: {result['total_records']:,}")
        click.echo(f"Equivalence groups found: {result['groups_found']:,}")
        click.echo(f"Multi-record groups: {result['multi_record_groups']:,}")
        click.echo(f"Records updated: {result['records_updated']:,}")

        database.close()

        # Canonicalize people (readonly=False for write operations)
        person_db = get_person_database(db_path=db_path_obj, readonly=False)
        click.echo("\nRunning people canonicalization...", err=True)

        people_result = person_db.canonicalize(batch_size=batch_size)

        click.echo("\nPeople Canonicalization Results")
        click.echo("=" * 40)
        click.echo(f"Total records processed: {people_result['total_records']:,}")
        click.echo(f"Matched by organization: {people_result['matched_by_org']:,}")
        click.echo(f"Matched by date overlap: {people_result['matched_by_date']:,}")
        click.echo(f"Canonical groups: {people_result['canonical_groups']:,}")
        click.echo(f"Records in multi-record groups: {people_result['records_in_groups']:,}")

        person_db.close()

    except Exception as e:
        raise click.ClickException(f"Canonicalization failed: {e}")


@click.command("download")
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--db", "db_path", type=click.Path(), help="Output path for database")
@click.option("--full", is_flag=True, help="Download full version (larger, includes record metadata)")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_download(repo: str, db_path: Optional[str], full: bool, force: bool, verbose: bool):
    """
    Download entity database from HuggingFace Hub.

    By default downloads the lite version (smaller, without record metadata).
    Use --full for the complete database with all source record data.

    \b
    Examples:
        corp-entity-db download
        corp-entity-db download --full
        corp-entity-db download --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from corp_entity_db.hub import download_database, db_filenames, USEARCH_INDEX_FILES

    ctx = click.get_current_context(silent=True)
    db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    full_fn, lite_fn, _ = db_filenames(db_version)
    filename = full_fn if full else lite_fn
    click.echo(f"Downloading {'full ' if full else 'lite '}database from {repo}...", err=True)

    try:
        path = download_database(
            repo_id=repo,
            filename=filename,
            force_download=force,
        )
        click.echo(f"Database downloaded to: {path}")

        # Create v2 symlink for backwards compatibility
        db_dir = path.parent
        v2_link = db_dir / "entities-v2.db"
        if not v2_link.exists():
            v2_link.symlink_to(path.name)
            click.echo(f"  Symlink: entities-v2.db -> {path.name}")

        # Report USearch index file status
        for idx_name in USEARCH_INDEX_FILES:
            idx_path = db_dir / idx_name
            if idx_path.exists():
                click.echo(f"  Index: {idx_name} ({idx_path.stat().st_size / 1024**2:.0f} MB)")
            else:
                click.echo(f"  Index: {idx_name} (not found — run: corp-entity-db build-index)")
    except Exception as e:
        raise click.ClickException(f"Download failed: {e}")


@click.command("upload")
@click.argument("db_path", type=click.Path(exists=True), required=False)
@click.option("--repo", type=str, default="Corp-o-Rate-Community/entity-references", help="HuggingFace repo ID")
@click.option("--message", type=str, default="Update entity database", help="Commit message")
@click.option("--no-lite", is_flag=True, help="Skip creating lite version (without record data)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_upload(db_path: Optional[str], repo: str, message: str, no_lite: bool, verbose: bool):
    """
    Upload entity database to HuggingFace Hub.

    First VACUUMs the database, then creates and uploads:
    - Full database + lite variant (without record data or embeddings)
    - USearch HNSW index files (.bin)

    If no path is provided, uploads from the default cache location.
    Requires HF_TOKEN environment variable to be set.

    \b
    Examples:
        corp-entity-db upload
        corp-entity-db upload /path/to/entities.db
        corp-entity-db upload --no-lite
        corp-entity-db upload --repo my-org/my-entity-db
    """
    _configure_logging(verbose)
    from corp_entity_db.hub import upload_database_with_variants, DEFAULT_CACHE_DIR, db_filenames

    ctx = click.get_current_context(silent=True)
    db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    full_fn, _, _ = db_filenames(db_version)

    # Use default cache location if no path provided
    if db_path is None:
        db_path = str(DEFAULT_CACHE_DIR / full_fn)
        if not Path(db_path).exists():
            raise click.ClickException(
                f"Database not found at default location: {db_path}\n"
                "Build the database first with import commands, or specify a path."
            )

    click.echo(f"Uploading {db_path} to {repo}...", err=True)
    click.echo("  - Running VACUUM to optimize database", err=True)
    if not no_lite:
        click.echo("  - Creating lite version (without record data)", err=True)

    try:
        results = upload_database_with_variants(
            db_path=db_path,
            repo_id=repo,
            commit_message=message,
            include_lite=not no_lite,
            version=db_version,
        )
        click.echo(f"\nUploaded {len(results)} file(s) successfully:")
        for filename, url in results.items():
            click.echo(f"  - {filename}")
    except Exception as e:
        raise click.ClickException(f"Upload failed: {e}")


@click.command("create-lite")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path (default: adds -lite suffix)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_create_lite(db_path: str, output: Optional[str], verbose: bool):
    """
    Create a lite version of the database without record data or embeddings.

    The lite version strips the `record` column and drops all embedding
    tables. Search uses USearch HNSW index files (.bin) instead.

    \b
    Examples:
        corp-entity-db create-lite entities.db
        corp-entity-db create-lite entities.db -o entities-lite.db
    """
    _configure_logging(verbose)
    from corp_entity_db.hub import create_lite_database

    click.echo(f"Creating lite database from {db_path}...", err=True)

    try:
        lite_path = create_lite_database(db_path, output)
        click.echo(f"Lite database created: {lite_path}")
    except Exception as e:
        raise click.ClickException(f"Failed to create lite database: {e}")


@click.command("repair-embeddings")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=1000, help="Batch size for embedding generation (default: 1000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_repair_embeddings(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Generate missing embeddings for organizations in the database.

    This repairs databases where organizations were imported without embeddings.

    \b
    Examples:
        corp-entity-db repair-embeddings
        corp-entity-db repair-embeddings --batch-size 500
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder

    # readonly=False for write operations (embedding repair)
    db_path_obj = _resolve_db_path(db_path)
    database = OrganizationDatabase(db_path=db_path_obj, readonly=False)
    embedder = CompanyEmbedder()

    # Check how many need repair
    missing_count = database.get_missing_embedding_count()
    if missing_count == 0:
        click.echo("All organizations have embeddings. Nothing to repair.")
        database.close()
        return

    click.echo(f"Found {missing_count:,} organizations without embeddings.", err=True)
    click.echo("Generating embeddings...", err=True)

    # Process in batches
    count = 0
    for batch in database.get_missing_embedding_ids(batch_size=batch_size):
        ids = [item[0] for item in batch]
        names = [item[1] for item in batch]

        embeddings = embedder.embed_batch(names)
        database.update_embeddings_batch(ids, embeddings)
        count += len(ids)
        click.echo(f"Repaired {count:,} / {missing_count:,} embeddings...", err=True)

    click.echo(f"\nRepaired {count:,} embeddings successfully.", err=True)
    database.close()


@click.command("build-index")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--M", type=int, default=32, help="Number of connections per node (default 32)")
@click.option("--ef-construction", type=int, default=200, help="Construction quality parameter (default 200)")
@click.option("--ef-search", type=int, default=200, help="Search quality parameter (default 200)")
@click.option("--people/--no-people", default=True, help="Build USearch index for people")
@click.option("--orgs/--no-orgs", default=True, help="Build USearch index for organizations")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_build_index(db_path: Optional[str], m: int, ef_construction: int, ef_search: int, people: bool, orgs: bool, verbose: bool):
    """
    Build USearch index for fast approximate nearest neighbor search.

    USearch uses HNSW algorithm (Hierarchical Navigable Small World) and provides
    sub-millisecond query times on millions of vectors. Much faster than GPU or PCA approaches.

    \b
    Parameters:
    - M: Higher = better quality, more memory (16-64, default 32)
    - ef_construction: Higher = better quality, slower build (100-500, default 200)
    - ef_search: Higher = better quality, slower search (50-500, default 200)

    \b
    Examples:
        corp-entity-db build-index
        corp-entity-db build-index --M 16              # Faster build, less memory
        corp-entity-db build-index --M 64 --ef-construction 400   # Highest quality
        corp-entity-db build-index --no-orgs           # People only
    """
    _configure_logging(verbose)

    from corp_entity_db.store import (
        _get_shared_connection,
        build_hnsw_index,
    )

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)
    click.echo(f"Parameters: M={m}, ef_construction={ef_construction}, ef_search={ef_search}", err=True)

    # Open read-only connection (we only read embeddings)
    conn = _get_shared_connection(db_path_obj, readonly=True)

    if people:
        click.echo("\n--- Building USearch index for people ---", err=True)

        def people_progress(done: int, total: int) -> None:
            click.echo(f"  Loaded {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_hnsw_index(
                conn, "people",
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=people_progress,
            )
            click.echo(f"People USearch index: {count:,} vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"People USearch index build failed: {e}")

    if orgs:
        click.echo("\n--- Building USearch index for organizations ---", err=True)

        def orgs_progress(done: int, total: int) -> None:
            click.echo(f"  Loaded {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_hnsw_index(
                conn, "organizations",
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=orgs_progress,
            )
            click.echo(f"Organizations USearch index: {count:,} vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"Organizations USearch index build failed: {e}")

    click.echo("\nUSearch index build complete!", err=True)


def _run_post_import(db_path_obj: Path, people: bool = True, orgs: bool = True, batch_size: int = 10000, embed_batch_size: int = 64) -> None:
    """
    Standard post-import steps: generate embeddings, build USearch indexes, VACUUM.

    Called automatically after imports or manually via `db post-import`.
    """
    import sqlite3

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder
    from corp_entity_db.store import (
        get_person_database,
        _get_shared_connection,
        build_hnsw_index,
    )

    embedder = None  # Lazy load

    # --- Step 1: Generate embeddings for new records ---
    click.echo("\n=== Step 1: Generate embeddings for new records ===", err=True)

    if orgs:
        org_db = OrganizationDatabase(db_path=db_path_obj, readonly=False)

        org_generated = 0
        for batch in org_db.get_missing_embedding_ids(batch_size=batch_size):
            if not batch:
                continue
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]
                fp32_batch = embedder.embed_batch(names, batch_size=embed_batch_size)
                org_db.update_embeddings_batch(ids, fp32_batch)
                org_generated += len(ids)
                if org_generated % 10000 == 0:
                    click.echo(f"  Generated {org_generated:,} org embeddings...", err=True)
        if org_generated:
            click.echo(f"  Generated {org_generated:,} org embeddings", err=True)
        else:
            click.echo("  Organizations: all embeddings up to date", err=True)

        org_db.close()

    if people:
        person_db = get_person_database(db_path=db_path_obj, readonly=False)

        person_generated = 0
        for batch in person_db.get_missing_embedding_ids(batch_size=batch_size):
            if not batch:
                continue
            if embedder is None:
                click.echo("  Loading embedding model...", err=True)
                embedder = CompanyEmbedder()
            for i in range(0, len(batch), embed_batch_size):
                sub_batch = batch[i:i + embed_batch_size]
                ids = [item[0] for item in sub_batch]
                names = [item[1] for item in sub_batch]
                fp32_batch = embedder.embed_batch(names, batch_size=embed_batch_size)
                person_db.update_embeddings_batch(ids, fp32_batch)
                person_generated += len(ids)
                if person_generated % 10000 == 0:
                    click.echo(f"  Generated {person_generated:,} person embeddings...", err=True)
        if person_generated:
            click.echo(f"  Generated {person_generated:,} person embeddings", err=True)
        else:
            click.echo("  People: all embeddings up to date", err=True)

        person_db.close()

    # --- Step 2: Build USearch indexes ---
    click.echo("\n=== Step 2: Build USearch indexes ===", err=True)

    conn = _get_shared_connection(db_path_obj, readonly=True)

    if people:
        count = build_hnsw_index(conn, "people")
        click.echo(f"  People USearch index: {count:,} vectors", err=True)

    if orgs:
        count = build_hnsw_index(conn, "organizations")
        click.echo(f"  Organizations USearch index: {count:,} vectors", err=True)

    # --- Step 3: VACUUM ---
    click.echo("\n=== Step 3: VACUUM database ===", err=True)
    vacuum_conn = sqlite3.connect(str(db_path_obj))
    vacuum_conn.execute("VACUUM")
    vacuum_conn.close()
    db_size_mb = db_path_obj.stat().st_size / 1024**2
    click.echo(f"  Database size after VACUUM: {db_size_mb:,.0f} MB", err=True)

    click.echo("\nPost-import complete!", err=True)


@click.command("post-import")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--people/--no-people", default=True, help="Process people")
@click.option("--orgs/--no-orgs", default=True, help="Process organizations")
@click.option("--batch-size", type=int, default=10000, help="Batch size (default: 10000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_post_import(db_path: Optional[str], people: bool, orgs: bool, batch_size: int, verbose: bool):
    """
    Run standard post-import steps: embeddings, USearch indexes, VACUUM.

    Run this after any import command to ensure search indexes are up to date.

    \b
    Steps:
    1. Generate embeddings for new records
    2. Build USearch HNSW indexes for fast search
    3. VACUUM database to reclaim space

    \b
    Examples:
        corp-entity-db post-import
        corp-entity-db post-import --no-orgs     # People only
        corp-entity-db post-import -v             # Verbose logging
    """
    _configure_logging(verbose)

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)
    _run_post_import(db_path_obj, people=people, orgs=orgs, batch_size=batch_size)


@click.command("migrate")
@click.argument("db_path", type=click.Path(exists=True))
@click.option("--rename-file", is_flag=True, help="Also rename companies.db to entities.db")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_migrate(db_path: str, rename_file: bool, yes: bool, verbose: bool):
    """
    Migrate database from legacy schema to new schema.

    Migrates from old naming (companies/company_embeddings tables)
    to new naming (organizations/organization_embeddings tables).

    \b
    What this does:
    - Renames 'companies' table to 'organizations'
    - Renames 'company_embeddings' table to 'organization_embeddings'
    - Updates all indexes

    \b
    Examples:
        corp-entity-db migrate companies.db
        corp-entity-db migrate companies.db --rename-file
        corp-entity-db migrate ~/.cache/corp-entity-db/companies.db --yes
    """
    _configure_logging(verbose)

    from pathlib import Path
    from corp_entity_db import OrganizationDatabase

    db_path_obj = Path(db_path)

    if not yes:
        click.confirm(
            f"This will migrate {db_path} from legacy schema (companies) to new schema (organizations).\n"
            "This operation cannot be undone. Continue?",
            abort=True
        )

    try:
        # readonly=False for schema migrations
        database = OrganizationDatabase(db_path=db_path, readonly=False)
        migrations = database.migrate_from_legacy_schema()
        database.close()

        if migrations:
            click.echo("Migration completed:")
            for table, action in migrations.items():
                click.echo(f"  {table}: {action}")
        else:
            click.echo("No migration needed. Database already uses new schema.")

        # Optionally rename the file
        if rename_file and db_path_obj.name.startswith("companies"):
            new_name = db_path_obj.name.replace("companies", "entities")
            new_path = db_path_obj.parent / new_name
            db_path_obj.rename(new_path)
            click.echo(f"Renamed file: {db_path} -> {new_path}")

    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}")


@click.command("migrate-embeddings")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10000, help="SQL batch size (default: 10000)")
@click.option("--embed-batch-size", type=int, default=64, help="Embedder batch size (default: 64)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_migrate_embeddings(db_path: Optional[str], batch_size: int, embed_batch_size: int, verbose: bool):
    """
    Migrate embeddings from old vec0 tables to the new embedding column.

    Copies float32 embeddings from the legacy sqlite-vec vec0 virtual tables
    (organization_embeddings, person_embeddings) into the embedding BLOB column
    on the main organizations/people tables. Generates any missing embeddings,
    enforces NOT NULL on the embedding column, then VACUUMs.

    Requires sqlite-vec to be installed to read from vec0 tables. Any records
    not found in vec0 tables will have embeddings generated from scratch.

    \b
    Steps:
    1. Copy embeddings from vec0 tables to main table column
    2. Generate embeddings for any records still missing them
    3. Enforce NOT NULL on embedding columns (table rebuild)
    4. Drop legacy vec0 tables
    5. VACUUM

    \b
    Examples:
        corp-entity-db migrate-embeddings
        corp-entity-db migrate-embeddings --db /path/to/entities.db
        corp-entity-db migrate-embeddings -v
    """
    import sqlite3

    _configure_logging(verbose)

    from corp_entity_db import CompanyEmbedder
    from corp_entity_db.schema_v2 import CREATE_ORGANIZATIONS_V2_INDEXES, CREATE_PEOPLE_V2_INDEXES

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    conn = sqlite3.connect(str(db_path_obj))
    conn.row_factory = sqlite3.Row

    # Performance pragmas for bulk operations
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA mmap_size=268435456")   # 256MB mmap
    conn.execute("PRAGMA cache_size=-500000")     # 500MB page cache
    conn.execute("PRAGMA threads=4")              # helper threads for sorting/subqueries

    # Ensure embedding column exists on both tables (may be missing on older schemas)
    for table in ["organizations", "people"]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding BLOB DEFAULT NULL")
            click.echo(f"  Added embedding column to {table}", err=True)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # --- Step 1: Copy embeddings from vec0 tables ---
    click.echo("\n=== Step 1: Copy embeddings from vec0 tables ===", err=True)

    # Try to load sqlite-vec for reading vec0 tables
    _sqlite_vec_mod = None
    try:
        import sqlite_vec as _sqlite_vec_mod
        conn.enable_load_extension(True)
        _sqlite_vec_mod.load(conn)
        conn.enable_load_extension(False)
        click.echo("  sqlite-vec loaded successfully", err=True)
    except (ImportError, Exception) as e:
        click.echo(f"  sqlite-vec not available ({e}), will generate all embeddings from scratch", err=True)

    vec0_tables = {
        "organizations": "organization_embeddings",
        "people": "person_embeddings",
    }
    id_columns = {
        "organizations": "org_id",
        "people": "person_id",
    }

    for table, vec_table in vec0_tables.items():
        # Check if the vec0 table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE name=?", (vec_table,)
        )
        if not cursor.fetchone():
            click.echo(f"  {vec_table}: not found, skipping", err=True)
            continue

        if _sqlite_vec_mod is None:
            click.echo(f"  {vec_table}: exists but sqlite-vec not loaded, cannot read", err=True)
            continue

        id_col = id_columns[table]

        # Get ID range and count for batching by ID bands (avoids full table scans)
        row = conn.execute(
            f"SELECT COUNT(*) as cnt, MIN({id_col}) as min_id, MAX({id_col}) as max_id "
            f"FROM {vec_table} WHERE embedding IS NOT NULL"
        ).fetchone()
        total, min_id, max_id = row["cnt"], row["min_id"], row["max_id"]
        click.echo(f"  Copying {total:,} embeddings from {vec_table} (IDs {min_id}-{max_id})...", err=True)

        # Bulk UPDATE in ID-range batches using UPDATE ... FROM (SQLite 3.33+)
        copy_batch = 100_000
        copied = 0
        batch_num = 0
        range_start = min_id
        while range_start <= max_id:
            batch_num += 1
            range_end = range_start + copy_batch - 1
            cursor = conn.execute(
                f"UPDATE {table} SET embedding = v.embedding "
                f"FROM {vec_table} v WHERE {table}.id = v.{id_col} "
                f"AND v.embedding IS NOT NULL "
                f"AND v.{id_col} >= ? AND v.{id_col} <= ?",
                (range_start, range_end),
            )
            affected = cursor.rowcount
            conn.commit()
            copied += affected
            if affected > 0:
                click.echo(f"  {table} batch {batch_num}: {affected:,} rows ({copied:,}/{total:,})", err=True)
            range_start = range_end + 1

        click.echo(f"  {vec_table}: copied {copied:,} embeddings total", err=True)

        # Drop all related vec0/shadow tables now that data is copied
        click.echo(f"  Dropping legacy embedding tables for {table}...", err=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"{vec_table}%",),
        )
        for row in cursor.fetchall():
            conn.execute(f"DROP TABLE IF EXISTS [{row['name']}]")
            click.echo(f"    Dropped {row['name']}", err=True)
        conn.commit()

        # VACUUM to reclaim space
        click.echo(f"  Vacuuming after {table}...", err=True)
        conn.close()
        vacuum_conn = sqlite3.connect(str(db_path_obj))
        vacuum_conn.execute("VACUUM")
        vacuum_conn.close()
        conn = sqlite3.connect(str(db_path_obj))
        conn.row_factory = sqlite3.Row
        if _sqlite_vec_mod is not None:
            conn.enable_load_extension(True)
            _sqlite_vec_mod.load(conn)
            conn.enable_load_extension(False)

    # --- Step 2: Generate missing embeddings ---
    click.echo("\n=== Step 2: Generate missing embeddings ===", err=True)

    embedder = None  # Lazy load

    for table in ["organizations", "people"]:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE embedding IS NULL"
        )
        missing_count = cursor.fetchone()[0]

        if missing_count == 0:
            click.echo(f"  {table}: all embeddings present", err=True)
            continue

        if embedder is None:
            click.echo("  Loading embedding model...", err=True)
            embedder = CompanyEmbedder()

        generated = 0
        with click.progressbar(length=missing_count, label=f"  {table}", file=click.get_text_stream("stderr")) as bar:
            while True:
                cursor = conn.execute(
                    f"SELECT id, name FROM {table} WHERE embedding IS NULL LIMIT ?",
                    (batch_size,),
                )
                rows = cursor.fetchall()
                if not rows:
                    break

                for i in range(0, len(rows), embed_batch_size):
                    sub_batch = rows[i : i + embed_batch_size]
                    ids = [r["id"] for r in sub_batch]
                    names = [r["name"] for r in sub_batch]

                    embeddings = embedder.embed_batch(names, batch_size=embed_batch_size)

                    for record_id, emb in zip(ids, embeddings):
                        embedding_blob = emb.astype("float32").tobytes()
                        conn.execute(
                            f"UPDATE {table} SET embedding = ? WHERE id = ?",
                            (embedding_blob, record_id),
                        )

                    generated += len(sub_batch)
                    bar.update(len(sub_batch))

                conn.commit()

        click.echo(f"  {table}: generated {generated:,} embeddings", err=True)

    # --- Step 3: Enforce NOT NULL on embedding columns ---
    click.echo("\n=== Step 3: Enforce NOT NULL on embedding columns ===", err=True)

    index_ddl = {
        "organizations": CREATE_ORGANIZATIONS_V2_INDEXES,
        "people": CREATE_PEOPLE_V2_INDEXES,
    }

    for table in ["organizations", "people"]:
        # Verify no NULLs remain
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE embedding IS NULL"
        )
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            raise click.ClickException(
                f"Cannot enforce NOT NULL on {table}.embedding: "
                f"{null_count:,} records still have NULL embeddings"
            )

        # Get the current CREATE TABLE statement
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        original_sql = cursor.fetchone()["sql"]

        if "embedding BLOB NOT NULL" in original_sql:
            click.echo(f"  {table}: already NOT NULL", err=True)
            continue

        # Replace the column definition
        new_sql = original_sql.replace(
            "embedding BLOB DEFAULT NULL",
            "embedding BLOB NOT NULL",
        )
        new_sql = new_sql.replace(
            f"CREATE TABLE {table}",
            f"CREATE TABLE {table}_new",
            1,
        )

        # Get column names
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row["name"] for row in cursor.fetchall()]
        col_list = ", ".join(columns)

        click.echo(f"  Rebuilding {table} with NOT NULL constraint...", err=True)
        conn.execute(new_sql)
        conn.execute(
            f"INSERT INTO {table}_new ({col_list}) SELECT {col_list} FROM {table}"
        )
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")

        # Recreate indexes
        for stmt in index_ddl[table].strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

        conn.commit()
        click.echo(f"  {table}: embedding column is now NOT NULL", err=True)

    # --- Step 4: Final VACUUM ---
    click.echo("\n=== Step 4: VACUUM ===", err=True)
    conn.close()

    # VACUUM requires its own connection (no active transactions)
    vacuum_conn = sqlite3.connect(str(db_path_obj))
    vacuum_conn.execute("VACUUM")
    vacuum_conn.close()

    db_size_mb = db_path_obj.stat().st_size / 1024**2
    click.echo(f"  Database size after VACUUM: {db_size_mb:,.0f} MB", err=True)

    click.echo("\nEmbedding migration complete!", err=True)


@click.command("migrate-v2")
@click.argument("source_db", type=click.Path(exists=True))
@click.argument("target_db", type=click.Path())
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--resume", is_flag=True, help="Resume from last completed step")
def db_migrate_v2(source_db: str, target_db: str, verbose: bool, resume: bool):
    """
    Migrate database from v1 schema to v2 normalized schema.

    Creates a NEW database file with the v2 normalized schema.
    The original database is preserved unchanged.

    Use --resume to continue a migration that was interrupted.

    \b
    V2 changes:
    - TEXT enum fields replaced with INTEGER foreign keys
    - New enum lookup tables (source_types, people_types, etc.)
    - New roles and locations tables
    - QIDs stored as integers (Q prefix stripped)
    - Human-readable views for queries

    \b
    Examples:
        corp-entity-db migrate-v2 entities.db entities-v2.db
        corp-entity-db migrate-v2 entities.db entities-v2.db --resume
        corp-entity-db migrate-v2 ~/.cache/corp-entity-db/entities.db ./entities-v2.db -v
    """
    _configure_logging(verbose)

    from pathlib import Path
    from corp_entity_db.migrate_v2 import migrate_database

    source_path = Path(source_db)
    target_path = Path(target_db)

    if target_path.exists() and not resume:
        raise click.ClickException(
            f"Target database already exists: {target_path}\n"
            "Use --resume to continue an interrupted migration."
        )

    if resume:
        click.echo(f"Resuming migration from {source_path} to {target_path}...")
    else:
        click.echo(f"Migrating {source_path} to {target_path}...")

    try:
        stats = migrate_database(source_path, target_path, resume=resume)

        click.echo("\nMigration complete:")
        for key, value in stats.items():
            click.echo(f"  {key}: {value:,}")

    except Exception as e:
        raise click.ClickException(f"Migration failed: {e}")
