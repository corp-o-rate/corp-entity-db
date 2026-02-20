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
    fast query times on millions of vectors. Much faster than GPU or PCA approaches.

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
                texts = [item[1] for item in sub_batch]
                fp32_batch = embedder.embed_batch(texts, batch_size=embed_batch_size)
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

