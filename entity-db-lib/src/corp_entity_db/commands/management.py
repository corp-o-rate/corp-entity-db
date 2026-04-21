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
        click.echo(f"Matched by QID: {people_result['matched_by_qid']:,}")
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
    - Full database + lite variant (without record data or name_normalized)
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
    Create a lite version of the database without record data.

    The lite version strips the `record` column content. The `name_normalized`
    column is kept on all tables (required by SQL name-lookup fallback in search).

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


@click.command("build-index")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--M", type=int, default=32, help="Number of connections per node (default 32)")
@click.option("--ef-construction", type=int, default=200, help="Construction quality parameter (default 200)")
@click.option("--ef-search", type=int, default=200, help="Search quality parameter (default 200)")
@click.option("--people/--no-people", default=True, help="Build USearch index for people")
@click.option("--orgs/--no-orgs", default=True, help="Build USearch index for organizations")
@click.option("--sharded/--no-sharded", default=True, help="Build popularity-sharded indexes for people (default: sharded)")
@click.option("--hot-size", type=int, default=3_000_000, help="Hot shard size (default: 3M vectors)")
@click.option("--cold-size", type=int, default=10_000_000, help="Cold shard size (default: 10M vectors)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_build_index(db_path: Optional[str], m: int, ef_construction: int, ef_search: int, people: bool, orgs: bool, sharded: bool, hot_size: int, cold_size: int, verbose: bool):
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
        ShardConfig,
        _get_shared_connection,
        build_hnsw_index,
        build_people_composite_index,
        build_people_identity_index,
    )
    from corp_entity_db.embeddings import CompanyEmbedder

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    shard_cfg = ShardConfig(hot_size=hot_size, cold_size=cold_size) if sharded else None

    click.echo(f"Database: {db_path_obj}", err=True)
    click.echo(f"Parameters: M={m}, ef_construction={ef_construction}, ef_search={ef_search}", err=True)
    if shard_cfg:
        click.echo(f"Sharding: hot={shard_cfg.hot_size:,}, cold={shard_cfg.cold_size:,}", err=True)

    conn = _get_shared_connection(db_path_obj, readonly=True)

    embedder = None  # Lazy load — created when first needed

    if people:
        click.echo("\n--- Building composite USearch index for people ---", err=True)

        if embedder is None:
            embedder = CompanyEmbedder()

        def people_progress(done: int, total: int) -> None:
            click.echo(f"  Generated + indexed {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_people_composite_index(
                conn, embedder,
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=people_progress,
                shard_config=shard_cfg,
            )
            click.echo(f"People USearch index: {count:,} composite vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"People USearch index build failed: {e}")

        click.echo("\n--- Building identity USearch index for people ---", err=True)

        def identity_progress(done: int, total: int) -> None:
            click.echo(f"  Generated + indexed {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_people_identity_index(
                conn, embedder,
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=identity_progress,
                shard_config=shard_cfg,
            )
            click.echo(f"People identity index: {count:,} name vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"People identity index build failed: {e}")

    if orgs:
        click.echo("\n--- Building USearch index for organizations ---", err=True)

        if embedder is None:
            embedder = CompanyEmbedder()

        def orgs_progress(done: int, total: int) -> None:
            click.echo(f"  Generated + indexed {done:,}/{total:,} vectors...", err=True)

        try:
            count = build_hnsw_index(
                conn, "organizations",
                embedder=embedder,
                M=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                progress_callback=orgs_progress,
            )
            click.echo(f"Organizations USearch index: {count:,} vectors indexed", err=True)
        except Exception as e:
            raise click.ClickException(f"Organizations USearch index build failed: {e}")

    click.echo("\nUSearch index build complete!", err=True)


@click.command("build-identity-index")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--M", type=int, default=32, help="Number of connections per node (default 32)")
@click.option("--ef-construction", type=int, default=200, help="Construction quality parameter (default 200)")
@click.option("--ef-search", type=int, default=200, help="Search quality parameter (default 200)")
@click.option("--batch-size", "embed_batch_size", type=int, default=192, help="Embedding batch size (default 192)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_build_identity_index(db_path: Optional[str], m: int, ef_construction: int, ef_search: int, embed_batch_size: int, verbose: bool):
    """
    Build identity (name-only) USearch index for people.

    Embeds each person's name, truncates to 256 dims (Matryoshka),
    and builds an HNSW index. Used as a fallback when composite search
    and SQL name lookup both fail.

    \b
    Examples:
        corp-entity-db build-identity-index
        corp-entity-db build-identity-index --batch-size 384
    """
    _configure_logging(verbose)

    from corp_entity_db.embeddings import CompanyEmbedder
    from corp_entity_db.store import _get_shared_connection, build_people_identity_index

    db_path_obj = _resolve_db_path(db_path)

    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    conn = _get_shared_connection(db_path_obj, readonly=True)
    embedder = CompanyEmbedder()

    def progress(done: int, total: int) -> None:
        click.echo(f"  Generated + indexed {done:,}/{total:,} vectors...", err=True)

    try:
        count = build_people_identity_index(
            conn, embedder,
            M=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            embed_batch_size=embed_batch_size,
            progress_callback=progress,
        )
        click.echo(f"Identity index built: {count:,} name vectors indexed", err=True)
    except Exception as e:
        raise click.ClickException(f"Identity index build failed: {e}")


@click.command("migrate")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_migrate(db_path: Optional[str], verbose: bool):
    """
    Migrate database schema to the latest version (v6).

    v4: Drops legacy embedding columns from organizations and people tables.
    v5: Merges scientist→academic and entrepreneur→executive person types.
    v6: Adds organization alias columns and new alias source types.

    \b
    Examples:
        corp-entity-db migrate
        corp-entity-db migrate --db /path/to/entities.db
    """
    import logging
    import sqlite3

    _configure_logging(verbose)
    logger = logging.getLogger(__name__)

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    conn = sqlite3.connect(str(db_path_obj), isolation_level=None)

    # Read current schema version
    current_version = 0
    try:
        row = conn.execute("SELECT value FROM db_info WHERE key = 'schema_version'").fetchone()
        if row:
            current_version = int(row[0])
    except sqlite3.OperationalError:
        pass  # db_info table doesn't exist

    click.echo(f"Current schema version: {current_version}", err=True)

    # Check if organizations table needs repair (CREATE TABLE AS SELECT lost constraints)
    needs_repair = False
    if current_version >= 6:
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='organizations'"
        ).fetchone()
        if create_sql and "id INT," in create_sql[0] and "PRIMARY KEY" not in create_sql[0]:
            click.echo("Detected broken schema (missing PRIMARY KEY on organizations) — repairing...", err=True)
            needs_repair = True
        else:
            click.echo("Already up to date (v6).", err=True)
            conn.close()
            return

    # --- v4 migration: drop embedding columns ---
    if current_version < 4:
        # Drop embedding column from organizations if present
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(organizations)").fetchall()}
            if "embedding" in cols:
                conn.execute("ALTER TABLE organizations DROP COLUMN embedding")
                logger.info("Dropped embedding column from organizations")
                click.echo("  Dropped embedding column from organizations", err=True)
            else:
                click.echo("  organizations: no embedding column (already clean)", err=True)
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not drop embedding from organizations: {e}")

        # Drop embedding column from people if present
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(people)").fetchall()}
            if "embedding" in cols:
                conn.execute("ALTER TABLE people DROP COLUMN embedding")
                logger.info("Dropped embedding column from people")
                click.echo("  Dropped embedding column from people", err=True)
            else:
                click.echo("  people: no embedding column (already clean)", err=True)
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not drop embedding from people: {e}")

        conn.execute("INSERT OR REPLACE INTO db_info (key, value) VALUES ('schema_version', '4')")
        click.echo("  Updated schema_version to 4", err=True)

    # --- v5 migration: merge scientist→academic, entrepreneur→executive ---
    if current_version < 5:
        click.echo("\n  Merging person types: scientist→academic, entrepreneur→executive", err=True)

        # Count records to remap
        row = conn.execute("SELECT COUNT(*) FROM people WHERE person_type_id = 14").fetchone()
        scientist_count = row[0] if row else 0
        row = conn.execute("SELECT COUNT(*) FROM people WHERE person_type_id = 11").fetchone()
        entrepreneur_count = row[0] if row else 0

        logger.info(f"Remapping {scientist_count:,} scientist records to academic (14→7)")
        logger.info(f"Remapping {entrepreneur_count:,} entrepreneur records to executive (11→1)")

        conn.execute("BEGIN")
        # Remap scientist (14) → academic (7)
        conn.execute("UPDATE people SET person_type_id = 7 WHERE person_type_id = 14")
        click.echo(f"  Remapped {scientist_count:,} scientist → academic", err=True)

        # Remap entrepreneur (11) → executive (1)
        conn.execute("UPDATE people SET person_type_id = 1 WHERE person_type_id = 11")
        click.echo(f"  Remapped {entrepreneur_count:,} entrepreneur → executive", err=True)

        # Remove obsolete type rows
        conn.execute("DELETE FROM people_types WHERE id IN (11, 14)")
        click.echo("  Removed obsolete people_types rows (scientist, entrepreneur)", err=True)

        conn.execute("INSERT OR REPLACE INTO db_info (key, value) VALUES ('schema_version', '5')")
        conn.execute("COMMIT")
        click.echo("  Updated schema_version to 5", err=True)

    # --- v6 migration: organization alias columns + new source types ---
    if current_version < 6 or needs_repair:
        click.echo("\n  v6: Adding organization alias columns...", err=True)
        conn.execute("BEGIN")

        # Insert new alias source types
        alias_source_types = [
            (5, "wikidata_alias"),
            (6, "gleif_other_name"),
            (7, "sec_ticker"),
            (8, "wiki_anchor"),
            (9, "paranames"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO source_types (id, name) VALUES (?, ?)",
            alias_source_types,
        )
        click.echo(f"  Inserted {len(alias_source_types)} alias source types", err=True)

        # Add alias columns
        cols = {r[1] for r in conn.execute("PRAGMA table_info(organizations)").fetchall()}
        if "alias_source_id" not in cols:
            conn.execute("ALTER TABLE organizations ADD COLUMN alias_source_id INTEGER DEFAULT NULL REFERENCES source_types(id)")
            click.echo("  Added alias_source_id column", err=True)
        if "alias_source_identifier" not in cols:
            conn.execute("ALTER TABLE organizations ADD COLUMN alias_source_identifier TEXT DEFAULT NULL")
            click.echo("  Added alias_source_identifier column", err=True)

        # Rebuild unique constraint: drop old UNIQUE(source_identifier, source_id)
        # and replace with UNIQUE INDEX including alias_source_id and name_normalized.
        # SQLite requires table rebuild to drop a table-level UNIQUE constraint.
        click.echo("  Rebuilding organizations table to replace unique constraint...", err=True)

        # Get current column list (after ALTER TABLEs above)
        col_info = conn.execute("PRAGMA table_info(organizations)").fetchall()
        col_names = [r[1] for r in col_info]

        # Check if old UNIQUE constraint exists on (source_identifier, source_id)
        # by looking at sqlite_master for the table's CREATE statement
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='organizations'"
        ).fetchone()[0]

        needs_rebuild = "UNIQUE(source_identifier, source_id)" in create_sql or "id INT," in create_sql
        if needs_rebuild:
            # Drop views that reference organizations before rebuilding
            conn.execute("DROP VIEW IF EXISTS organizations_view")
            conn.execute("DROP VIEW IF EXISTS people_view")

            # Use proper CREATE TABLE with constraints (CREATE TABLE AS SELECT loses them)
            conn.execute("""
                CREATE TABLE organizations_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qid INTEGER,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    source_identifier TEXT NOT NULL,
                    region_id INTEGER,
                    entity_type_id INTEGER NOT NULL DEFAULT 17,
                    from_date TEXT DEFAULT NULL,
                    to_date TEXT DEFAULT NULL,
                    record TEXT NOT NULL DEFAULT '{}',
                    canon_id INTEGER DEFAULT NULL,
                    canon_size INTEGER DEFAULT 1,
                    alias_source_id INTEGER DEFAULT NULL,
                    alias_source_identifier TEXT DEFAULT NULL,
                    FOREIGN KEY (source_id) REFERENCES source_types(id),
                    FOREIGN KEY (region_id) REFERENCES locations(id),
                    FOREIGN KEY (entity_type_id) REFERENCES organization_types(id),
                    FOREIGN KEY (alias_source_id) REFERENCES source_types(id)
                )
            """)
            cols_csv = ", ".join(col_names)
            conn.execute(f"INSERT INTO organizations_new ({cols_csv}) SELECT {cols_csv} FROM organizations")
            conn.execute("DROP TABLE organizations")
            conn.execute("ALTER TABLE organizations_new RENAME TO organizations")
            click.echo("  Rebuilt organizations table with proper constraints", err=True)

            # Recreate people_view (references organizations via known_for_org_id)
            conn.execute("""
                CREATE VIEW IF NOT EXISTS people_view AS
                SELECT
                    p.id, p.qid, p.name, p.name_normalized,
                    s.name as source, p.source_identifier,
                    l.name as country, pt.name as person_type,
                    r.name as known_for_role, kfo.name as known_for_org,
                    p.known_for_org_id, p.from_date, p.to_date,
                    p.birth_date, p.death_date, p.canon_id, p.canon_size
                FROM people p
                JOIN source_types s ON p.source_id = s.id
                LEFT JOIN locations l ON p.country_id = l.id
                JOIN people_types pt ON p.person_type_id = pt.id
                LEFT JOIN roles r ON p.known_for_role_id = r.id
                LEFT JOIN organizations kfo ON p.known_for_org_id = kfo.id
            """)

        # Create new unique index
        conn.execute("DROP INDEX IF EXISTS idx_orgs_unique")
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_unique ON organizations(
                source_identifier, source_id, IFNULL(alias_source_id, 0), name_normalized
            )
        """)
        click.echo("  Created new unique index (source_identifier, source_id, alias_source_id, name_normalized)", err=True)

        # Re-create other indexes that may have been lost during table rebuild
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_qid ON organizations(qid)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_source_identifier ON organizations(source_identifier)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_region_id ON organizations(region_id)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_entity_type_id ON organizations(entity_type_id)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_canon_id ON organizations(canon_id)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_alias_source_id ON organizations(alias_source_id)",
        ]:
            conn.execute(idx_sql)

        # Recreate organizations_view with alias columns
        conn.execute("DROP VIEW IF EXISTS organizations_view")
        conn.execute("""
            CREATE VIEW organizations_view AS
            SELECT
                o.id,
                o.qid,
                o.name,
                o.name_normalized,
                s.name as source,
                o.source_identifier,
                l.name as region,
                slt.name as region_type,
                ot.name as entity_type,
                o.from_date,
                o.to_date,
                o.canon_id,
                o.canon_size,
                als.name as alias_source,
                o.alias_source_identifier
            FROM organizations o
            JOIN source_types s ON o.source_id = s.id
            LEFT JOIN locations l ON o.region_id = l.id
            LEFT JOIN location_types lt ON l.location_type_id = lt.id
            LEFT JOIN simplified_location_types slt ON lt.simplified_id = slt.id
            JOIN organization_types ot ON o.entity_type_id = ot.id
            LEFT JOIN source_types als ON o.alias_source_id = als.id
        """)
        click.echo("  Recreated organizations_view with alias columns", err=True)

        conn.execute("INSERT OR REPLACE INTO db_info (key, value) VALUES ('schema_version', '6')")
        conn.execute("COMMIT")
        click.echo("  Updated schema_version to 6", err=True)

    # VACUUM
    click.echo("  Running VACUUM...", err=True)
    conn.execute("VACUUM")
    conn.close()

    db_size_mb = db_path_obj.stat().st_size / 1024**2
    click.echo(f"Migration complete. Database size: {db_size_mb:.1f} MB", err=True)


@click.command("populate-aliases")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_populate_aliases(db_path: Optional[str], verbose: bool):
    """
    Create alias records from existing organization record JSON.

    Scans primary org records and creates alias rows for Wikidata aliases,
    GLEIF other entity names, and SEC tickers. Must be run AFTER canonicalization.

    \b
    Pipeline: import → canonicalize → populate-aliases → build-index

    \b
    Examples:
        corp-entity-db populate-aliases
        corp-entity-db populate-aliases --db /path/to/entities.db
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    database = OrganizationDatabase(db_path=db_path_obj, readonly=False)
    result = database.populate_aliases()

    click.echo(f"\nAlias Population Results", err=True)
    click.echo("=" * 40, err=True)
    click.echo(f"Wikidata aliases:    {result['wikidata_aliases']:,}", err=True)
    click.echo(f"GLEIF other names:   {result['gleif_other_names']:,}", err=True)
    click.echo(f"SEC tickers:         {result['sec_tickers']:,}", err=True)
    click.echo(f"Total inserted:      {result['total']:,}", err=True)

    database.close()


@click.command("backfill-ch-orgs")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=50000, help="Batch size for commits (default: 50000)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_backfill_ch_orgs(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Create stub org records for CH officers with missing organizations.

    Scans CH officer records where known_for_org_id is NULL, extracts the
    company_number and company_name from the record JSON, creates stub
    organization records for companies not already in the DB, and links
    the officers to their orgs.

    \b
    Pipeline: import-companies-house → import-ch-officers → backfill-ch-orgs → post-import

    \b
    Examples:
        corp-entity-db backfill-ch-orgs
        corp-entity-db backfill-ch-orgs --db /path/to/entities.db
    """
    import json
    import sqlite3

    from corp_names import normalize_company

    from corp_entity_db.seed_data import SOURCE_NAME_TO_ID, ORG_TYPE_NAME_TO_ID

    _configure_logging(verbose)

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    click.echo(f"Database: {db_path_obj}", err=True)

    conn = sqlite3.connect(str(db_path_obj))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA mmap_size=268435456")

    ch_source_id = SOURCE_NAME_TO_ID["companies_house"]
    unknown_type_id = ORG_TYPE_NAME_TO_ID["unknown"]

    # Phase 1: Collect missing company_number → company_name from officer records
    click.echo("Phase 1: Scanning CH officers with missing org links...", err=True)

    cursor = conn.execute(
        "SELECT id, record FROM people WHERE source_id = ? AND known_for_org_id IS NULL",
        (ch_source_id,),
    )

    # Map company_number → (company_name, [officer_ids])
    missing: dict[str, tuple[str, list[int]]] = {}
    scanned = 0

    for row in cursor:
        scanned += 1
        rec = json.loads(row["record"]) if row["record"] else {}
        company_number = rec.get("company_number", "")
        company_name = rec.get("company_name", "")
        if not company_number:
            continue

        if company_number in missing:
            missing[company_number][1].append(row["id"])
        else:
            missing[company_number] = (company_name, [row["id"]])

        if scanned % 1_000_000 == 0:
            click.echo(f"  Scanned {scanned:,} officers, {len(missing):,} missing companies...", err=True)

    click.echo(f"  Scanned {scanned:,} officers, {len(missing):,} distinct companies to resolve", err=True)

    if not missing:
        click.echo("No missing org links found.", err=True)
        conn.close()
        return

    # Phase 2: Filter out companies that already exist in the organizations table
    click.echo("Phase 2: Checking which companies already exist...", err=True)

    # Build a lookup of existing CH company numbers → org IDs
    existing_cursor = conn.execute(
        "SELECT id, source_identifier FROM organizations WHERE source_id = ? AND alias_source_id IS NULL",
        (ch_source_id,),
    )
    existing_orgs: dict[str, int] = {}
    for row in existing_cursor:
        existing_orgs[row["source_identifier"]] = row["id"]

    click.echo(f"  {len(existing_orgs):,} existing CH organizations in DB", err=True)

    # Split into already-existing (just need FK update) and truly new
    to_link: list[tuple[int, int]] = []  # (officer_id, org_id)
    to_create: dict[str, tuple[str, list[int]]] = {}  # company_number → (name, [officer_ids])

    for company_number, (company_name, officer_ids) in missing.items():
        if company_number in existing_orgs:
            org_id = existing_orgs[company_number]
            for oid in officer_ids:
                to_link.append((oid, org_id))
        else:
            to_create[company_number] = (company_name, officer_ids)

    click.echo(f"  {len(to_link):,} officers can be linked to existing orgs", err=True)
    click.echo(f"  {len(to_create):,} new stub organizations to create", err=True)

    # Phase 3: Create stub org records and collect all FK updates
    click.echo("Phase 3: Creating stub organizations and linking officers...", err=True)

    created = 0
    for company_number, (company_name, officer_ids) in to_create.items():
        name = company_name.strip() if company_name else company_number
        name_normalized = normalize_company(name).normalized

        cursor = conn.execute("""
            INSERT OR IGNORE INTO organizations
            (name, name_normalized, source_id, source_identifier, entity_type_id, record)
            VALUES (?, ?, ?, ?, ?, '{}')
        """, (name, name_normalized, ch_source_id, company_number, unknown_type_id))

        if cursor.lastrowid and cursor.rowcount > 0:
            org_id = cursor.lastrowid
            created += 1
        else:
            # Already existed (race or IGNORE), look it up
            row = conn.execute(
                "SELECT id FROM organizations WHERE source_id = ? AND source_identifier = ? AND alias_source_id IS NULL",
                (ch_source_id, company_number),
            ).fetchone()
            org_id = row["id"] if row else None

        if org_id:
            for oid in officer_ids:
                to_link.append((oid, org_id))

        if created % 10000 == 0 and created > 0:
            conn.commit()
            click.echo(f"  Created {created:,} stub orgs...", err=True)

    conn.commit()
    click.echo(f"  Created {created:,} stub organizations", err=True)

    # Phase 4: Update all officer FK links
    click.echo(f"Phase 4: Linking {len(to_link):,} officers to organizations...", err=True)

    linked = 0
    skipped = 0
    for officer_id, org_id in to_link:
        try:
            conn.execute("UPDATE people SET known_for_org_id = ? WHERE id = ?", (org_id, officer_id))
            linked += 1
        except sqlite3.IntegrityError:
            skipped += 1

        if (linked + skipped) % batch_size == 0:
            conn.commit()
            click.echo(f"  Linked {linked:,} officers ({skipped:,} skipped)...", err=True)

    conn.commit()

    click.echo(f"\nBackfill complete!", err=True)
    click.echo(f"  Stub orgs created:  {created:,}", err=True)
    click.echo(f"  Officers linked:    {linked:,}", err=True)
    if skipped:
        click.echo(f"  Skipped (UNIQUE):   {skipped:,}", err=True)
    click.echo("Run `corp-entity-db post-import` to rebuild search indexes.", err=True)

    conn.close()


def _run_post_import(db_path_obj: Path, people: bool = True, orgs: bool = True, embed_batch_size: int = 192) -> None:
    """
    Standard post-import steps: build USearch indexes, VACUUM.

    All embeddings (orgs + people) are generated on-the-fly during index building.
    No embeddings are stored in SQLite.

    Called automatically after imports or manually via `db post-import`.
    """
    import sqlite3

    from corp_entity_db.embeddings import CompanyEmbedder
    from corp_entity_db.store import (
        ShardConfig,
        _get_shared_connection,
        build_hnsw_index,
        build_people_composite_index,
        build_people_identity_index,
    )

    shard_cfg = ShardConfig()  # Default sharding for post-import
    embedder = None  # Lazy load

    # --- Step 1: Build USearch indexes ---
    click.echo("\n=== Step 1: Build USearch indexes ===", err=True)

    conn = _get_shared_connection(db_path_obj, readonly=True)

    if people:
        if embedder is None:
            click.echo("  Loading embedding model...", err=True)
            embedder = CompanyEmbedder()

        def people_progress(done: int, total: int) -> None:
            click.echo(f"  People: {done:,}/{total:,} vectors...", err=True)

        count = build_people_composite_index(
            conn, embedder, embed_batch_size=embed_batch_size,
            progress_callback=people_progress,
            shard_config=shard_cfg,
        )
        click.echo(f"  People USearch index: {count:,} composite vectors", err=True)

        def identity_progress(done: int, total: int) -> None:
            click.echo(f"  Identity: {done:,}/{total:,} vectors...", err=True)

        count = build_people_identity_index(
            conn, embedder, embed_batch_size=embed_batch_size,
            progress_callback=identity_progress,
            shard_config=shard_cfg,
        )
        click.echo(f"  People identity index: {count:,} name vectors", err=True)

    if orgs:
        if embedder is None:
            click.echo("  Loading embedding model...", err=True)
            embedder = CompanyEmbedder()

        def orgs_progress(done: int, total: int) -> None:
            click.echo(f"  Organizations: {done:,}/{total:,} vectors...", err=True)

        count = build_hnsw_index(conn, "organizations", embedder=embedder,
                                 embed_batch_size=embed_batch_size,
                                 progress_callback=orgs_progress)
        click.echo(f"  Organizations USearch index: {count:,} vectors", err=True)

    # --- Step 2: VACUUM ---
    click.echo("\n=== Step 2: VACUUM database ===", err=True)
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
@click.option("--batch-size", "embed_batch_size", type=int, default=192, help="Embedding batch size (default: 192)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_post_import(db_path: Optional[str], people: bool, orgs: bool, embed_batch_size: int, verbose: bool):
    """
    Run standard post-import steps: build USearch indexes, VACUUM.

    Run this after any import command to ensure search indexes are up to date.

    \b
    Steps:
    1. Build USearch HNSW indexes for fast search (embeddings generated on-the-fly)
    2. VACUUM database to reclaim space

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
    _run_post_import(db_path_obj, people=people, orgs=orgs, embed_batch_size=embed_batch_size)


@click.command("reclassify-people")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10_000, help="Batch size for updates")
@click.option("--dry-run", is_flag=True, help="Show changes without writing to DB")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_reclassify_people(db_path: Optional[str], batch_size: int, dry_run: bool, verbose: bool):
    """
    Recalculate person_type for all people using current classification rules.

    Re-applies the position (P39) and occupation (P106) classification logic
    from the Wikidata dump importer to every person record that has stored
    positions/occupations in its record JSON.

    \b
    Examples:
        corp-entity-db reclassify-people
        corp-entity-db reclassify-people --dry-run
        corp-entity-db reclassify-people -v
    """
    import json
    import logging
    import sqlite3
    import time

    _configure_logging(verbose)
    logger = logging.getLogger("corp_entity_db")

    from corp_entity_db.importers.wikidata_dump import (
        EXECUTIVE_POSITION_QIDS,
        OCCUPATION_TO_TYPE,
        POLITICIAN_POSITION_QIDS,
    )
    from corp_entity_db.models import PersonType
    from corp_entity_db.seed_data import PEOPLE_TYPE_NAME_TO_ID

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    conn = sqlite3.connect(str(db_path_obj))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-500000")

    total = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    click.echo(f"Reclassifying {total:,} people in {db_path_obj}", err=True)
    if dry_run:
        click.echo("DRY RUN — no changes will be written", err=True)

    def classify(positions: list[str], occupations: list[str]) -> PersonType:
        """Classify person type from position QIDs and occupation QIDs."""
        for qid in positions:
            if qid in EXECUTIVE_POSITION_QIDS:
                return PersonType.EXECUTIVE
            if qid in POLITICIAN_POSITION_QIDS:
                return PersonType.POLITICIAN
        for qid in occupations:
            if qid in OCCUPATION_TO_TYPE:
                return OCCUPATION_TO_TYPE[qid]
        return PersonType.UNKNOWN

    updated = 0
    unchanged = 0
    last_id = 0
    start = time.time()
    changes_by_type: dict[str, int] = {}

    while True:
        rows = conn.execute(
            "SELECT id, person_type_id, record FROM people WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break

        updates: list[tuple[int, int]] = []
        for row_id, current_type_id, record_json in rows:
            if not record_json:
                unchanged += 1
                continue

            rec = json.loads(record_json)
            positions = rec.get("positions", [])
            occupations = rec.get("occupations", [])

            new_type = classify(positions, occupations)
            new_type_id = PEOPLE_TYPE_NAME_TO_ID[new_type.value]

            if new_type_id != current_type_id:
                updates.append((new_type_id, row_id))
                key = f"{current_type_id}->{new_type_id}"
                changes_by_type[key] = changes_by_type.get(key, 0) + 1
            else:
                unchanged += 1

        if updates and not dry_run:
            conn.executemany(
                "UPDATE people SET person_type_id = ? WHERE id = ?",
                updates,
            )
            conn.commit()

        updated += len(updates)
        last_id = rows[-1][0]
        elapsed = time.time() - start
        processed = updated + unchanged
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(f"Processed {processed:,}/{total:,} ({rate:,.0f}/s) — {updated:,} changed")

    elapsed = time.time() - start
    click.echo(f"\nDone in {elapsed:.1f}s. {updated:,} changed, {unchanged:,} unchanged.", err=True)

    if changes_by_type:
        # Resolve type IDs to names for display
        type_id_to_name: dict[int, str] = {}
        for name, tid in PEOPLE_TYPE_NAME_TO_ID.items():
            type_id_to_name[tid] = name

        click.echo("\nChanges by type:", err=True)
        for key, count in sorted(changes_by_type.items(), key=lambda x: -x[1]):
            from_id, to_id = key.split("->")
            from_name = type_id_to_name.get(int(from_id), from_id)
            to_name = type_id_to_name.get(int(to_id), to_id)
            click.echo(f"  {from_name} -> {to_name}: {count:,}", err=True)

    conn.close()


@click.command("normalize-people")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10_000, help="Batch size for updates")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_normalize_people(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Normalize all people names using corp-names and store in name_normalized column.

    Strips titles, suffixes, middle initials, and resolves nicknames to canonical forms.
    E.g. "Dr. Robert S. Mueller III" → "robert mueller"

    \b
    Examples:
        corp-entity-db normalize-people
        corp-entity-db normalize-people --batch-size 50000
    """
    import logging
    import sqlite3
    import time

    from corp_names import normalize_name

    _configure_logging(verbose)
    logger = logging.getLogger("corp_entity_db")

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    conn = sqlite3.connect(str(db_path_obj))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-500000")

    total = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    click.echo(f"Normalizing {total:,} people names in {db_path_obj}", err=True)

    updated = 0
    last_id = 0
    start = time.time()

    while True:
        rows = conn.execute(
            "SELECT id, name FROM people WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break

        updates: list[tuple[str, int]] = []
        for row_id, name in rows:
            result = normalize_name(name)
            updates.append((result.normalized, row_id))

        conn.executemany(
            "UPDATE people SET name_normalized = ? WHERE id = ?",
            updates,
        )
        conn.commit()

        updated += len(rows)
        last_id = rows[-1][0]
        elapsed = time.time() - start
        rate = updated / elapsed if elapsed > 0 else 0
        logger.info(f"Normalized {updated:,}/{total:,} ({rate:,.0f}/s)")

    conn.close()
    elapsed = time.time() - start
    click.echo(f"Done. Normalized {updated:,} names in {elapsed:.1f}s", err=True)


@click.command("normalize-orgs")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--batch-size", type=int, default=10_000, help="Batch size for updates")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_normalize_orgs(db_path: Optional[str], batch_size: int, verbose: bool):
    """
    Normalize all organization names using corp-names and store in name_normalized column.

    Strips legal suffixes (Inc, Ltd, AG, etc.), lowercases, and canonicalizes.
    E.g. "Apple Inc." → "apple", "Amazon.com, Inc." → "amazon com"

    \b
    Examples:
        corp-entity-db normalize-orgs
        corp-entity-db normalize-orgs --batch-size 50000
    """
    import logging
    import sqlite3
    import time

    from corp_names import normalize_company

    _configure_logging(verbose)
    logger = logging.getLogger("corp_entity_db")

    db_path_obj = _resolve_db_path(db_path)
    if not db_path_obj.exists():
        raise click.ClickException(f"Database not found: {db_path_obj}")

    conn = sqlite3.connect(str(db_path_obj))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-500000")

    total = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()[0]
    click.echo(f"Normalizing {total:,} organization names in {db_path_obj}", err=True)

    updated = 0
    last_id = 0
    start = time.time()

    while True:
        rows = conn.execute(
            "SELECT id, name FROM organizations WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break

        updates: list[tuple[str, int]] = []
        for row_id, name in rows:
            result = normalize_company(name)
            updates.append((result.normalized, row_id))

        conn.executemany(
            "UPDATE organizations SET name_normalized = ? WHERE id = ?",
            updates,
        )
        conn.commit()

        updated += len(rows)
        last_id = rows[-1][0]
        elapsed = time.time() - start
        rate = updated / elapsed if elapsed > 0 else 0
        logger.info(f"Normalized {updated:,}/{total:,} ({rate:,.0f}/s)")

    conn.close()
    elapsed = time.time() - start
    click.echo(f"Done. Normalized {updated:,} names in {elapsed:.1f}s", err=True)

