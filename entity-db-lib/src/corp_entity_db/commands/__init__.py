"""CLI commands package — main click group and command registration."""

import click

from corp_entity_db import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--db-version", type=int, default=None, hidden=True, help="Database schema version for filenames (default: latest)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def main(ctx: click.Context, db_version: int | None, verbose: bool):
    """
    Manage entity/organization embedding database.

    \b
    Commands:
        import-gleif           Import GLEIF LEI data (~3M records)
        import-sec             Import SEC Edgar bulk data (~100K+ filers)
        import-sec-officers    Import SEC Form 4 officers/directors
        import-ch-officers     Import UK Companies House officers (Prod195)
        import-companies-house Import UK Companies House (~5M records)
        import-wikidata        Import Wikidata organizations (SPARQL, may timeout)
        import-people          Import Wikidata notable people (SPARQL, may timeout)
        import-wikidata-dump   Import from Wikidata JSON dump (recommended)
        canonicalize           Link equivalent records across sources
        status                 Show database status
        search                 Search for an organization
        search-people          Search for a person
        download               Download database from HuggingFace
        upload                 Upload database with lite variant
        create-lite            Create lite version (no record data)
        repair-resume          Backfill people from org executive data (DB only)
        fix-resume             Backfill people by re-scanning Wikidata dump

    \b
    Examples:
        corp-entity-db import-sec --download
        corp-entity-db import-sec-officers --start-year 2023 --limit 10000
        corp-entity-db import-gleif --download --limit 100000
        corp-entity-db import-wikidata-dump --download --limit 50000
        corp-entity-db canonicalize
        corp-entity-db status
        corp-entity-db search "Apple Inc"
        corp-entity-db search-people "Tim Cook"
        corp-entity-db upload entities.db
    """
    ctx.ensure_object(dict)
    ctx.obj["db_version"] = db_version
    ctx.obj["verbose"] = verbose


# Register all commands
from .imports import (
    db_gleif_info,
    db_import_gleif,
    db_import_sec,
    db_import_sec_officers,
    db_import_ch_officers,
    db_import_wikidata,
    db_import_people,
    db_import_companies_house,
    db_import_locations,
)

main.add_command(db_gleif_info)
main.add_command(db_import_gleif)
main.add_command(db_import_sec)
main.add_command(db_import_sec_officers)
main.add_command(db_import_ch_officers)
main.add_command(db_import_wikidata)
main.add_command(db_import_people)
main.add_command(db_import_companies_house)
main.add_command(db_import_locations)

from .wikidata_dump import db_import_wikidata_dump

main.add_command(db_import_wikidata_dump)

from .search import (
    db_search,
    db_search_people,
    db_search_people_perf_test,
    db_search_roles,
    db_search_locations,
)

main.add_command(db_search)
main.add_command(db_search_people)
main.add_command(db_search_people_perf_test)
main.add_command(db_search_roles)
main.add_command(db_search_locations)

from .management import (
    db_status,
    db_canonicalize,
    db_download,
    db_upload,
    db_create_lite,
    db_repair_embeddings,
    db_build_index,
    db_post_import,
)

main.add_command(db_status)
main.add_command(db_canonicalize)
main.add_command(db_download)
main.add_command(db_upload)
main.add_command(db_create_lite)
main.add_command(db_repair_embeddings)
main.add_command(db_build_index)
main.add_command(db_post_import)

from .repair import db_repair_resume, db_fix_resume

main.add_command(db_repair_resume)
main.add_command(db_fix_resume)

from .serve import serve_cmd

main.add_command(serve_cmd)
