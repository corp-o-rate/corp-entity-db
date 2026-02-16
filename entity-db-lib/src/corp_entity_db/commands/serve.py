"""Serve command - start the entity database server."""

import logging

import click

logger = logging.getLogger(__name__)


@click.command("serve")
@click.option("--port", default=8222, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--no-warmup", is_flag=True, help="Skip eager loading of models and databases.")
@click.option("--db-path", default=None, help="Path to database file.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def serve_cmd(port: int, host: str, no_warmup: bool, db_path: str, verbose: bool):
    """Start the entity database server."""
    from corp_entity_db.server import run_server

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    run_server(
        host=host,
        port=port,
        do_warmup=not no_warmup,
        db_path=db_path,
        verbose=verbose,
    )
