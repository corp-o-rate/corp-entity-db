"""Shared utilities used across CLI command modules."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click


def _configure_logging(verbose: bool) -> None:
    """Configure logging for the entity database."""
    level = logging.DEBUG if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Set level for corp_entity_db loggers
    for logger_name in [
        "corp_entity_db",
        "corp_entity_db.store",
        "corp_entity_db.embeddings",
        "corp_entity_db.hub",
        "corp_entity_db.resolver",
        "corp_entity_db.canonicalization",
        "corp_entity_db.importers",
    ]:
        logging.getLogger(logger_name).setLevel(level)

    # Suppress noisy third-party loggers
    for noisy_logger in [
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "asyncio",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _resolve_db_path(db_path: Optional[str] = None) -> Path:
    """Resolve the database path from an explicit --db value or --db-version context.

    Checks v3 first, then falls back to v2 if the v3 file doesn't exist.
    """
    if db_path is not None:
        return Path(db_path)
    # Check for --db-version in the Click context chain
    try:
        ctx = click.get_current_context(silent=True)
        db_version = ctx.obj.get("db_version") if ctx and ctx.obj else None
    except RuntimeError:
        db_version = None
    if db_version is not None:
        from corp_entity_db.hub import DEFAULT_CACHE_DIR, db_filenames
        full_fn, _, _ = db_filenames(db_version)
        return DEFAULT_CACHE_DIR / full_fn
    # Default: try v3, fall back to v2
    from corp_entity_db.hub import DEFAULT_CACHE_DIR
    from corp_entity_db.store import DEFAULT_DB_PATH
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    v2_path = DEFAULT_CACHE_DIR / "entities-v2.db"
    if v2_path.exists():
        return v2_path
    return DEFAULT_DB_PATH  # Return v3 path even if missing (for creation)
