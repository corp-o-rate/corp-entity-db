"""
Shared test fixtures for corp-entity-db.

Provides fresh temp databases, database instances, fake embeddings,
and mock embedders. Resets module-level singletons between tests.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from corp_entity_db.schema_v2 import create_all_tables
from corp_entity_db.seed_data import seed_all_enums
from corp_entity_db.store import (
    LocationsDatabase,
    OrganizationDatabase,
    PersonDatabase,
    RolesDatabase,
)


# ---------------------------------------------------------------------------
# Singleton reset (autouse) -- clears module-level caches every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Clear all module-level singletons/caches so tests are fully isolated."""
    import corp_entity_db.store as _store
    import corp_entity_db.embeddings as _embeddings
    import corp_entity_db.resolver as _resolver

    yield

    # store.py singletons
    _store._shared_connections.clear()
    _store._shared_readonly_connections.clear()
    _store._database_instances.clear()
    _store._person_database_instances.clear()

    # embeddings.py singleton
    _embeddings._default_embedder = None

    # resolver.py singleton
    _resolver._default_resolver = None


# ---------------------------------------------------------------------------
# Database path & connection
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a path to a fresh temporary database file."""
    return tmp_path / "test_entities.db"


@pytest.fixture
def db_conn(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection with full v3 schema and seeded enums."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    create_all_tables(conn)
    seed_all_enums(conn)
    return conn


# ---------------------------------------------------------------------------
# Database instances (writable)
# ---------------------------------------------------------------------------

@pytest.fixture
def org_db(db_path: Path, db_conn: sqlite3.Connection) -> OrganizationDatabase:
    """Writable OrganizationDatabase backed by the temp DB."""
    return OrganizationDatabase(db_path, readonly=False)


@pytest.fixture
def person_db(db_path: Path, db_conn: sqlite3.Connection) -> PersonDatabase:
    """Writable PersonDatabase backed by the temp DB."""
    return PersonDatabase(db_path, readonly=False)


@pytest.fixture
def roles_db(db_path: Path, db_conn: sqlite3.Connection) -> RolesDatabase:
    """Writable RolesDatabase backed by the temp DB."""
    return RolesDatabase(db_path, readonly=False)


@pytest.fixture
def locations_db(db_path: Path, db_conn: sqlite3.Connection) -> LocationsDatabase:
    """Writable LocationsDatabase backed by the temp DB."""
    return LocationsDatabase(db_path, readonly=False)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_embedding():
    """Factory that returns random normalized float32 768-dim vectors."""

    def _make(dim: int = 768) -> np.ndarray:
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    return _make


@pytest.fixture
def mock_embedder():
    """MagicMock standing in for CompanyEmbedder.

    Pre-configured so that:
      - embedding_dim returns 768
      - embed() returns a random normalised 768-d vector
      - embed_batch() returns an array of random normalised vectors
    """
    embedder = MagicMock()
    embedder.embedding_dim = 768

    def _embed(text: str) -> np.ndarray:
        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def _embed_batch(texts: list[str], batch_size: int = 192) -> np.ndarray:
        vecs = np.random.randn(len(texts), 768).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    embedder.embed.side_effect = _embed
    embedder.embed_batch.side_effect = _embed_batch
    return embedder
