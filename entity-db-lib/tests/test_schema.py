"""Tests for schema_v2 DDL creation."""

import sqlite3

from corp_entity_db.schema_v2 import create_all_tables


def test_create_all_tables_creates_expected_tables(db_path):
    """After create_all_tables, sqlite_master should contain all expected tables."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    create_all_tables(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row["name"] for row in cursor}

    expected = {
        "source_types",
        "people_types",
        "organization_types",
        "simplified_location_types",
        "location_types",
        "roles",
        "locations",
        "organizations",
        "people",
        "db_info",
    }
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


def test_schema_version_is_3(db_path):
    """db_info should record schema_version = '3' after table creation."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    create_all_tables(conn)

    cursor = conn.execute(
        "SELECT value FROM db_info WHERE key = 'schema_version'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row["value"] == "3"


def test_views_created(db_path):
    """All four human-readable views should exist after create_all_tables."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    create_all_tables(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
    )
    views = {row["name"] for row in cursor}

    expected_views = {
        "organizations_view",
        "people_view",
        "roles_view",
        "locations_view",
    }
    assert expected_views.issubset(views), f"Missing views: {expected_views - views}"


def test_idempotent_creation(db_path):
    """Calling create_all_tables twice on the same connection should not raise."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    create_all_tables(conn)
    create_all_tables(conn)  # second call must not error

    cursor = conn.execute(
        "SELECT value FROM db_info WHERE key = 'schema_version'"
    )
    assert cursor.fetchone()["value"] == "3"
