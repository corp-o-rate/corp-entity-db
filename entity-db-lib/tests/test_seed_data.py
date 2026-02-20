"""Tests for seed_data enum tables and pycountry integration."""

from corp_entity_db.seed_data import (
    ORGANIZATION_TYPES,
    ORG_TYPE_ID_TO_NAME,
    ORG_TYPE_NAME_TO_ID,
    PEOPLE_TYPE_ID_TO_NAME,
    PEOPLE_TYPE_NAME_TO_ID,
    PEOPLE_TYPES,
    SOURCE_ID_TO_NAME,
    SOURCE_NAME_TO_ID,
    SOURCE_TYPES,
    seed_all_enums,
    seed_pycountry_locations,
)


def test_source_name_to_id_consistency():
    """Forward and reverse source maps should be consistent."""
    for id_, name in SOURCE_ID_TO_NAME.items():
        assert SOURCE_NAME_TO_ID[name] == id_
    # Every canonical entry in SOURCE_TYPES should appear in the reverse map
    for id_, name in SOURCE_TYPES:
        assert SOURCE_ID_TO_NAME[id_] == name


def test_org_type_maps_consistency():
    """ORG_TYPE_NAME_TO_ID and ORG_TYPE_ID_TO_NAME should be consistent."""
    for name, id_ in ORG_TYPE_NAME_TO_ID.items():
        assert ORG_TYPE_ID_TO_NAME[id_] == name
    for id_, name in ORG_TYPE_ID_TO_NAME.items():
        assert ORG_TYPE_NAME_TO_ID[name] == id_


def test_people_type_maps_consistency():
    """PEOPLE_TYPE_NAME_TO_ID and PEOPLE_TYPE_ID_TO_NAME should be consistent."""
    for name, id_ in PEOPLE_TYPE_NAME_TO_ID.items():
        assert PEOPLE_TYPE_ID_TO_NAME[id_] == name
    for id_, name in PEOPLE_TYPE_ID_TO_NAME.items():
        assert PEOPLE_TYPE_NAME_TO_ID[name] == id_


def test_seed_all_enums_populates_tables(db_conn):
    """seed_all_enums (already called by db_conn fixture) should populate enum tables."""
    cursor = db_conn.execute("SELECT COUNT(*) FROM source_types")
    assert cursor.fetchone()[0] == len(SOURCE_TYPES)

    cursor = db_conn.execute("SELECT COUNT(*) FROM people_types")
    assert cursor.fetchone()[0] == len(PEOPLE_TYPES)

    cursor = db_conn.execute("SELECT COUNT(*) FROM organization_types")
    assert cursor.fetchone()[0] == len(ORGANIZATION_TYPES)


def test_seed_pycountry_locations(db_conn):
    """seed_pycountry_locations should insert countries; spot-check United States."""
    seed_pycountry_locations(db_conn)

    cursor = db_conn.execute(
        "SELECT name FROM locations WHERE source_identifier = 'US'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row["name"] == "United States"


def test_source_types_count():
    """SOURCE_TYPES should have exactly 4 entries."""
    assert len(SOURCE_TYPES) == 4


def test_organization_types_count():
    """ORGANIZATION_TYPES should have exactly 17 entries."""
    assert len(ORGANIZATION_TYPES) == 17
