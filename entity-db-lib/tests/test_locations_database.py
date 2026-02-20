"""Tests for LocationsDatabase CRUD, search, and region resolution."""

from corp_entity_db.store import LocationsDatabase


def test_get_or_create_new_location(locations_db: LocationsDatabase):
    """Creating a new location with type_id=2 (country) should return ID > 0."""
    loc_id = locations_db.get_or_create(
        "Testland", location_type_id=2, source_identifier="TL"
    )
    assert loc_id > 0


def test_get_or_create_idempotent(locations_db: LocationsDatabase):
    """Same name + source_identifier should return the same ID."""
    id1 = locations_db.get_or_create(
        "Testland", location_type_id=2, source_identifier="TL"
    )
    id2 = locations_db.get_or_create(
        "Testland", location_type_id=2, source_identifier="TL"
    )
    assert id1 == id2


def test_resolve_region_text_by_name(locations_db: LocationsDatabase):
    """resolve_region_text should find a location inserted by name."""
    # Insert a location first (locations are imported from wikidata, not pycountry)
    locations_db.get_or_create(
        "United States of America", location_type_id=2, source_identifier="Q30", qid=30
    )
    loc_id = locations_db.resolve_region_text("United States of America")
    assert loc_id is not None
    assert loc_id > 0


def test_resolve_region_text_returns_none_for_missing(locations_db: LocationsDatabase):
    """resolve_region_text should return None for a location not in the database."""
    loc_id = locations_db.resolve_region_text("Nonexistent Country")
    assert loc_id is None


def test_search_locations(locations_db: LocationsDatabase):
    """Inserting 'California' and searching for it should find it."""
    locations_db.get_or_create(
        "California", location_type_id=5, source_identifier="CA-US"
    )
    results = locations_db.search("California")
    assert len(results) >= 1
    names = [name for _, name, _ in results]
    assert "California" in names


def test_get_stats(locations_db: LocationsDatabase):
    """total_locations should reflect inserted records."""
    stats_before = locations_db.get_stats()
    locations_db.get_or_create(
        "Atlantis", location_type_id=33, source_identifier="ATL"
    )
    stats_after = locations_db.get_stats()
    assert stats_after["total_locations"] == stats_before["total_locations"] + 1
