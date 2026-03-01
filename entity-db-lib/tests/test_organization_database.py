"""Tests for OrganizationDatabase insert, lookup, and stats."""

from corp_entity_db.models import CompanyRecord, EntityType
from corp_entity_db.store import OrganizationDatabase


def test_insert_single_record(org_db: OrganizationDatabase):
    """Inserting a single CompanyRecord should return a row_id > 0."""
    record = CompanyRecord(
        name="Acme Corp",
        source="sec_edgar",
        source_id="CIK0001234",
    )
    row_id = org_db.insert(record)
    assert row_id > 0


def test_insert_batch(org_db: OrganizationDatabase):
    """insert_batch with 3 records should return count = 3."""
    records = [
        CompanyRecord(name=f"Company {i}", source="gleif", source_id=f"LEI{i:04d}")
        for i in range(3)
    ]
    count = org_db.insert_batch(records)
    assert count == 3


def test_get_by_source_id(org_db: OrganizationDatabase):
    """After insert, get_by_source_id should find the record."""
    record = CompanyRecord(
        name="Widget Inc",
        source="sec_edgar",
        source_id="CIK0005678",
    )
    org_db.insert(record)

    found = org_db.get_by_source_id("sec_edgar", "CIK0005678")
    assert found is not None
    assert found.name == "Widget Inc"


def test_get_stats(org_db: OrganizationDatabase):
    """get_stats should reflect inserted record count."""
    records = [
        CompanyRecord(name=f"Org {i}", source="gleif", source_id=f"LEI{i:04d}")
        for i in range(5)
    ]
    org_db.insert_batch(records)

    stats = org_db.get_stats()
    assert stats.total_records == 5


def test_get_stats_empty(org_db: OrganizationDatabase):
    """Empty database should have total_records = 0."""
    stats = org_db.get_stats()
    assert stats.total_records == 0


def test_insert_with_entity_type(org_db: OrganizationDatabase):
    """Inserting with entity_type=FUND should persist and return the correct type."""
    record = CompanyRecord(
        name="Vanguard S&P 500 ETF",
        source="sec_edgar",
        source_id="CIK0009999",
        entity_type=EntityType.FUND,
    )
    org_db.insert(record)

    found = org_db.get_by_source_id("sec_edgar", "CIK0009999")
    assert found is not None
    assert found.entity_type == EntityType.FUND
