"""Tests for PersonDatabase insert, lookup, and stats."""

import numpy as np

from corp_entity_db.models import PersonRecord, PersonType
from corp_entity_db.store import PersonDatabase


def test_insert_single_person(person_db: PersonDatabase, fake_embedding):
    """Inserting a single PersonRecord should return a row_id > 0."""
    record = PersonRecord(
        name="Tim Cook",
        source="wikidata",
        source_id="Q265398",
        person_type=PersonType.EXECUTIVE,
    )
    row_id = person_db.insert(record, fake_embedding())
    assert row_id > 0


def test_insert_batch(person_db: PersonDatabase, fake_embedding):
    """insert_batch with 2 records should return count = 2."""
    records = [
        PersonRecord(
            name="Alice Smith",
            source="wikidata",
            source_id="Q100001",
            person_type=PersonType.POLITICIAN,
        ),
        PersonRecord(
            name="Bob Jones",
            source="wikidata",
            source_id="Q100002",
            person_type=PersonType.ATHLETE,
        ),
    ]
    embeddings = np.stack([fake_embedding() for _ in records])
    count = person_db.insert_batch(records, embeddings)
    assert count == 2


def test_get_by_source_id(person_db: PersonDatabase, fake_embedding):
    """After insert, get_by_source_id should find the person."""
    record = PersonRecord(
        name="Satya Nadella",
        source="wikidata",
        source_id="Q528233",
        person_type=PersonType.EXECUTIVE,
    )
    person_db.insert(record, fake_embedding())

    found = person_db.get_by_source_id("wikidata", "Q528233")
    assert found is not None
    assert found.name == "Satya Nadella"


def test_get_stats(person_db: PersonDatabase, fake_embedding):
    """get_stats should reflect inserted record count."""
    records = [
        PersonRecord(
            name=f"Person {i}",
            source="wikidata",
            source_id=f"Q{90000 + i}",
        )
        for i in range(3)
    ]
    embeddings = np.stack([fake_embedding() for _ in records])
    person_db.insert_batch(records, embeddings)

    stats = person_db.get_stats()
    assert stats["total_records"] == 3
