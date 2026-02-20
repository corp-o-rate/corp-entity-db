"""Tests for Pydantic models: CompanyRecord, PersonRecord, RoleRecord, LocationRecord, matches, enums."""

import json

from corp_entity_db.models import (
    CompanyMatch,
    CompanyRecord,
    EntityType,
    LocationRecord,
    PersonMatch,
    PersonRecord,
    PersonType,
    RoleRecord,
    SimplifiedLocationType,
    SourceTypeEnum,
)


# ---------------------------------------------------------------------------
# CompanyRecord
# ---------------------------------------------------------------------------


class TestCompanyRecord:
    def test_canonical_id_format(self):
        rec = CompanyRecord(name="Acme", source="gleif", source_id="LEI123")
        assert rec.canonical_id == "gleif:LEI123"

    def test_canonical_id_sec(self):
        rec = CompanyRecord(name="X Corp", source="sec_edgar", source_id="000999")
        assert rec.canonical_id == "sec_edgar:000999"

    def test_model_dump_for_db_keys(self):
        rec = CompanyRecord(
            name="Foo Ltd",
            source="gleif",
            source_id="ABC",
            entity_type=EntityType.FUND,
        )
        d = rec.model_dump_for_db()
        assert set(d.keys()) == {
            "name", "source", "source_id", "region",
            "entity_type", "from_date", "to_date", "record",
        }

    def test_model_dump_entity_type_serialized_as_value(self):
        rec = CompanyRecord(name="X", source="gleif", source_id="1", entity_type=EntityType.FUND)
        assert rec.model_dump_for_db()["entity_type"] == "fund"

    def test_model_dump_defaults(self):
        rec = CompanyRecord(name="X", source="gleif", source_id="1")
        d = rec.model_dump_for_db()
        assert d["from_date"] == ""
        assert d["to_date"] == ""
        assert d["region"] == ""
        assert d["record"] == {}


# ---------------------------------------------------------------------------
# PersonRecord
# ---------------------------------------------------------------------------


class TestPersonRecord:
    def test_is_historic_with_death_date(self):
        rec = PersonRecord(name="A", source="wikidata", source_id="Q1", death_date="1900-01-01")
        assert rec.is_historic is True

    def test_is_historic_without_death_date(self):
        rec = PersonRecord(name="A", source="wikidata", source_id="Q1")
        assert rec.is_historic is False

    def test_is_historic_empty_string(self):
        rec = PersonRecord(name="A", source="wikidata", source_id="Q1", death_date="")
        assert rec.is_historic is False

    def test_get_embedding_text_full_context(self):
        rec = PersonRecord(
            name="Tim Cook",
            source="wikidata",
            source_id="Q1",
            known_for_role="CEO",
            known_for_org_name="Apple Inc.",
        )
        assert rec.get_embedding_text() == "Tim Cook | CEO | Apple Inc."

    def test_get_embedding_text_name_only(self):
        rec = PersonRecord(name="Tim Cook", source="wikidata", source_id="Q1")
        assert rec.get_embedding_text() == "Tim Cook"

    def test_get_embedding_text_role_only(self):
        rec = PersonRecord(name="Tim Cook", source="wikidata", source_id="Q1", known_for_role="CEO")
        assert rec.get_embedding_text() == "Tim Cook | CEO"

    def test_model_dump_for_db_person_type_value(self):
        rec = PersonRecord(
            name="A", source="wikidata", source_id="Q1",
            person_type=PersonType.POLITICIAN,
        )
        assert rec.model_dump_for_db()["person_type"] == "politician"

    def test_model_dump_for_db_defaults(self):
        rec = PersonRecord(name="A", source="wikidata", source_id="Q1")
        d = rec.model_dump_for_db()
        assert d["from_date"] == ""
        assert d["birth_date"] == ""
        assert d["death_date"] == ""
        assert d["known_for_org_id"] is None


# ---------------------------------------------------------------------------
# RoleRecord
# ---------------------------------------------------------------------------


class TestRoleRecord:
    def test_canonical_id_with_source_id(self):
        rec = RoleRecord(name="CEO", source_id="Q484876")
        assert rec.canonical_id == "wikidata:Q484876"

    def test_canonical_id_without_source_id(self):
        rec = RoleRecord(name="CEO")
        assert rec.canonical_id == "wikidata:CEO"


# ---------------------------------------------------------------------------
# LocationRecord
# ---------------------------------------------------------------------------


class TestLocationRecord:
    def test_model_dump_parent_ids_json(self):
        rec = LocationRecord(name="California", parent_ids=[10, 20])
        d = rec.model_dump_for_db()
        assert d["parent_ids"] == json.dumps([10, 20])

    def test_model_dump_empty_parent_ids(self):
        rec = LocationRecord(name="Earth")
        d = rec.model_dump_for_db()
        assert d["parent_ids"] == "[]"


# ---------------------------------------------------------------------------
# Match from_record helpers
# ---------------------------------------------------------------------------


class TestMatchFromRecord:
    def test_company_match_from_record(self):
        rec = CompanyRecord(name="Apple Inc.", source="sec_edgar", source_id="320193")
        m = CompanyMatch.from_record("Apple", rec, 0.95)
        assert m.query_name == "Apple"
        assert m.source == "sec_edgar"
        assert m.source_id == "320193"
        assert m.canonical_id == "sec_edgar:320193"
        assert m.similarity_score == 0.95
        assert m.llm_confirmed is False

    def test_person_match_from_record(self):
        rec = PersonRecord(name="Tim Cook", source="wikidata", source_id="Q1")
        m = PersonMatch.from_record("Cook", rec, 0.88, llm_confirmed=True)
        assert m.query_name == "Cook"
        assert m.canonical_id == "wikidata:Q1"
        assert m.llm_confirmed is True


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestEnumCompleteness:
    def test_entity_type_values(self):
        values = {e.value for e in EntityType}
        assert "business" in values
        assert "unknown" in values
        assert len(values) >= 17

    def test_person_type_values(self):
        values = {e.value for e in PersonType}
        assert "executive" in values
        assert "unknown" in values
        assert len(values) >= 15

    def test_source_type_enum_values(self):
        values = {e.value for e in SourceTypeEnum}
        assert values == {"gleif", "sec_edgar", "companies_house", "wikidata"}

    def test_simplified_location_type_values(self):
        values = {e.value for e in SimplifiedLocationType}
        assert "country" in values
        assert "city" in values
        assert len(values) == 7
