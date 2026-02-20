"""Tests for importers/import_utils.py pure functions."""

from corp_entity_db.importers.import_utils import (
    format_qid,
    get_entity_type_id,
    get_location_type_id,
    get_person_type_id,
    get_source_id,
    normalize_name,
    parse_qid,
)


# ---------------------------------------------------------------------------
# parse_qid
# ---------------------------------------------------------------------------


class TestParseQid:
    def test_with_q_prefix(self):
        assert parse_qid("Q12345") == 12345

    def test_without_prefix(self):
        assert parse_qid("12345") == 12345

    def test_lowercase_q(self):
        assert parse_qid("q999") == 999

    def test_none(self):
        assert parse_qid(None) is None

    def test_empty(self):
        assert parse_qid("") is None

    def test_invalid(self):
        assert parse_qid("abc") is None

    def test_whitespace(self):
        assert parse_qid("  Q42  ") == 42


# ---------------------------------------------------------------------------
# format_qid
# ---------------------------------------------------------------------------


class TestFormatQid:
    def test_int_to_string(self):
        assert format_qid(12345) == "Q12345"

    def test_none_returns_none(self):
        assert format_qid(None) is None


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_whitespace(self):
        assert normalize_name("  Apple Inc  ") == "apple inc"

    def test_empty(self):
        assert normalize_name("") == ""

    def test_mixed_case(self):
        assert normalize_name("GOOGLE") == "google"


# ---------------------------------------------------------------------------
# get_source_id
# ---------------------------------------------------------------------------


class TestGetSourceId:
    def test_gleif(self):
        assert get_source_id("gleif") == 1

    def test_sec_edgar(self):
        assert get_source_id("sec_edgar") == 2

    def test_companies_house(self):
        assert get_source_id("companies_house") == 3

    def test_wikidata(self):
        assert get_source_id("wikidata") == 4

    def test_unknown_defaults_to_wikidata(self):
        assert get_source_id("nonexistent") == 4

    def test_legacy_wikipedia(self):
        assert get_source_id("wikipedia") == 4


# ---------------------------------------------------------------------------
# get_entity_type_id
# ---------------------------------------------------------------------------


class TestGetEntityTypeId:
    def test_business(self):
        assert get_entity_type_id("business") == 1

    def test_fund(self):
        assert get_entity_type_id("fund") == 2

    def test_unknown_defaults(self):
        assert get_entity_type_id("nonexistent") == 17


# ---------------------------------------------------------------------------
# get_person_type_id
# ---------------------------------------------------------------------------


class TestGetPersonTypeId:
    def test_executive(self):
        assert get_person_type_id("executive") == 1

    def test_politician(self):
        assert get_person_type_id("politician") == 2

    def test_unknown_defaults(self):
        assert get_person_type_id("nonexistent") == 15


# ---------------------------------------------------------------------------
# get_location_type_id
# ---------------------------------------------------------------------------


class TestGetLocationTypeId:
    def test_country(self):
        assert get_location_type_id("country") == 2

    def test_city(self):
        assert get_location_type_id("city") == 19

    def test_unknown_defaults(self):
        assert get_location_type_id("nonexistent") == 36
