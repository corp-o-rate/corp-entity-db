"""Tests for store.py helper functions (pure functions, no DB needed)."""

from corp_names import normalize_company, normalize_name

from corp_entity_db.store import (
    UnionFind,
    _extract_search_terms,
    _names_match_for_canon,
    _normalize_region,
    _regions_match,
)


# ---------------------------------------------------------------------------
# normalize_company (corp_names — replaces _normalize_name)
# ---------------------------------------------------------------------------


class TestNormalizeCompany:
    def test_basic(self):
        assert normalize_company("Apple Inc.").normalized == "apple"

    def test_possessive(self):
        result = normalize_company("McDonald's Corp").normalized
        assert "mcdonald" in result

    def test_empty(self):
        assert normalize_company("").normalized == ""

    def test_suffix_only(self):
        # Name consisting only of a suffix normalizes to empty (suffix stripped, nothing left)
        result = normalize_company("Ltd").normalized
        assert result == ""


# ---------------------------------------------------------------------------
# _extract_search_terms
# ---------------------------------------------------------------------------


class TestExtractSearchTerms:
    def test_basic_tokenization(self):
        terms = _extract_search_terms("Apple Inc")
        assert "Apple" in terms

    def test_short_word_filtering(self):
        # Words shorter than 3 chars are dropped when there are multiple words
        terms = _extract_search_terms("A Big Company")
        assert "A" not in terms

    def test_single_short_word_kept(self):
        terms = _extract_search_terms("AI")
        assert terms == ["AI"]

    def test_limited_to_three(self):
        terms = _extract_search_terms("One Two Three Four Five")
        assert len(terms) <= 3

    def test_sorted_longest_first(self):
        terms = _extract_search_terms("Big Company International")
        assert terms == sorted(terms, key=len, reverse=True)


# ---------------------------------------------------------------------------
# _names_match_for_canon (now uses corp_names.normalize_company)
# ---------------------------------------------------------------------------


class TestNamesMatchForCanon:
    def test_exact_match(self):
        assert _names_match_for_canon("Apple Inc", "apple inc") is True

    def test_suffix_expanded_match(self):
        assert _names_match_for_canon("Acme Ltd", "Acme Limited") is True

    def test_mismatch(self):
        assert _names_match_for_canon("Apple", "Google") is False

    def test_trailing_dot_match(self):
        assert _names_match_for_canon("Acme Inc.", "Acme Inc") is True


# ---------------------------------------------------------------------------
# _normalize_region / _regions_match
# ---------------------------------------------------------------------------


class TestNormalizeRegion:
    def test_country_code(self):
        assert _normalize_region("US") == "US"

    def test_alias_uk(self):
        assert _normalize_region("UK") == "GB"

    def test_empty(self):
        assert _normalize_region("") == ""

    def test_alpha3(self):
        assert _normalize_region("GBR") == "GB"

    def test_alias_usa(self):
        assert _normalize_region("USA") == "US"


class TestRegionsMatch:
    def test_same_after_norm(self):
        assert _regions_match("US", "USA") is True

    def test_empty_matches_anything(self):
        assert _regions_match("", "US") is True
        assert _regions_match("GB", "") is True

    def test_different(self):
        assert _regions_match("US", "GB") is False


# ---------------------------------------------------------------------------
# normalize_name (corp_names — replaces _normalize_person_name)
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_prefix_removal(self):
        assert normalize_name("Dr. John Smith").normalized == "john smith"

    def test_suffix_removal(self):
        result = normalize_name("John Smith Jr.").normalized
        assert "john smith" in result

    def test_both(self):
        result = normalize_name("Prof. Jane Doe PhD").normalized
        assert "jane doe" in result

    def test_plain_name(self):
        assert normalize_name("Jane Doe").normalized == "jane doe"

    def test_empty(self):
        assert normalize_name("").normalized == ""

    def test_only_title_normalizes_to_empty(self):
        # A name that is purely a title normalizes to empty (title stripped, nothing left)
        result = normalize_name("Dr.").normalized
        assert result == ""
