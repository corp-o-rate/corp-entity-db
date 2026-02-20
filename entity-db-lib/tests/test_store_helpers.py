"""Tests for store.py helper functions (pure functions, no DB needed)."""

from corp_entity_db.store import (
    UnionFind,
    _clean_org_name,
    _expand_suffix,
    _extract_search_terms,
    _names_match_for_canon,
    _normalize_for_canon,
    _normalize_name,
    _normalize_person_name,
    _normalize_region,
    _regions_match,
    _remove_suffix,
)


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_basic(self):
        assert _normalize_name("Apple Inc.") == "apple"

    def test_possessive(self):
        assert _normalize_name("McDonald's Corp") == "mcdonald"

    def test_empty(self):
        assert _normalize_name("") == ""

    def test_special_chars(self):
        result = _normalize_name("[Acme] (Holdings)")
        assert "acme" in result
        assert "[" not in result

    def test_suffix_only(self):
        # Name consisting only of a suffix should still return something
        result = _normalize_name("Ltd")
        assert result != ""


# ---------------------------------------------------------------------------
# _clean_org_name
# ---------------------------------------------------------------------------


class TestCleanOrgName:
    def test_brackets(self):
        assert _clean_org_name("[Acme Corp]") == "Acme Corp"

    def test_parens(self):
        assert _clean_org_name("Acme (Holdings)") == "Acme Holdings"

    def test_quotes(self):
        result = _clean_org_name('"Acme Corp"')
        assert result == "Acme Corp"

    def test_none(self):
        assert _clean_org_name(None) == ""

    def test_empty(self):
        assert _clean_org_name("") == ""


# ---------------------------------------------------------------------------
# _remove_suffix
# ---------------------------------------------------------------------------


class TestRemoveSuffix:
    def test_strips_ltd(self):
        assert _remove_suffix("Acme Ltd") == "Acme"

    def test_strips_inc(self):
        assert _remove_suffix("Apple Inc.") == "Apple"

    def test_strips_corp(self):
        assert _remove_suffix("Microsoft Corp") == "Microsoft"

    def test_no_match(self):
        assert _remove_suffix("Google") == "Google"

    def test_strips_possessive(self):
        assert _remove_suffix("McDonald's") == "McDonald"


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
# _normalize_for_canon / _expand_suffix
# ---------------------------------------------------------------------------


class TestNormalizeForCanon:
    def test_lowercase_strip(self):
        # Trailing space prevents rstrip(".") from removing the dot
        assert _normalize_for_canon("  Apple Inc.  ") == "apple inc."

    def test_trailing_dot_stripped_when_final(self):
        assert _normalize_for_canon("Apple Inc.") == "apple inc"

    def test_trailing_dot(self):
        assert _normalize_for_canon("Apple Inc.") == "apple inc"


class TestExpandSuffix:
    def test_ltd_expanded(self):
        assert _expand_suffix("Acme Ltd").endswith("limited")

    def test_corp_expanded(self):
        assert _expand_suffix("Big Corp").endswith("corporation")

    def test_no_expansion(self):
        assert _expand_suffix("Google") == "google"


# ---------------------------------------------------------------------------
# _names_match_for_canon
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
# _normalize_person_name
# ---------------------------------------------------------------------------


class TestNormalizePersonName:
    def test_prefix_removal(self):
        assert _normalize_person_name("Dr. John Smith") == "john smith"

    def test_suffix_removal(self):
        assert _normalize_person_name("John Smith Jr.") == "john smith"

    def test_both(self):
        assert _normalize_person_name("Prof. Jane Doe PhD") == "jane doe"

    def test_plain_name(self):
        assert _normalize_person_name("Jane Doe") == "jane doe"

    def test_empty(self):
        assert _normalize_person_name("") == ""

    def test_only_title_falls_back(self):
        result = _normalize_person_name("Dr.")
        assert result != ""
