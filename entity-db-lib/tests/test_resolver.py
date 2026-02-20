"""Tests for corp_entity_db.resolver — OrganizationResolver."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from corp_entity_db.models import CompanyRecord
from corp_entity_db.resolver import OrganizationResolver


def _make_mock_database(results: list[tuple]):
    """Create a mock database that returns the given (record, score) pairs."""
    db = MagicMock()
    db.search.return_value = results
    return db


def _make_mock_embedder():
    """Create a mock embedder that returns a random normalised vector."""
    embedder = MagicMock()
    vec = np.random.randn(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    embedder.embed.return_value = vec
    return embedder


def _sample_record(**overrides) -> CompanyRecord:
    defaults = dict(
        name="Test Corp Inc",
        source="sec_edgar",
        source_id="0001234567",
        region="US",
    )
    defaults.update(overrides)
    return CompanyRecord(**defaults)


class TestResolveEmpty:
    def test_empty_string(self):
        """resolve('') returns None immediately."""
        resolver = OrganizationResolver()
        assert resolver.resolve("") is None

    def test_whitespace_only(self):
        """resolve('   ') returns None (empty after strip is still falsy in the check)."""
        resolver = OrganizationResolver()
        # The resolver checks `if not org_name` — whitespace is truthy,
        # so it will proceed. But the cache key will be empty after strip,
        # and we need to mock the database to avoid real lookups.
        # Actually, "   " is truthy, so it won't short-circuit.
        # We mock _get_database to return None to avoid real DB access.
        resolver._get_database = MagicMock(return_value=None)
        result = resolver.resolve("   ")
        assert result is None


class TestCaching:
    def test_resolve_uses_cache(self):
        """Second resolve() call should hit cache, not call database.search again."""
        mock_db = _make_mock_database([(_sample_record(), 0.9)])
        mock_emb = _make_mock_embedder()

        resolver = OrganizationResolver()
        resolver._get_database = MagicMock(return_value=mock_db)
        resolver._get_embedder = MagicMock(return_value=mock_emb)

        result1 = resolver.resolve("Test Corp")
        result2 = resolver.resolve("Test Corp")

        assert result1 is not None
        assert result1 is result2
        mock_db.search.assert_called_once()


class TestBelowMinSimilarity:
    def test_returns_none(self):
        """Results below min_similarity threshold yield None."""
        mock_db = _make_mock_database([(_sample_record(), 0.3)])
        mock_emb = _make_mock_embedder()

        resolver = OrganizationResolver(min_similarity=0.7)
        resolver._get_database = MagicMock(return_value=mock_db)
        resolver._get_embedder = MagicMock(return_value=mock_emb)

        result = resolver.resolve("Test Corp")
        assert result is None


class TestResolvedOrganizationFields:
    def test_fields(self):
        """ResolvedOrganization should carry the correct name, source, source_id, and confidence."""
        record = _sample_record(name="Acme Inc", source="sec_edgar", source_id="0009999999")
        mock_db = _make_mock_database([(record, 0.9)])
        mock_emb = _make_mock_embedder()

        resolver = OrganizationResolver(min_similarity=0.7)
        resolver._get_database = MagicMock(return_value=mock_db)
        resolver._get_embedder = MagicMock(return_value=mock_emb)

        resolved = resolver.resolve("Acme")

        assert resolved is not None
        assert resolved.canonical_name == "Acme Inc"
        assert resolved.source == "sec_edgar"
        assert resolved.source_id == "0009999999"
        assert resolved.match_confidence == pytest.approx(0.9)
        assert resolved.canonical_id == "SEC-CIK:0009999999"


class TestResolveWithCandidates:
    def test_empty_name(self):
        """resolve_with_candidates('') returns an empty list."""
        resolver = OrganizationResolver()
        assert resolver.resolve_with_candidates("") == []
