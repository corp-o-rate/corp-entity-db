"""Tests for FastAPI server endpoints in corp_entity_db.server."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from corp_entity_db.models import CompanyRecord, PersonRecord, PersonType
from corp_entity_db.server import app


client = TestClient(app)


def _make_embedding(dim: int = 768) -> np.ndarray:
    """Return a normalized random embedding vector."""
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_company_record(**kwargs) -> CompanyRecord:
    defaults = dict(name="Microsoft Corp", source="gleif", source_id="LEI001", region="US")
    defaults.update(kwargs)
    return CompanyRecord(**defaults)


def _make_person_record(**kwargs) -> PersonRecord:
    defaults = dict(
        name="Tim Cook",
        source="wikidata",
        source_id="Q265398",
        country="US",
        person_type=PersonType.EXECUTIVE,
        known_for_role="CEO",
        known_for_org_name="Apple Inc.",
    )
    defaults.update(kwargs)
    return PersonRecord(**defaults)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @patch("corp_entity_db.hub.get_database_path", return_value=None)
    def test_health_endpoint(self, _mock_path):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Search organizations
# ---------------------------------------------------------------------------


class TestSearchOrganizations:
    @patch("corp_entity_db.server._get_org_db")
    @patch("corp_entity_db.server._get_embedder")
    def test_search_organizations(self, mock_get_embedder, mock_get_org_db):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = _make_embedding()
        mock_get_embedder.return_value = mock_embedder

        record = _make_company_record()
        mock_org_db = MagicMock()
        mock_org_db.search.return_value = [(record, 0.95)]
        mock_get_org_db.return_value = mock_org_db

        resp = client.post("/search", json={"query": "Microsoft", "limit": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["record"]["name"] == "Microsoft Corp"
        assert results[0]["similarity_score"] == pytest.approx(0.95)

        mock_embedder.embed.assert_called_once_with("Microsoft")
        mock_org_db.search.assert_called_once()


# ---------------------------------------------------------------------------
# Search people
# ---------------------------------------------------------------------------


class TestSearchPeople:
    @patch("corp_entity_db.server._get_person_db")
    @patch("corp_entity_db.server._get_embedder")
    def test_search_people(self, mock_get_embedder, mock_get_person_db):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = _make_embedding()
        mock_get_embedder.return_value = mock_embedder

        record = _make_person_record()
        mock_person_db = MagicMock()
        mock_person_db.search.return_value = [(record, 0.92)]
        mock_get_person_db.return_value = mock_person_db

        resp = client.post("/search-people", json={"query": "Tim Cook", "limit": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["record"]["name"] == "Tim Cook"
        assert results[0]["similarity_score"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Search roles
# ---------------------------------------------------------------------------


class TestSearchRoles:
    @patch("corp_entity_db.server._get_roles_db")
    def test_search_roles(self, mock_get_roles_db):
        mock_roles_db = MagicMock()
        mock_roles_db.search.return_value = [(1, "Chief Executive Officer", 0.99)]
        mock_get_roles_db.return_value = mock_roles_db

        resp = client.post("/search-roles", json={"query": "CEO", "limit": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["name"] == "Chief Executive Officer"
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Search locations
# ---------------------------------------------------------------------------


class TestSearchLocations:
    @patch("corp_entity_db.server._get_locations_db")
    def test_search_locations(self, mock_get_locations_db):
        mock_locations_db = MagicMock()
        mock_locations_db.search.return_value = [(42, "California", 0.88)]
        mock_get_locations_db.return_value = mock_locations_db

        resp = client.post("/search-locations", json={"query": "California", "limit": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["name"] == "California"
        assert results[0]["id"] == 42
        assert results[0]["score"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


class TestResolve:
    @patch("corp_entity_db.resolver.OrganizationResolver")
    def test_resolve_org(self, MockResolver):
        from corp_entity_db.models import ResolvedOrganization

        resolved = ResolvedOrganization(
            canonical_name="Microsoft Corporation",
            canonical_id="gleif:LEI001",
            source="gleif",
            source_id="LEI001",
            region="US",
            match_confidence=0.98,
        )
        mock_instance = MagicMock()
        mock_instance.resolve.return_value = resolved
        MockResolver.return_value = mock_instance

        resp = client.post("/resolve", json={"name": "MSFT", "type": "org"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["canonical_name"] == "Microsoft Corporation"
        assert body["match_confidence"] == pytest.approx(0.98)

    @patch("corp_entity_db.server._get_person_db")
    @patch("corp_entity_db.server._get_embedder")
    def test_resolve_person(self, mock_get_embedder, mock_get_person_db):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = _make_embedding()
        mock_get_embedder.return_value = mock_embedder

        record = _make_person_record()
        mock_person_db = MagicMock()
        mock_person_db.search.return_value = [(record, 0.91)]
        mock_get_person_db.return_value = mock_person_db

        resp = client.post("/resolve", json={"name": "Tim Cook", "type": "person"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["record"]["name"] == "Tim Cook"
        assert body["similarity_score"] == pytest.approx(0.91)
