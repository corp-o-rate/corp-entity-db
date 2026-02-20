"""Tests for EntityDBClient in corp_entity_db.client."""

from unittest.mock import MagicMock, patch

from corp_entity_db.client import EntityDBClient


def _make_client() -> tuple[EntityDBClient, MagicMock]:
    """Create an EntityDBClient with a mocked httpx module."""
    mock_httpx = MagicMock()
    with patch("corp_entity_db.client._get_httpx", return_value=mock_httpx):
        client = EntityDBClient("http://localhost:9999")
    return client, mock_httpx


class TestEntityDBClient:
    def test_health(self):
        client, mock_httpx = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_httpx.get.return_value = mock_resp

        result = client.health()

        assert result == {"status": "ok"}
        mock_httpx.get.assert_called_once_with("http://localhost:9999/", timeout=120)
        mock_resp.raise_for_status.assert_called_once()

    def test_search_organizations(self):
        client, mock_httpx = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"record": {"name": "Microsoft"}, "similarity_score": 0.95}]
        mock_httpx.post.return_value = mock_resp

        result = client.search_organizations("Microsoft", limit=5, hybrid=True)

        assert len(result) == 1
        assert result[0]["record"]["name"] == "Microsoft"
        mock_httpx.post.assert_called_once_with(
            "http://localhost:9999/search",
            json={"query": "Microsoft", "limit": 5, "hybrid": True},
            timeout=120,
        )

    def test_search_people(self):
        client, mock_httpx = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"record": {"name": "Tim Cook"}, "similarity_score": 0.92}]
        mock_httpx.post.return_value = mock_resp

        result = client.search_people("Tim Cook", limit=3)

        assert len(result) == 1
        assert result[0]["record"]["name"] == "Tim Cook"
        mock_httpx.post.assert_called_once_with(
            "http://localhost:9999/search-people",
            json={"query": "Tim Cook", "limit": 3},
            timeout=120,
        )

    def test_resolve(self):
        client, mock_httpx = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"canonical_name": "Microsoft Corporation"}
        mock_httpx.post.return_value = mock_resp

        result = client.resolve("MSFT", type="org")

        assert result["canonical_name"] == "Microsoft Corporation"
        mock_httpx.post.assert_called_once_with(
            "http://localhost:9999/resolve",
            json={"name": "MSFT", "type": "org"},
            timeout=120,
        )
