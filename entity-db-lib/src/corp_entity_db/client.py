"""HTTP client for delegating to a running corp-entity-db server.

Requires the 'client' extra: pip install corp-entity-db[client]
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # 2 minutes — embedding + search should be fast


def _get_httpx():
    """Lazy import httpx, raising a clear error if not installed."""
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for EntityDBClient. "
            "Install it with: pip install corp-entity-db[client]"
        )


class EntityDBClient:
    """Client for the entity database server."""

    def __init__(self, server_url: str = "http://localhost:8222"):
        self.server_url = server_url.rstrip("/")
        self._httpx = _get_httpx()

    def health(self) -> dict:
        """Check server health."""
        resp = self._httpx.get(f"{self.server_url}/", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def search_organizations(
        self,
        query: str,
        limit: int = 10,
        hybrid: bool = False,
    ) -> list[dict]:
        """Search organizations by name."""
        resp = self._httpx.post(
            f"{self.server_url}/search",
            json={"query": query, "limit": limit, "hybrid": hybrid},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_people(self, query: str, limit: int = 10) -> list[dict]:
        """Search people by name."""
        resp = self._httpx.post(
            f"{self.server_url}/search-people",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_roles(self, query: str, limit: int = 10) -> list[dict]:
        """Search roles/job titles."""
        resp = self._httpx.post(
            f"{self.server_url}/search-roles",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_locations(self, query: str, limit: int = 10) -> list[dict]:
        """Search locations."""
        resp = self._httpx.post(
            f"{self.server_url}/search-locations",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def resolve(self, name: str, type: str = "org") -> Optional[dict]:
        """Resolve an entity name to a canonical record."""
        resp = self._httpx.post(
            f"{self.server_url}/resolve",
            json={"name": name, "type": type},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
