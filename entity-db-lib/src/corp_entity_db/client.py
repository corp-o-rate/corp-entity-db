"""HTTP client for delegating to a running corp-entity-db server."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # 2 minutes — embedding + search should be fast


class EntityDBClient:
    """Client for the entity database server."""

    def __init__(self, server_url: str = "http://localhost:8222"):
        self.server_url = server_url.rstrip("/")

    def health(self) -> dict:
        """Check server health."""
        resp = httpx.get(f"{self.server_url}/", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def search_organizations(
        self,
        query: str,
        limit: int = 10,
        hybrid: bool = False,
    ) -> list[dict]:
        """Search organizations by name."""
        resp = httpx.post(
            f"{self.server_url}/search",
            json={"query": query, "limit": limit, "hybrid": hybrid},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_people(self, query: str, limit: int = 10) -> list[dict]:
        """Search people by name."""
        resp = httpx.post(
            f"{self.server_url}/search-people",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_roles(self, query: str, limit: int = 10) -> list[dict]:
        """Search roles/job titles."""
        resp = httpx.post(
            f"{self.server_url}/search-roles",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_locations(self, query: str, limit: int = 10) -> list[dict]:
        """Search locations."""
        resp = httpx.post(
            f"{self.server_url}/search-locations",
            json={"query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def resolve(self, name: str, type: str = "org") -> Optional[dict]:
        """Resolve an entity name to a canonical record."""
        resp = httpx.post(
            f"{self.server_url}/resolve",
            json={"name": name, "type": type},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
