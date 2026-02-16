"""
Entity resolver utilities for database lookups.

Provides shared functionality for resolving entity names against
the organization and person databases.
"""

import logging
from typing import Optional

from .models import CompanyRecord, ResolvedOrganization

logger = logging.getLogger(__name__)

# Source prefix mapping for canonical IDs
SOURCE_PREFIX_MAP = {
    "gleif": "LEI",
    "sec_edgar": "SEC-CIK",
    "companies_house": "UK-CH",
    "wikidata": "WIKIDATA",
    "wikipedia": "WIKIDATA",
}


def get_source_prefix(source: str) -> str:
    """Get the canonical ID prefix for a data source."""
    return SOURCE_PREFIX_MAP.get(source, source.upper())


class OrganizationResolver:
    """
    Resolves organization names against the organization database.

    Shared utility that can be used by both EmbeddingCompanyQualifier
    and PersonQualifierPlugin for resolving organization references.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.7,
        auto_download_db: bool = True,
    ):
        """
        Initialize the organization resolver.

        Args:
            db_path: Path to database (auto-detects if None)
            top_k: Number of candidates to retrieve
            min_similarity: Minimum similarity threshold
            auto_download_db: Whether to auto-download database
        """
        self._db_path = db_path
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._auto_download_db = auto_download_db

        # Lazy-loaded components
        self._database = None
        self._embedder = None
        self._cache: dict[str, Optional[ResolvedOrganization]] = {}

    def _get_database(self):
        """Get or initialize the organization database."""
        if self._database is not None:
            return self._database

        try:
            from .store import get_database
            from .hub import get_database_path

            db_path = self._db_path
            if db_path is None:
                db_path = get_database_path(auto_download=self._auto_download_db)

            if db_path is None:
                logger.warning("Organization database not available.")
                return None

            self._database = get_database(db_path=db_path)
            return self._database
        except Exception as e:
            logger.warning(f"Failed to load organization database: {e}")
            return None

    def _get_embedder(self):
        """Get or initialize the embedder."""
        if self._embedder is not None:
            return self._embedder

        try:
            from .embeddings import CompanyEmbedder
            self._embedder = CompanyEmbedder()
            return self._embedder
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}")
            return None

    def resolve(self, org_name: str, use_cache: bool = True) -> Optional[ResolvedOrganization]:
        """
        Resolve an organization name against the database.

        Args:
            org_name: Organization name to resolve
            use_cache: Whether to use cached results

        Returns:
            ResolvedOrganization if found, None otherwise
        """
        if not org_name:
            return None

        # Check cache
        cache_key = org_name.lower().strip()
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        database = self._get_database()
        if database is None:
            return None

        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            # Embed the org name
            query_embedding = embedder.embed(org_name)

            # Search with embeddings-only (fast KNN MATCH path)
            results = database.search(
                query_embedding,
                top_k=self._top_k,
            )

            # Filter by similarity threshold
            results = [(r, s) for r, s in results if s >= self._min_similarity]

            if not results:
                if use_cache:
                    self._cache[cache_key] = None
                return None

            # Take the best match
            record, similarity = results[0]
            resolved = self._build_resolved_organization(record, similarity)

            if use_cache:
                self._cache[cache_key] = resolved

            return resolved

        except Exception as e:
            logger.debug(f"Failed to resolve organization '{org_name}': {e}")
            if use_cache:
                self._cache[cache_key] = None
            return None

    def resolve_with_candidates(
        self,
        org_name: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Get organization candidates with similarity scores.

        Args:
            org_name: Organization name to search
            top_k: Number of candidates (uses instance default if None)

        Returns:
            List of (CompanyRecord, similarity) tuples
        """
        if not org_name:
            return []

        database = self._get_database()
        if database is None:
            return []

        embedder = self._get_embedder()
        if embedder is None:
            return []

        try:
            query_embedding = embedder.embed(org_name)
            results = database.search(
                query_embedding,
                top_k=top_k or self._top_k,
            )
            return [(r, s) for r, s in results if s >= self._min_similarity]
        except Exception as e:
            logger.debug(f"Failed to search for organization '{org_name}': {e}")
            return []

    def _build_resolved_organization(
        self,
        record: CompanyRecord,
        similarity: float,
    ) -> ResolvedOrganization:
        """Build ResolvedOrganization from a database record."""
        source_prefix = get_source_prefix(record.source)

        return ResolvedOrganization(
            canonical_name=record.name,
            canonical_id=f"{source_prefix}:{record.source_id}",
            source=record.source,
            source_id=record.source_id,
            region=record.region or None,
            match_confidence=min(max(similarity, 0.0), 1.0),
            match_details={"similarity": similarity},
        )


# Singleton instance for shared use
_default_resolver: Optional[OrganizationResolver] = None


def get_organization_resolver(
    db_path: Optional[str] = None,
    auto_download_db: bool = True,
) -> OrganizationResolver:
    """
    Get or create a shared OrganizationResolver instance.

    Args:
        db_path: Path to database
        auto_download_db: Whether to auto-download database

    Returns:
        OrganizationResolver instance
    """
    global _default_resolver

    if _default_resolver is None:
        _default_resolver = OrganizationResolver(
            db_path=db_path,
            auto_download_db=auto_download_db,
        )

    return _default_resolver
