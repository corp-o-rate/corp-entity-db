"""
Persistent local server for entity database search.

Keeps databases, USearch indexes, and embedding model warm in memory
so repeated CLI invocations avoid the ~30s startup cost.

Usage:
    corp-entity-db serve                    # Start on localhost:8222
    corp-entity-db serve --port 9000        # Custom port
"""

import logging
import time
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    hybrid: bool = False


class SearchPeopleRequest(BaseModel):
    query: str
    limit: int = 10


class SearchRolesRequest(BaseModel):
    query: str
    limit: int = 10


class SearchLocationsRequest(BaseModel):
    query: str
    limit: int = 10


class ResolveRequest(BaseModel):
    name: str
    type: Literal["org", "person"] = "org"


# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------

_org_db = None
_person_db = None
_roles_db = None
_locations_db = None
_embedder = None
_db_path: Optional[str] = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from .embeddings import CompanyEmbedder
        _embedder = CompanyEmbedder()
    return _embedder


def _get_org_db():
    global _org_db
    if _org_db is None:
        from .store import get_database
        from .hub import get_database_path
        db_path = _db_path or get_database_path(auto_download=True)
        _org_db = get_database(db_path=db_path, readonly=True)
    return _org_db


def _get_person_db():
    global _person_db
    if _person_db is None:
        from .store import get_person_database
        from .hub import get_database_path
        db_path = _db_path or get_database_path(auto_download=True)
        _person_db = get_person_database(db_path=db_path, readonly=True)
    return _person_db


def _get_roles_db():
    global _roles_db
    if _roles_db is None:
        from .store import RolesDatabase
        from .hub import get_database_path
        db_path = _db_path or get_database_path(auto_download=True)
        _roles_db = RolesDatabase(db_path=db_path, readonly=True)
    return _roles_db


def _get_locations_db():
    global _locations_db
    if _locations_db is None:
        from .store import LocationsDatabase
        from .hub import get_database_path
        db_path = _db_path or get_database_path(auto_download=True)
        _locations_db = LocationsDatabase(db_path=db_path, readonly=True)
    return _locations_db


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Corp Entity DB Server",
    description="Persistent local server for entity database search with warm indexes.",
)


@app.get("/")
def health():
    """Health check and status info."""
    from .hub import get_database_path

    db_path = _db_path or get_database_path(auto_download=False)

    result: dict[str, Any] = {
        "status": "ok",
        "db_path": str(db_path) if db_path else None,
        "indexes_loaded": False,
    }

    if _org_db is not None:
        try:
            stats = _org_db.get_stats()
            result["org_count"] = stats.total_records
            result["indexes_loaded"] = _org_db._hnsw_index is not None
        except Exception:
            result["org_count"] = 0

    if _person_db is not None:
        try:
            stats = _person_db.get_stats()
            result["person_count"] = stats.get("total_records", 0)
        except Exception:
            result["person_count"] = 0

    return result


@app.post("/search")
def search_organizations(req: SearchRequest):
    """Search organizations by name."""
    t0 = time.time()
    logger.info(f"Search organizations: '{req.query}' (limit={req.limit}, hybrid={req.hybrid})")

    embedder = _get_embedder()
    org_db = _get_org_db()

    query_embedding = embedder.embed(req.query)

    results = org_db.search(
        query_embedding,
        top_k=req.limit,
        query_text=req.query if req.hybrid else None,
    )

    elapsed = time.time() - t0
    logger.info(f"Search returned {len(results)} results in {elapsed:.3f}s")

    from .models import CompanyMatch
    matches = []
    for record, score in results:
        match = CompanyMatch.from_record(
            query_name=req.query,
            record=record,
            similarity_score=score,
        )
        matches.append(match.model_dump())

    return matches


@app.post("/search-people")
def search_people(req: SearchPeopleRequest):
    """Search people by name."""
    t0 = time.time()
    logger.info(f"Search people: '{req.query}' (limit={req.limit})")

    embedder = _get_embedder()
    person_db = _get_person_db()

    query_embedding = embedder.embed(req.query)

    results = person_db.search(
        query_embedding,
        top_k=req.limit,
        query_text=req.query,
    )

    elapsed = time.time() - t0
    logger.info(f"People search returned {len(results)} results in {elapsed:.3f}s")

    from .models import PersonMatch
    matches = []
    for record, score in results:
        match = PersonMatch.from_record(
            query_name=req.query,
            record=record,
            similarity_score=score,
        )
        matches.append(match.model_dump())

    return matches


@app.post("/search-roles")
def search_roles(req: SearchRolesRequest):
    """Search roles/job titles."""
    t0 = time.time()
    logger.info(f"Search roles: '{req.query}' (limit={req.limit})")

    roles_db = _get_roles_db()
    results = roles_db.search(req.query, top_k=req.limit)

    elapsed = time.time() - t0
    logger.info(f"Roles search returned {len(results)} results in {elapsed:.3f}s")

    return [
        {"id": role_id, "name": name, "score": score}
        for role_id, name, score in results
    ]


@app.post("/search-locations")
def search_locations(req: SearchLocationsRequest):
    """Search locations."""
    t0 = time.time()
    logger.info(f"Search locations: '{req.query}' (limit={req.limit})")

    locations_db = _get_locations_db()
    results = locations_db.search(req.query, top_k=req.limit)

    elapsed = time.time() - t0
    logger.info(f"Locations search returned {len(results)} results in {elapsed:.3f}s")

    return [
        {"id": loc_id, "name": name, "score": score}
        for loc_id, name, score in results
    ]


@app.post("/resolve")
def resolve_entity(req: ResolveRequest):
    """Resolve an entity name to a canonical record."""
    t0 = time.time()
    logger.info(f"Resolve: '{req.name}' (type={req.type})")

    if req.type == "org":
        from .resolver import OrganizationResolver
        resolver = OrganizationResolver(db_path=_db_path)
        result = resolver.resolve(req.name)
        elapsed = time.time() - t0
        logger.info(f"Resolve completed in {elapsed:.3f}s: {'found' if result else 'not found'}")
        return result.model_dump() if result else None
    else:
        # Person resolution: find best match via embedding search
        embedder = _get_embedder()
        person_db = _get_person_db()
        query_embedding = embedder.embed(req.name)
        results = person_db.search(query_embedding, top_k=1, query_text=req.name)
        elapsed = time.time() - t0
        if results:
            record, score = results[0]
            from .models import PersonMatch
            match = PersonMatch.from_record(
                query_name=req.name,
                record=record,
                similarity_score=score,
            )
            logger.info(f"Resolve completed in {elapsed:.3f}s: found '{record.name}' (score={score:.3f})")
            return match.model_dump()
        logger.info(f"Resolve completed in {elapsed:.3f}s: not found")
        return None


# ---------------------------------------------------------------------------
# Warmup and run
# ---------------------------------------------------------------------------


def warmup(db_path: Optional[str] = None):
    """Eagerly load all databases and embedding model."""
    global _db_path
    if db_path:
        _db_path = db_path

    logger.info("Warming up entity database server...")
    t0 = time.time()

    # Load embedding model
    t1 = time.time()
    embedder = _get_embedder()
    logger.info(f"  Embedding model loaded (dim={embedder.embedding_dim}) ({time.time() - t1:.1f}s)")

    # Load organization database
    t1 = time.time()
    org_db = _get_org_db()
    org_stats = org_db.get_stats()
    logger.info(f"  Organization DB loaded ({org_stats.total_records} records) ({time.time() - t1:.1f}s)")

    # Load person database
    t1 = time.time()
    person_db = _get_person_db()
    person_stats = person_db.get_stats()
    logger.info(f"  Person DB loaded ({person_stats.get('total_records', 0)} records) ({time.time() - t1:.1f}s)")

    # Load roles database
    t1 = time.time()
    roles_db = _get_roles_db()
    roles_stats = roles_db.get_stats()
    logger.info(f"  Roles DB loaded ({roles_stats.get('total_roles', 0)} roles) ({time.time() - t1:.1f}s)")

    # Load locations database
    t1 = time.time()
    locations_db = _get_locations_db()
    locations_stats = locations_db.get_stats()
    logger.info(f"  Locations DB loaded ({locations_stats.get('total_locations', 0)} locations) ({time.time() - t1:.1f}s)")

    # Warm up embedding model with a test query
    t1 = time.time()
    test_embedding = embedder.embed("test warmup")
    org_db.search(test_embedding, top_k=1)
    logger.info(f"  Warmup search done ({time.time() - t1:.1f}s)")

    logger.info(f"Warmup complete in {time.time() - t0:.1f}s")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8222,
    do_warmup: bool = True,
    db_path: Optional[str] = None,
    verbose: bool = False,
):
    """Run the server with uvicorn."""
    import uvicorn

    log_level = "debug" if verbose else "info"

    if do_warmup:
        warmup(db_path=db_path)

    logger.info(f"Starting entity DB server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
