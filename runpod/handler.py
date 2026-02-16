"""
RunPod Serverless Handler for Entity Database Search

Provides organization, person, role, and location search against
the corp-entity-db database with USearch HNSW indexes.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

import runpod

from corp_entity_db.embeddings import CompanyEmbedder
from corp_entity_db.hub import download_database, get_database_path
from corp_entity_db.store import (
    LocationsDatabase,
    OrganizationDatabase,
    PersonDatabase,
    RolesDatabase,
    get_database,
    get_locations_database,
    get_person_database,
    get_roles_database,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration (default 1GB - search results are small)
MAX_CACHE_SIZE_BYTES = 1 * 1024 * 1024 * 1024

# Global state
embedder: Optional[CompanyEmbedder] = None
org_db: Optional[OrganizationDatabase] = None
person_db: Optional[PersonDatabase] = None
roles_db: Optional[RolesDatabase] = None
locations_db: Optional[LocationsDatabase] = None

result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0


def get_cache_key(query: str, search_type: str, limit: int, hybrid: bool) -> str:
    """Generate a cache key from input parameters."""
    key_str = f"{search_type}:limit={limit}:hybrid={hybrid}:{query}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def get_entry_size(key: str, value: str) -> int:
    """Estimate memory size of a cache entry in bytes."""
    return len(key.encode()) + len(value.encode())


def evict_if_needed(new_entry_size: int):
    """Evict oldest entries if cache would exceed size limit."""
    global cache_size_bytes

    while result_cache and (cache_size_bytes + new_entry_size) > MAX_CACHE_SIZE_BYTES:
        oldest_key, oldest_value = result_cache.popitem(last=False)
        evicted_size = get_entry_size(oldest_key, oldest_value)
        cache_size_bytes -= evicted_size
        logger.info(f"Evicted cache entry: {oldest_key[:16]}... (freed {evicted_size} bytes)")


def cache_result(cache_key: str, result: str):
    """Store result in cache."""
    global cache_size_bytes
    entry_size = get_entry_size(cache_key, result)
    evict_if_needed(entry_size)
    result_cache[cache_key] = result
    cache_size_bytes += entry_size


def initialize():
    """Load embedder and databases at startup."""
    global embedder, org_db, person_db, roles_db, locations_db

    logger.info("Initializing entity search handler...")

    # Ensure database is downloaded
    db_path = get_database_path(auto_download=True)
    if db_path is None:
        logger.info("Database not found locally, downloading...")
        db_path = download_database()
    logger.info(f"Using database at {db_path}")

    # Initialize embedder (needed for org and person vector search)
    logger.info("Loading embedding model...")
    embedder = CompanyEmbedder()
    # Force model load
    _ = embedder.embedding_dim
    logger.info(f"Embedder loaded (dim={embedder.embedding_dim})")

    # Initialize databases (singleton pattern, readonly)
    org_db = get_database(db_path=db_path)
    person_db = get_person_database(db_path=db_path)
    roles_db = get_roles_database(db_path=db_path)
    locations_db = get_locations_database(db_path=db_path)

    logger.info("All databases initialized")


def search_organizations(query: str, limit: int, hybrid: bool) -> list[dict]:
    """Search organizations using embedding similarity."""
    embedding = embedder.embed(query)
    results = org_db.search(
        query_embedding=embedding,
        top_k=limit,
        query_text=query if hybrid else None,
    )
    return [
        {"record": record.model_dump(mode="json"), "score": float(score)}
        for record, score in results
    ]


def search_people(query: str, limit: int, hybrid: bool) -> list[dict]:
    """Search people using embedding similarity."""
    embedding = embedder.embed(query)
    results = person_db.search(
        query_embedding=embedding,
        top_k=limit,
        query_text=query if hybrid else None,
    )
    return [
        {"record": record.model_dump(mode="json"), "score": float(score)}
        for record, score in results
    ]


def search_roles(query: str, limit: int) -> list[dict]:
    """Search roles by name (text-based, no embeddings)."""
    results = roles_db.search(query=query, top_k=limit)
    return [
        {"role_id": role_id, "role_name": role_name, "score": float(score)}
        for role_id, role_name, score in results
    ]


def search_locations(query: str, limit: int) -> list[dict]:
    """Search locations by name (text-based, no embeddings)."""
    results = locations_db.search(query=query, top_k=limit)
    return [
        {"location_id": loc_id, "location_name": loc_name, "score": float(score)}
        for loc_id, loc_name, score in results
    ]


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "query": "Microsoft",
            "type": "org",        // "org" | "person" | "role" | "location"
            "limit": 10,          // optional, default 10
            "hybrid": false       // optional, default false (text + embeddings for org/person)
        }
    }
    """
    logger.info(f"Received job: {job.get('id', 'unknown')}")

    job_input = job.get("input", {})
    query = job_input.get("query", "").strip()

    if not query:
        return {"error": "No query provided. Send {\"input\": {\"query\": \"...\", \"type\": \"org\"}}"}

    search_type = job_input.get("type", "org")
    limit = int(job_input.get("limit", 10))
    hybrid = bool(job_input.get("hybrid", False))

    if search_type not in ("org", "person", "role", "location"):
        return {"error": f"Invalid type '{search_type}'. Must be one of: org, person, role, location"}

    logger.info(f"Search: type={search_type}, query='{query}', limit={limit}, hybrid={hybrid}")

    # Check cache
    cache_key = get_cache_key(query, search_type, limit, hybrid)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        logger.info(f"Cache hit: {cache_key[:16]}...")
        cached = json.loads(result_cache[cache_key])
        cached["cached"] = True
        return cached

    start_time = time.time()

    if search_type == "org":
        results = search_organizations(query, limit, hybrid)
    elif search_type == "person":
        results = search_people(query, limit, hybrid)
    elif search_type == "role":
        results = search_roles(query, limit)
    elif search_type == "location":
        results = search_locations(query, limit)

    elapsed = time.time() - start_time
    logger.info(f"Search complete in {elapsed:.3f}s, {len(results)} results")

    response = {
        "query": query,
        "type": search_type,
        "results": results,
        "count": len(results),
        "elapsed_ms": round(elapsed * 1000, 1),
    }

    # Cache the result
    cache_result(cache_key, json.dumps(response))

    return response


# Initialize at startup for faster cold starts
initialize()

# Start the serverless handler
runpod.serverless.start({"handler": handler})
