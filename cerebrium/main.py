"""Cerebrium handler for corp-entity-db entity search.

Mirrors runpod/handler.py. Cerebrium auto-exposes every public top-level function
as a POST endpoint, so `search(...)` below becomes:

  POST https://api.aws.us-east-1.cerebrium.ai/v4/<project>/corp-entity-db/search

Request body keys map directly to function kwargs.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

print("=" * 60)
print("[init] probing /persistent-storage for project volume")
_VOLUME: Optional[Path] = None
for _p in ("/persistent-storage", "/workspace"):
    _P = Path(_p)
    if _P.is_dir():
        _free_gb = shutil.disk_usage(_p).free / 1024 ** 3
        _is_mount = os.path.ismount(_p)
        print(f"[init]   {_p}: exists is_mount={_is_mount} free={_free_gb:.1f} GB")
        if _free_gb >= 100 and _VOLUME is None:
            _VOLUME = _P
            print(f"[init] selected volume: {_p}")
    else:
        print(f"[init]   {_p}: missing")

if _VOLUME is not None:
    import corp_entity_db.hub as _hub
    _hub.DEFAULT_CACHE_DIR = _VOLUME
    print(f"[init] hub.DEFAULT_CACHE_DIR = {_VOLUME}")
else:
    print("[init] WARNING: no volume with >=100 GB free — resize the project volume to 200 GB before sending requests")
print("=" * 60)

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

MAX_CACHE_SIZE_BYTES = 1 * 1024 * 1024 * 1024

embedder: Optional[CompanyEmbedder] = None
org_db: Optional[OrganizationDatabase] = None
person_db: Optional[PersonDatabase] = None
roles_db: Optional[RolesDatabase] = None
locations_db: Optional[LocationsDatabase] = None
_initialized = False

result_cache: OrderedDict[str, str] = OrderedDict()
cache_size_bytes = 0


def _cache_key(query: str, search_type: str, limit: int, role: Optional[str], org: Optional[str]) -> str:
    return hashlib.sha256(f"{search_type}:limit={limit}:role={role}:org={org}:{query}".encode()).hexdigest()


def _evict_if_needed(new_entry_size: int) -> None:
    global cache_size_bytes
    while result_cache and (cache_size_bytes + new_entry_size) > MAX_CACHE_SIZE_BYTES:
        _, oldest_value = result_cache.popitem(last=False)
        cache_size_bytes -= len(oldest_value.encode())


def _cache_put(key: str, payload: str) -> None:
    global cache_size_bytes
    size = len(key.encode()) + len(payload.encode())
    _evict_if_needed(size)
    result_cache[key] = payload
    cache_size_bytes += size


def _initialize() -> None:
    """Lazy: runs on the first call to search().

    Done lazily so Cerebrium can boot the replica even when the volume is too
    small (allowing the operator to resize without a crash loop).
    """
    global embedder, org_db, person_db, roles_db, locations_db, _initialized
    if _initialized:
        return

    logger.info("Initialising entity search handler...")
    db_path = get_database_path(auto_download=True)
    if db_path is None:
        logger.info("Database not found locally, downloading...")
        db_path = download_database()
    logger.info(f"Using database at {db_path}")

    logger.info("Loading embedding model...")
    embedder = CompanyEmbedder()
    _ = embedder.embedding_dim
    logger.info(f"Embedder loaded (dim={embedder.embedding_dim})")

    org_db = get_database(db_path=db_path)
    person_db = get_person_database(db_path=db_path)
    roles_db = get_roles_database(db_path=db_path)
    locations_db = get_locations_database(db_path=db_path)
    logger.info("All databases initialised")

    logger.info("Eager-loading + prewarming HNSW indexes...")
    t0 = time.time()
    org_db._load_hnsw_index()
    if org_db._hnsw_index is not None:
        org_db._hnsw_index.preload_all()
    person_db._load_hnsw_index()
    if person_db._hnsw_index is not None:
        person_db._hnsw_index.preload_all()
    person_db._load_identity_hnsw_index()
    if person_db._identity_hnsw_index is not None:
        person_db._identity_hnsw_index.preload_all()
    logger.info(f"All HNSW indexes ready in {time.time() - t0:.1f}s")

    _initialized = True


def _search_organizations(query: str, limit: int) -> list[dict]:
    embedding = embedder.embed(query)
    return [
        {"record": record.model_dump(mode="json"), "score": float(score)}
        for record, score in org_db.search(query_embedding=embedding, top_k=limit)
    ]


def _search_people(query: str, limit: int, role: Optional[str], org: Optional[str], person_type: Optional[str]) -> list[dict]:
    embedding = embedder.embed_composite_person(query, role=role, org=org)
    results = person_db.search(
        query_embedding=embedding,
        top_k=limit,
        query_name=query,
        embedder=embedder,
        query_person_type=person_type,
        query_role=role,
        query_org=org,
    )
    return [{"record": record.model_dump(mode="json"), "score": float(score)} for record, score in results]


def _search_roles(query: str, limit: int) -> list[dict]:
    return [
        {"role_id": role_id, "role_name": role_name, "score": float(score)}
        for role_id, role_name, score in roles_db.search(query=query, top_k=limit)
    ]


def _search_locations(query: str, limit: int) -> list[dict]:
    return [
        {"location_id": loc_id, "location_name": loc_name, "score": float(score)}
        for loc_id, loc_name, score in locations_db.search(query=query, top_k=limit)
    ]


def search(
    query: str,
    type: str = "org",
    limit: int = 10,
    role: Optional[str] = None,
    org: Optional[str] = None,
    person_type: Optional[str] = None,
) -> dict:
    """Entity search endpoint. Maps the RunPod handler's `input` payload to kwargs.

    type: "org" | "person" | "role" | "location"
    """
    _initialize()

    query = (query or "").strip()
    if not query:
        return {"error": "No query provided. Send {\"query\": \"...\", \"type\": \"org\"}"}

    if type not in ("org", "person", "role", "location"):
        return {"error": f"Invalid type '{type}'. Must be one of: org, person, role, location"}

    limit = int(limit)
    logger.info(f"Search: type={type}, query='{query}', limit={limit}, role={role}, org={org}")

    cache_key = _cache_key(query, type, limit, role, org)
    if cache_key in result_cache:
        result_cache.move_to_end(cache_key)
        cached = json.loads(result_cache[cache_key])
        cached["cached"] = True
        return cached

    t0 = time.time()
    if type == "org":
        results = _search_organizations(query, limit)
    elif type == "person":
        results = _search_people(query, limit, role=role, org=org, person_type=person_type)
    elif type == "role":
        results = _search_roles(query, limit)
    else:
        results = _search_locations(query, limit)
    elapsed = time.time() - t0

    response = {
        "query": query,
        "type": type,
        "results": results,
        "count": len(results),
        "elapsed_ms": round(elapsed * 1000, 1),
    }
    _cache_put(cache_key, json.dumps(response))
    logger.info(f"Search complete in {elapsed:.3f}s, {len(results)} results")
    return response
