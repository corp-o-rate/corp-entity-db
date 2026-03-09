# corp-entity-db

Entity database and semantic search engine for organizations, people, roles, and locations. Combines data from GLEIF, SEC Edgar, Wikidata, and UK Companies House into a unified searchable database with embedding-based similarity search.

## Quick Start

```bash
pip install corp-entity-db

# Download the pre-built database (~30GB lite) + USearch indexes
corp-entity-db download

# Search organizations
corp-entity-db search "Apple Inc"

# Search people — composite embeddings match on name + role + org
corp-entity-db search-people "Tim Cook" --role CEO --org Apple

# Search roles and locations
corp-entity-db search-roles "CEO"
corp-entity-db search-locations "California"

# Show database statistics
corp-entity-db status
```

## Python API

```python
from corp_entity_db import OrganizationDatabase, CompanyEmbedder, get_database_path

# Search organizations
db = OrganizationDatabase(get_database_path())
embedder = CompanyEmbedder()
matches = db.search(embedder.embed("Microsoft"), top_k=10)
for record, score in matches:
    print(f"{record.name} ({record.entity_type}) - score: {score:.3f}")

# Search people (composite embeddings + name fallback + identity fallback)
from corp_entity_db import PersonDatabase, get_person_database
person_db = get_person_database()
query_emb = embedder.embed_composite_person("Tim Cook", role="CEO", org="Apple")
matches = person_db.search(
    query_emb, top_k=5, query_name="Tim Cook",
    embedder=embedder, query_role="CEO", query_org="Apple",
)

# Resolve organization names to canonical records
from corp_entity_db import OrganizationResolver, get_organization_resolver
resolver = get_organization_resolver()
result = resolver.resolve("MSFT")
```

## Server Mode

Keep models warm in memory for low-latency repeated searches (avoids ~30s startup per invocation):

```bash
pip install "corp-entity-db[serve]"
corp-entity-db serve                  # Start on localhost:8222
corp-entity-db serve --port 9000      # Custom port
```

## Components

| Component | Description |
|-----------|-------------|
| **`entity-db-lib/`** | Python library ([PyPI: `corp-entity-db`](https://pypi.org/project/corp-entity-db/)) — CLI, importers, and search API |
| **`src/`** | Next.js frontend search demo (React 19, Tailwind CSS) |
| **`runpod/`** | Docker image for RunPod serverless deployment |

## Data Sources

| Source | Description | Scale |
|--------|-------------|-------|
| Wikidata | Organizations, people, roles & locations | ~1.6M orgs, ~28M people, ~700K locations, ~180K roles |
| GLEIF | Legal Entity Identifier records | ~2.6M orgs |
| SEC Edgar | US public company filers & officers | ~73K orgs |
| Companies House | UK registered companies & officers | ~5.5M orgs |

## Entity Types

Organizations are classified into 17 types: `business`, `fund`, `branch`, `nonprofit`, `ngo`, `foundation`, `government`, `international_org`, `educational`, `research`, `healthcare`, `media`, `sports`, `political_party`, `religious`, `trade_union`, `unknown`.

People are classified into 12 types: `executive`, `politician`, `government`, `military`, `legal`, `professional`, `academic`, `artist`, `media`, `athlete`, `journalist`, `activist`.

## Search Architecture

All embeddings are generated using `google/embeddinggemma-300m` (768 dimensions) on-the-fly during index building and stored only in USearch HNSW indexes (never in SQLite). Search uses int8-quantized embeddings for fast approximate nearest neighbor lookup.

People search uses a **three-tier fallback strategy** achieving **100% acc@1** and 100% acc@20 on 280 queries across 12 person types (100-200ms per query after warmup):

1. **Composite HNSW index** (`people_usearch_v5.bin`, 768-dim) — name, role, and organization embedded as separate 256-dim vectors via Matryoshka truncation, independently L2-normalized, weighted (name=8, role=1, org=4), and concatenated. Only indexes people with org associations. AND-style matching: a poor match on organization cannot be compensated by a good match on role.

2. **SQL name-lookup fallback** — when composite scores are below threshold, exact name matching via `name_normalized` (using `corp-names` for normalization). Disambiguation blends description embedding similarity (40%), display-name Levenshtein similarity (45%), and popularity via log-scaled canon_size (15%). Multi-description support tries alternative role/org combinations within canonical groups.

3. **Identity HNSW index** (`people_identity_usearch_v5.bin`, 256-dim) — name-only embeddings for all people as a final fallback.

## Database Variants

| Variant | Description | Default |
|---------|-------------|---------|
| **Lite** | `record` column stripped, `name_normalized` kept | Yes |
| **Full** | All columns with full source record metadata | No |

Both variants store embeddings only in versioned USearch index files (`organizations_usearch_v5.bin`, `people_usearch_v5.bin`, `people_identity_usearch_v5.bin`), never in SQLite. Database is hosted on HuggingFace: [Corp-o-Rate-Community/entity-references](https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references).

## Frontend

```bash
pnpm install && pnpm dev
```

The frontend connects to either a local server (`corp-entity-db serve`) or a RunPod serverless endpoint, configured via environment variables (`RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY`).

## Installation Extras

```bash
pip install corp-entity-db              # Default: search only
pip install "corp-entity-db[build]"     # Add import/build support (orjson, indexed-bzip2)
pip install "corp-entity-db[serve]"     # Add HTTP server (FastAPI, uvicorn)
pip install "corp-entity-db[client]"    # Add remote client (httpx)
pip install "corp-entity-db[all]"       # Everything
```

See [`entity-db-lib/README.md`](entity-db-lib/README.md) for full library documentation including database management commands, import workflows, and detailed API reference.
