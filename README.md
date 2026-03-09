# corp-entity-db

Entity database and semantic search engine for organizations, people, roles, and locations. Combines data from GLEIF, SEC Edgar, Wikidata, and UK Companies House into a unified searchable database with embedding-based similarity search. Features dual-index people search: composite embeddings for precise AND-style matching across name, role, and organization, with an identity index fallback for people known by name and type alone.

## Components

- **`entity-db-lib/`** — Python library ([PyPI: `corp-entity-db`](https://pypi.org/project/corp-entity-db/)) with CLI, importers, and search API
- **`src/`** — Next.js frontend search demo (React 19, Tailwind CSS)
- **`runpod/`** — Docker image for RunPod serverless deployment

## Quick Start

```bash
pip install corp-entity-db

# Download the database and search
corp-entity-db download
corp-entity-db search "Apple Inc"
corp-entity-db search-people "Tim Cook"
corp-entity-db search-people "Tim Cook" --role CEO --org Apple
```

## Frontend

```bash
pnpm install && pnpm dev
```

The frontend connects to either a local server (`corp-entity-db serve`) or a RunPod serverless endpoint, configured via environment variables (`RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY`).

## Data Sources

| Source | Scale |
|--------|-------|
| Wikidata | ~1.6M orgs, ~28M people, ~700K locations, ~180K roles |
| GLEIF | ~2.6M orgs |
| SEC Edgar | ~73K orgs |
| Companies House | ~5.5M orgs |

## Architecture

All embeddings are generated using `google/embeddinggemma-300m` (768 dimensions) on-the-fly during index building and stored only in USearch HNSW indexes (never in SQLite). Search uses int8-quantized embeddings for fast approximate nearest neighbor lookup. The lite database variant drops `record` and `name_normalized` columns for a smaller download.

People search uses a **three-tier fallback strategy** achieving 100% acc@1 and 100% acc@20 on 280 queries across 12 person types (100-200ms per query after warmup). The primary composite index (`people_usearch_v5.bin`) embeds name, role, and organization as separate 256-dim vectors (via Matryoshka truncation), independently L2-normalized, weighted (name=8, role=1, org=4), and concatenated into a 768-dim vector — only for people with org associations. This gives AND-style matching where a poor match on organization cannot be compensated by a good match on role. When composite scores are below threshold, a SQL name_normalized lookup (using `corp-names` for normalization) provides exact name matching with disambiguation that blends description embedding similarity (40%), display-name Levenshtein similarity (45%), and popularity via log-scaled canon_size (15%), with multi-description support that tries alternative role/org combinations within canonical groups. A secondary identity index (`people_identity_usearch_v5.bin`) uses 256-dim Matryoshka-truncated name-only embeddings for all people as a final fallback.

See [`entity-db-lib/README.md`](entity-db-lib/README.md) for full library documentation.
