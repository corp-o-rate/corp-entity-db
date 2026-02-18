# corp-entity-db

Entity database and semantic search engine for organizations, people, roles, and locations. Combines data from GLEIF, SEC Edgar, Wikidata, and UK Companies House into a unified searchable database with embedding-based similarity search.

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
```

## Frontend

```bash
pnpm install && pnpm dev
```

The frontend connects to either a local server (`corp-entity-db serve`) or a RunPod serverless endpoint, configured via environment variables (`RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY`).

## Data Sources

| Source | Scale |
|--------|-------|
| Companies House | ~5.5M orgs, ~27.5M people |
| Wikidata | ~1.7M orgs, ~39.4M people |
| GLEIF | ~2.6M orgs |
| SEC Edgar | ~73K orgs |

## Architecture

Embeddings are generated using `google/embeddinggemma-300m` (768 dimensions) and stored as float32 BLOBs directly in the SQLite `organizations` and `people` tables. Search uses USearch HNSW indexes built from int8-quantized embeddings for fast approximate nearest neighbor lookup. The lite database variant drops the embedding column entirely and relies on pre-built USearch index files.

See [`entity-db-lib/README.md`](entity-db-lib/README.md) for full library documentation.
