# corp-entity-db

Entity database library and search engine for organizations, people, roles, and locations. Provides embedding-based semantic search over entities imported from GLEIF, SEC Edgar, Wikidata, and Companies House.

## Installation

```bash
# Default: search and resolve (no build dependencies)
pip install corp-entity-db

# With database build/import support
pip install "corp-entity-db[build]"

# With HTTP server (corp-entity-db serve)
pip install "corp-entity-db[serve]"

# With remote client (EntityDBClient)
pip install "corp-entity-db[client]"

# Everything
pip install "corp-entity-db[all]"
```

The default install includes sentence-transformers, USearch, and huggingface_hub for searching and downloading pre-built databases. The embedding model (`google/embeddinggemma-300m`, 300M params) is downloaded automatically on first use.

## Quick Start

```bash
# Download the lite database + USearch indexes
corp-entity-db download

# Search organizations
corp-entity-db search "Microsoft"
corp-entity-db search "Microsoft" --hybrid

# Search people
corp-entity-db search-people "Tim Cook"

# Show database statistics
corp-entity-db status
```

## Python API

```python
from corp_entity_db import OrganizationDatabase, get_database_path

db = OrganizationDatabase(get_database_path())
matches = db.search("Microsoft", limit=10)
for match in matches:
    print(f"{match.record.name} ({match.record.entity_type}) - score: {match.score:.3f}")
```

## Server Mode

Keep models warm in memory for low-latency repeated searches (requires `[serve]` extra):

```bash
corp-entity-db serve                  # Start on localhost:8222
corp-entity-db serve --port 9000      # Custom port
```

## Data Sources

| Source | Description | Scale |
|--------|-------------|-------|
| Companies House | UK registered companies + officers | ~5.5M orgs, ~27.5M people |
| Wikidata | Organizations & notable people | ~1.7M orgs, ~39.4M people |
| GLEIF | Legal Entity Identifier records | ~2.6M orgs |
| SEC Edgar | US public company filers & officers | ~73K orgs |
| **Total** | | **~9.9M orgs, ~66.9M people** |

## Embedding Architecture

Embeddings are stored as float32 BLOBs directly in the `organizations` and `people` tables. The full database enforces NOT NULL on the embedding column. Int8 scalar quantization is computed on-the-fly during USearch HNSW index building and is not stored separately.

## Database Variants

- **Lite** (default download): Embedding column dropped, uses pre-built USearch HNSW indexes for search
- **Full**: Includes float32 embedding BLOBs in main tables

## Database Management

```bash
corp-entity-db post-import             # Generate embeddings + build USearch indexes + VACUUM
corp-entity-db build-index             # Rebuild USearch HNSW indexes only
corp-entity-db repair-embeddings       # Generate missing embeddings
corp-entity-db migrate-embeddings      # Migrate from legacy vec0 tables to embedding column
```

HuggingFace dataset: [Corp-o-Rate-Community/entity-references](https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references)
