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

# Search people (composite embeddings: name + role + org)
corp-entity-db search-people "Tim Cook"
corp-entity-db search-people "Tim Cook" --role CEO --org Apple

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

# Search people with composite embeddings + name fallback + identity fallback
from corp_entity_db import PersonDatabase, get_person_database
person_db = get_person_database()
query_emb = embedder.embed_composite_person("Tim Cook", role="CEO", org="Apple")
matches = person_db.search(query_emb, top_k=5, query_name="Tim Cook", embedder=embedder, query_role="CEO", query_org="Apple")
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
| Wikidata | Organizations, people, roles & locations | ~1.6M orgs, ~28M people, ~700K locations, ~180K roles |
| GLEIF | Legal Entity Identifier records | ~2.6M orgs |
| SEC Edgar | US public company filers & officers | ~73K orgs |
| Companies House | UK registered companies | ~5.5M orgs |

## Embedding Architecture

**All embeddings** (organizations and people) exist only in USearch HNSW indexes, never in SQLite. They are generated on-the-fly during index building using `google/embeddinggemma-300m`. Int8 scalar quantization is computed during index building and is not stored separately.

**People (Dual-Index Search)**: People use two USearch HNSW indexes:

- **Primary composite index** (`people_usearch_v5.bin`, 768-dim): Name, role, and organization are embedded as separate 256-dim vectors using Matryoshka truncation, independently L2-normalized, weighted (name=8, role=1, org=4), and concatenated. Only indexes people with org associations. This gives AND-style matching: a poor match on organization cannot be compensated by a good match on name, enabling precise queries like "find the CEO named Tim Cook at Apple." Built by `build_people_composite_index()`.
- **Secondary identity index** (`people_identity_usearch_v5.bin`, 256-dim): Name-only embeddings with Matryoshka truncation to 256 dims for all people. Consulted as fallback when composite search and SQL name lookup fail. Built by `build_people_identity_index()`.

Search accuracy: 96.1% acc@1, 98.9% acc@20 on 280 queries across 12 person types (95-165ms per query after model warmup). Three-tier fallback: composite HNSW → SQL name_normalized lookup → identity HNSW.

## Database Variants

- **Lite** (default download): `record` and `name_normalized` columns dropped for smaller download
- **Full**: Includes all columns with source record metadata

In both variants, all embeddings exist only in versioned USearch index files (`organizations_usearch_v5.bin`, `people_usearch_v5.bin`, `people_identity_usearch_v5.bin`), never in SQLite.

## Database Management

```bash
corp-entity-db migrate                 # Migrate schema to latest (v5)
corp-entity-db post-import             # Build USearch indexes + VACUUM
corp-entity-db build-index             # Rebuild all USearch HNSW indexes
corp-entity-db build-identity-index    # Rebuild only the people identity index
corp-entity-db normalize-people        # Normalize people names using corp-names
corp-entity-db reclassify-people       # Recalculate person_type classifications
corp-entity-db people-test             # Run people search accuracy test
corp-entity-db org-test                # Run organization search accuracy test
```

HuggingFace dataset: [Corp-o-Rate-Community/entity-references](https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references)
