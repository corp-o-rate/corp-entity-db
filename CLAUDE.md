# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Global Preferences

### Core Principles

* I use zsh shell.
* **Fail Fast** - raise exceptions and let them bubble up. Avoid try/catch blocks unless at the top level.
* Don't add fallbacks or backwards compatibility unless instructed explicitly.
* Don't change tests to fit the code. If tests fail, **fix the code** not the test.
* We don't do silent failures - all failures MUST appear in logs or cause the application to fail.
* Everything should be strongly typed, use Pydantic models not dicts.
* Use mermaid for markdown docs when diagrams are needed.
* This is startup code - prefer lean, simple, and to the point over enterprise abstractions.
* I like logging statements, please log progress where possible.
* DO NOT REPEAT existing code (DRY) - prefer tweaking existing implementations over adding new code.

### Instruction Following

* **Be explicit and specific**: Clear, thorough implementation expected.
* **Action-oriented by default**: Proceed with implementation rather than only suggesting.
* **Concise but informative**: Brief progress summaries, avoid unnecessary verbosity.
* **Flowing prose over excessive formatting**: Use clear paragraphs. Reserve markdown primarily for `inline code`, code blocks, and simple headings.

## Project Overview

corp-entity-db is a standalone entity database library and search demo for organizations, people, roles, and locations. Extracted from the statement-extractor-lib project. It provides embedding-based semantic search over entities imported from GLEIF, SEC Edgar, Wikidata, and Companies House.

## Commands

### Frontend (Next.js)
```bash
pnpm install     # Install dependencies
pnpm dev         # Start dev server at localhost:3000
pnpm build       # Production build
pnpm lint        # Run ESLint
```

### Python Library
```bash
cd entity-db-lib
uv sync                        # Install dependencies (default: search only)
uv sync --extra build          # Install with build/import extras
uv sync --extra serve          # Install with server extras
uv sync --all-extras           # Install everything
uv run pytest                  # Run tests
uv build                       # Build package
uv publish                     # Publish to PyPI (requires credentials)
```

### CLI Examples
```bash
corp-entity-db search "Microsoft"              # Search organizations (USearch HNSW)
corp-entity-db search-people "Tim Cook"        # Search people (composite embeddings)
corp-entity-db search-people "Tim Cook" --role CEO --org Apple  # Constrained composite search
corp-entity-db search-roles "CEO"              # Search roles
corp-entity-db search-locations "California"   # Search locations
corp-entity-db status                          # Show database statistics
corp-entity-db status --for-llm               # Output schema and enum tables for LLM docs

# Persistent server (keeps models warm, avoids ~30s startup per invocation)
corp-entity-db serve                           # Start on localhost:8222
corp-entity-db serve --port 9000               # Custom port

# Data import
corp-entity-db import-sec --download           # Bulk SEC data (73K filers)
corp-entity-db import-gleif --download         # GLEIF LEI records (2.6M)
corp-entity-db import-companies-house          # UK Companies House (~5M records)
corp-entity-db import-people --all             # Notable people from Wikidata
corp-entity-db import-sec-officers --start-year 2023 --limit 10000
corp-entity-db import-ch-officers --file officers.zip --limit 10000
corp-entity-db import-wikidata-dump --download --limit 50000

# Database management
corp-entity-db migrate                         # Migrate schema to latest (v5)
corp-entity-db canonicalize                    # Link equivalent records across sources
corp-entity-db post-import                     # Run after any import: build USearch indexes + VACUUM
corp-entity-db build-index                     # Build all USearch HNSW indexes
corp-entity-db build-index --identity-only     # Build only the people identity index
corp-entity-db download                        # Download lite version + USearch indexes (default)
corp-entity-db download --full                 # Download full version + USearch indexes
corp-entity-db upload                          # Upload with lite variant + USearch indexes
```

### RunPod Deployment
```bash
cd runpod
# Build for Linux/amd64 (required on Mac)
docker build --platform linux/amd64 -t corp-entity-db-runpod .
```

## Architecture

### Two Deployment Modes
The frontend can connect to the entity database via two backends (configured by environment variables):
1. **RunPod Serverless** (`RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`) - Production, pay-per-use GPU
2. **Local Server** (`corp-entity-db serve`) - Self-hosted FastAPI server on port 8222

### Directory Structure
- `entity-db-lib/` - Python library (PyPI package: `corp-entity-db`)
- `entity-db-lib/src/corp_entity_db/` - Core library code
- `entity-db-lib/src/corp_entity_db/commands/` - CLI commands (Click)
- `entity-db-lib/src/corp_entity_db/importers/` - Data importers (GLEIF, SEC, Wikidata, CH)
- `src/` - Next.js frontend (React 19, Tailwind CSS)
- `src/app/api/search/` - API route that proxies to search backends
- `runpod/` - Docker + handler for RunPod serverless deployment

### Database Classes
- `OrganizationDatabase` - Search and manage organizations
- `PersonDatabase` - Search and manage people (composite embedding search)
- `RolesDatabase` - Search roles/positions
- `LocationsDatabase` - Search locations
- `CompanyEmbedder` - Generate embeddings using google/embeddinggemma-300m (300M params), including composite person embeddings and identity index embeddings
- `OrganizationResolver` - Resolve organization names to canonical records
- `Canonicalizer` - Link equivalent records across data sources

### Database Schema
- Schema version: v5 with normalized FK references (INTEGER FKs replace TEXT enums), no embedding columns
- Variants: full (`entities-v4.db`), lite (`entities-v4-lite.db` - drops `record` and `name_normalized` columns)
- Default DB path: `~/.cache/corp-extractor/entities-v2.db` (symlinked to v4)
- USearch indexes: `people_usearch.bin`, `people_identity_usearch.bin`, `organizations_usearch.bin` (same dir as DB)
- **All embeddings** (orgs + people) exist **only** in USearch HNSW indexes, never in SQLite — generated on-the-fly during index building
- Two people indexes: primary composite (768-dim, name|role|org segments) and secondary identity (256-dim Matryoshka, `"{name}, a {type_label}"`)
- Tables: `organizations`, `people`, `roles`, `locations`, `location_types`, `db_info`

### Organization EntityType Classification

| EntityType | Description | Examples |
|------------|-------------|----------|
| `business` | Commercial companies | Apple Inc., Amazon |
| `fund` | Investment funds, ETFs | Vanguard S&P 500 ETF |
| `branch` | Branch offices | Deutsche Bank London |
| `nonprofit` | Non-profit organizations | Red Cross |
| `ngo` | Non-governmental orgs | Greenpeace |
| `foundation` | Charitable foundations | Gates Foundation |
| `government` | Government agencies | SEC, FDA |
| `international_org` | International organizations | UN, WHO, IMF |
| `educational` | Schools, universities | MIT, Stanford |
| `research` | Research institutes | CERN, NIH |
| `healthcare` | Hospitals, health orgs | Mayo Clinic |
| `media` | Studios, record labels | Warner Bros |
| `sports` | Sports clubs/teams | Manchester United |
| `political_party` | Political parties | Democratic Party |
| `religious` | Religious organizations | Vatican, churches |
| `trade_union` | Labor unions | AFL-CIO |
| `unknown` | Unclassified organizations | — |

### Person Types

| PersonType | Description | Examples |
|------------|-------------|----------|
| `executive` | CEOs, board members, C-suite, founders | Tim Cook, Jeff Bezos |
| `politician` | Elected officials | Joe Biden, Angela Merkel |
| `government` | Civil servants, diplomats | Ambassadors, agency heads |
| `military` | Military officers | Generals, admirals |
| `legal` | Judges, lawyers | Supreme Court justices |
| `professional` | Known for profession | Famous surgeons, architects |
| `athlete` | Sports figures | LeBron James, Lionel Messi |
| `artist` | Traditional creatives | Tom Hanks, Taylor Swift |
| `media` | Internet/social media personalities | YouTubers, influencers |
| `academic` | Professors, researchers, scientists | Neil deGrasse Tyson, Albert Einstein |
| `journalist` | Reporters, news presenters | Anderson Cooper |
| `activist` | Advocates, campaigners | Greta Thunberg |
| `unknown` | Unclassified people | — |

### Data Sources

| Importer | Description | Scale |
|----------|-------------|-------|
| GLEIF | Legal Entity Identifier records | ~2.6M records |
| SEC Edgar | US public company filers | ~73K filers |
| SEC Form 4 | Officers and directors filings | Variable |
| Companies House | UK registered companies | ~5.5M records |
| CH Officers | UK Companies House officers | Variable |
| Wikidata SPARQL | Organizations via SPARQL queries | Variable (may timeout) |
| Wikidata People | Notable people via SPARQL | Variable (may timeout) |
| Wikidata Dump | Full JSON dump import (recommended) | ~100GB, 3-thread parallel |

### Dependency Extras
The default install (`pip install corp-entity-db`) includes only search dependencies. Optional extras:
- `[build]` — orjson, indexed-bzip2 (for building/importing databases)
- `[serve]` — fastapi, uvicorn (for `corp-entity-db serve`)
- `[client]` — httpx (for `EntityDBClient` remote proxy)
- `[all]` — everything combined

### Key Technical Notes
- USearch `expansion_search=200` must be set after `Index.restore()` (default resets to 64)
- SQLite pragmas: 256MB mmap, 500MB page cache, WAL journal mode
- **All embeddings** (orgs + people): stored **only** in USearch HNSW indexes, NOT in SQLite — generated on-the-fly during index building
- **Primary people index** (`people_usearch.bin`): 768-dim composite embeddings (name|role|org as 3×256-dim segments). Name, role, and org are embedded separately, independently L2-normalized (Matryoshka truncation), weighted (name=8, role=1, org=4), and concatenated. This gives AND-style matching where a bad match on org cannot be compensated by a good match on role. Built by `build_people_composite_index()`.
- **Secondary identity index** (`people_identity_usearch.bin`): 256-dim Matryoshka-truncated embeddings of `format_person_query()` text (e.g. `"Taylor Swift, an artist"`, `"Tim Cook, a CEO of Apple"`). Used as fallback when primary composite scores are below threshold (0.75). Built by `build_people_identity_index()`.
- Lite database drops `record` and `name_normalized` columns — uses USearch HNSW indexes for ANN search
- Int8 quantization for USearch is computed on-the-fly during index build (not stored)
- Embedding model: `google/embeddinggemma-300m` (300M params)
- PyPI package: `corp-entity-db`
- CLI entry point: `corp-entity-db`
- Server default port: 8222
- Person records include `birth_date` and `death_date` fields, with `is_historic` property for deceased individuals
- Canonicalization priority: wikidata > sec_edgar > companies_house
- People search uses dual-index strategy: primary composite index searched first, identity index as fallback when scores < 0.75 threshold
- People search performance: 55.7% acc@1, 96.1% acc@20 on 280 queries across 12 person types, 300-400ms per query after model warmup (identity fallback improves accuracy for identity-defined types like artists, athletes, activists)

### Python Library API

```python
from corp_entity_db import OrganizationDatabase, CompanyEmbedder, get_database_path

# Search organizations
db = OrganizationDatabase(get_database_path())
matches = db.search("Microsoft", limit=10)
for match in matches:
    print(f"{match.record.name} ({match.record.entity_type}) - score: {match.score:.3f}")

# Search people (composite embeddings + identity fallback)
from corp_entity_db import PersonDatabase, get_person_database
from corp_entity_db.store import format_person_query
person_db = get_person_database()
embedder = CompanyEmbedder()

# Org-defined person: composite embedding is authoritative
query_emb = embedder.embed_composite_person("Tim Cook", role="CEO", org="Apple")
identity_emb = embedder.embed_for_identity_index(format_person_query("Tim Cook", person_type="executive"))
matches = person_db.search(query_emb, top_k=5, identity_query_embedding=identity_emb)

# Identity-defined person: composite scores low, identity fallback kicks in
query_emb = embedder.embed_composite_person("Taylor Swift")
identity_emb = embedder.embed_for_identity_index(format_person_query("Taylor Swift", person_type="artist"))
matches = person_db.search(query_emb, top_k=5, identity_query_embedding=identity_emb)

# Resolve organization names
from corp_entity_db import OrganizationResolver, get_organization_resolver
resolver = get_organization_resolver()
result = resolver.resolve("MSFT")
```
