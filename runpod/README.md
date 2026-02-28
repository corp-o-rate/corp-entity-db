# RunPod Serverless Deployment

Deploys the corp-entity-db search API as a RunPod serverless endpoint. The Docker image bundles the lite database, USearch indexes, and embedding model for CPU-only inference.

## Building

Requires Docker and a HuggingFace token (for gated model access):

```bash
export HF_TOKEN=your_token_here
./build.sh <version>     # e.g. ./build.sh 1
```

This builds a `linux/amd64` image, tags it as `neilellis/corp-entity-db-runpod:v<version>`, and pushes to Docker Hub.

## What's in the image

- Python 3.11 slim + PyTorch CPU
- `corp-entity-db[serve]` library
- Pre-downloaded lite database + USearch HNSW indexes from HuggingFace
- Pre-downloaded `google/embeddinggemma-300m` embedding model

## API

The handler accepts a JSON payload:

```json
{
  "input": {
    "query": "Microsoft",
    "type": "org",
    "limit": 10,
    "hybrid": false
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search query |
| `type` | string | `"org"` | `"org"`, `"person"`, `"role"`, or `"location"` |
| `limit` | int | `10` | Max results |
| `hybrid` | bool | `false` | Text + embedding hybrid search (org/person only) |
| `role` | string | `null` | Role/job title for composite person search (person type only) |
| `org` | string | `null` | Organization for composite person search (person type only) |
| `person_type` | string | `null` | Person type hint for identity fallback (e.g. `"artist"`, `"athlete"`) |

Person search uses a dual-index strategy: a primary composite index (name + role + org as 768-dim vector for AND-style matching) and a secondary identity index (256-dim Matryoshka embeddings of natural language descriptions). The identity index is consulted as fallback when composite scores are below threshold, improving accuracy for people known by name and type alone (artists, athletes, etc.).

Results are cached in-memory (up to 1GB LRU cache) for repeated queries.
