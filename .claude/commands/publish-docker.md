---
description: Build and publish the RunPod Docker image
allowed-tools: Bash(cd:*), Bash(./build.sh:*), Bash(echo:*), Bash(git:*), Read, Edit
---

Build and publish the corp-entity-db RunPod Docker image:

**Pre-build Checklist:**

1. **Docker Image Version:**
   - Check the latest image version on Docker Hub: `docker inspect --format='{{index .RepoTags 0}}' neilellis/corp-entity-db-runpod:latest 2>/dev/null` or list tags via `curl -s https://hub.docker.com/v2/repositories/neilellis/corp-entity-db-runpod/tags/?page_size=5 | python3 -m json.tool` to find the highest existing `v*` tag.
   - Increment that version number by 1 for the new build. If no tags exist yet, start at 1.

2. **Requirements Check:**
   - Read `entity-db-lib/pyproject.toml` to get the current library version.
   - Read `runpod/requirements.txt` and verify the `corp-entity-db` version pin is correct.
   - If the version pin is outdated, update `requirements.txt` before building.

3. **Review Dockerfile:**
   - Read `runpod/Dockerfile` and `runpod/handler.py` to confirm they look correct.

4. **Check HF_TOKEN:**
   - If `HF_TOKEN` is not set in the environment, stop and ask the user for it.

5. **Build and Push:**
   Run the build script from the `runpod/` directory, passing the version number:
   ```bash
   cd runpod && ./build.sh {version}
   ```
   This builds for linux/amd64, tags as `neilellis/corp-entity-db-runpod:v{version}`, and pushes to Docker Hub.

6. **Post-publish:**
   - Commit any changes to `runpod/requirements.txt` if it was updated, and push to Github.

**Note:** Requires Docker Hub credentials (`docker login`) and `HF_TOKEN` environment variable for gated model access.
