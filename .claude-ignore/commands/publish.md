---
description: Publish the Python library to PyPI
allowed-tools: Bash(uv:*), Bash(git:*), Read, Edit
---

Publish the corp-entity-db library to PyPI:

**Pre-publish Checklist:**

1. **Version Check:**
   - Read `entity-db-lib/pyproject.toml`
   - Verify version number is incremented appropriately. Check PyPi - https://pypi.org/project/corp-entity-db/ - use semantic versioning: bump the middle number for non-minor functional changes and the final number for non-functional changes (fixes, tidy up etc.)

Make sure `entity-db-lib/src/corp_entity_db/__init__.py` has the correct same version number.

2. **Run Tests:**
   ```bash
   cd entity-db-lib && uv run pytest
   ```

3. **Build Package:**
   ```bash
   cd entity-db-lib && uv build
   ```

4. **Verify Build:**
   - Check dist/ directory for wheel and tarball
   - Verify package contents look correct

5. **Check Documentation:**

   Make sure these files are accurate and up to date.
   - `entity-db-lib/README.md` - Library README
   - `CLAUDE.md` - Claude Code Guidance

   Update the `runpod/Dockerfile` with the new version number

6. **Publish:**
   ```bash
   cd entity-db-lib && uv publish
   ```

   Then commit and push to Github.

7. **Post-publish:**
   - Tag the release in git: `git tag v{version}`
   - Push tags: `git push --tags`

**Note:** Requires PyPI credentials configured in environment.