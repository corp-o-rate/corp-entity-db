---
description: Commit changes with tests and documentation
allowed-tools: Bash(git:*), Bash(uv:*), Bash(pnpm:*), Read, Edit
---

In the following order please:

1. **Run Tests**
   - For Python library: `cd entity-db-lib && uv run pytest`
   - For frontend: `pnpm build && pnpm lint`

2. **Generate Documentation**
   - If any new files were created, ensure they have appropriate docstrings
   - Update README.md files if API changes were made

3. **Stage Changes**
   - Review changed files: !`git diff --stat HEAD`
   - Stage relevant changes with `git add`

4. **Commit Changes**
   - Commit with a descriptive message explaining WHY the changes were made
   - Include any issue references if applicable

5. **Fix Any Issues**
   - If tests fail, fix the issues and re-commit
   - If pre-commit hooks fail, address the issues

6. **Push to GitHub**
   - Push changes to the remote repository

Update your TODOs and start.