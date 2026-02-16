---
description: Update all documentation files with recent changes
allowed-tools: Read, Edit, Write, Glob, Grep
---

Please update all the docs with all the changes.

## Files to update

1. **Root documentation:**
   - `CLAUDE.md` - Claude Code guidance
   - `README.md` - Main project README

2. **Python library documentation:**
   - `entity-db-lib/README.md` - Library README

3. **RunPod Documentation:**
   - `runpod/README.md`

## Specific periodic updates

### Database Changes

The embedding database documentation names approximate sizes and other info. Please check the file size of the database in HuggingFace (https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references/tree/main) and then run:

`uv run corp-entity-db status --for-llm`

To get the current sizes and other database information.

## Process

$ARGUMENTS$

1. First, review recent code changes to understand what needs documenting.
2. Search for any inconsistencies between code and documentation
3. Update all relevant documentation files
4. Ensure version numbers, feature descriptions, and examples are accurate
5. Verify database classes, importers, and API signatures match the code
