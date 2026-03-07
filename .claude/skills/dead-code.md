# Dead Code Detection and Removal

Detect and remove unused Python code in the entity-db-lib using `deadcode`.

## Trigger
Use when the user asks to find dead code, unused code, redundant code, or clean up the codebase.

## Tool: deadcode

We use [deadcode](https://github.com/albertas/deadcode) (not vulture) — it uses AST parsing, supports `--fix`, and has TOML configuration. Known false positives are configured in `pyproject.toml` under `[tool.deadcode]`.

## Running

```bash
cd /Users/neil/IdeaProjects/corp-entity-db/entity-db-lib
uv run --with deadcode deadcode src/corp_entity_db/ 2>&1
```

## IMPORTANT: Recursive Removal

Dead code removal must be done recursively. Removing one piece of dead code can expose new dead code (e.g. a helper function that was only called by the function you just removed). After each removal pass:

1. Run deadcode again
2. Check for newly exposed dead code
3. Remove and repeat until no new dead code appears
4. Run tests after the final pass

## Known False Positives (DO NOT REMOVE)

These are configured as `ignore_names` in `pyproject.toml [tool.deadcode]`. If you encounter new false positives, add them there.

Categories:
- **Enum members** — accessed dynamically (e.g. `DataSource.GLEIF`)
- **Pydantic model fields/methods** — `model_dump_for_db()`, `is_historic`, `canonical_name`, etc.
- **FastAPI route handlers** — registered via decorators in `server.py`
- **Public API** — `EntityDBClient`, `resolve_with_candidates`
- **SQLite** — `row_factory` assignments
- **NamedTuple fields** — `embedding_texts`
- **Iterator unpacking** — `event` in XML parsing

## Triage Process

When reviewing deadcode output:

1. **Run deadcode** with the command above
2. **Filter out known false positives** (should be suppressed by pyproject.toml config)
3. **For each remaining item, verify** it's truly unused:
   - Search for references: `grep -r "function_name" src/ tests/`
   - Check `__init__.py` exports
   - Check if it's a Click CLI command, FastAPI route, or Pydantic method
   - Check if tests import it (test-only code may still be worth keeping if tests are valuable)
4. **Remove confirmed dead code** — prefer deleting entire functions/methods over commenting out
5. **Run deadcode again** — repeat steps 1-4 until clean (recursive!)
6. **Run tests** after final pass: `cd entity-db-lib && uv run pytest`
7. **If an entire module becomes empty**, delete the file and remove imports
8. **If new false positives appear**, add them to `pyproject.toml [tool.deadcode] ignore_names`

## Previously Confirmed Dead Code (removed 2026-03-02)

The following were confirmed dead and removed in the initial cleanup:

- `import_utils.py` — entire file (13 unused utility functions)
- `store.py`: `format_person_query_variants`, `close_shared_connection`, `_ensure_dir` (x2), `get_canon_stats`, `migrate_name_normalized`, `delete_source`, `iter_all_for_embedding`, `get_people_count`, `_location_type_qid_cache`
- `store.py` (LocationsDatabase): `get_or_create_by_qid`, `get_by_id`, `_get_location_type_name`, `get_location_type_id`, `get_location_type_id_from_qid`, `get_simplified_type`
- `embeddings.py`: `embed_and_quantize`, `embed_batch_and_quantize`
- `seed_data.py`: `SIMPLIFIED_LOCATION_TYPE_NAME_TO_ID`, `LOCATION_TYPE_ID_TO_NAME`, `LOCATION_TYPE_QID_TO_ID`, `get_pycountry_countries`, `seed_pycountry_locations`
- `hub.py`: `check_for_updates`, `get_latest_version`
- `wikidata_dump.py` importer: `import_organizations`, `get_discovered_organizations`, `clear_discovered_organizations`
- `wikidata_people.py`: `search_person`, `get_discovered_organizations`, `clear_discovered_organizations`
- `wikidata.py`: `search_company`
- `sec_form4.py`: `get_available_quarters`
- `companies_house.py`: `get_company`, `CH_COMPANY_URL`
- Dead imports removed from `store.py`: `ORG_TYPE_ID_TO_NAME`, `PEOPLE_TYPE_ID_TO_NAME`, `LOCATION_TYPE_QID_TO_ID`
- Various unused variables in commands/ (prefixed with `_` or removed)
