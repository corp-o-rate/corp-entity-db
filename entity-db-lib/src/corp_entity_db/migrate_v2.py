"""
Migration script from v1 to v2 normalized schema.

Transforms TEXT-based enum storage to INTEGER FK references,
adds roles and locations tables, and converts QIDs to integers.

Usage:
    corp-extractor db migrate-v2 entities.db entities-v2.db
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pycountry
import sqlite_vec

from .schema_v2 import create_all_tables
from .seed_data import (
    LOCATION_TYPE_NAME_TO_ID,
    ORG_TYPE_NAME_TO_ID,
    PEOPLE_TYPE_NAME_TO_ID,
    SOURCE_NAME_TO_ID,
    seed_all_enums,
    seed_pycountry_locations,
)

logger = logging.getLogger(__name__)


def parse_qid(qid_text: Optional[str]) -> Optional[int]:
    """
    Parse a QID string to integer.

    Args:
        qid_text: QID string like "Q12345" or just "12345"

    Returns:
        Integer QID or None if invalid
    """
    if not qid_text:
        return None

    # Strip whitespace
    qid_text = qid_text.strip()

    # Handle "Q12345" format
    if qid_text.startswith("Q") or qid_text.startswith("q"):
        qid_text = qid_text[1:]

    try:
        return int(qid_text)
    except ValueError:
        return None


def normalize_name_for_lookup(name: str) -> str:
    """Normalize a name for database lookup."""
    if not name:
        return ""
    return name.lower().strip()


class DatabaseMigrator:
    """
    Migrates v1 entity database to v2 normalized schema.

    Handles:
    - Creating v2 schema with enum tables
    - Seeding enum lookup data
    - Importing pycountry countries
    - Migrating organizations with FK resolution
    - Migrating people with FK resolution
    - Converting QIDs from TEXT to INTEGER
    - Preserving embeddings
    """

    def __init__(
        self,
        source_path: str | Path,
        target_path: str | Path,
        embedding_dim: int = 768,
        resume: bool = False,
    ):
        """
        Initialize the migrator.

        Args:
            source_path: Path to v1 database
            target_path: Path for v2 database (will be created)
            embedding_dim: Embedding dimension (default 768)
            resume: If True, resume from last completed step
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.embedding_dim = embedding_dim
        self.resume = resume

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source database not found: {self.source_path}")

        if self.target_path.exists() and not resume:
            raise FileExistsError(f"Target database already exists: {self.target_path}. Use resume=True to continue.")

        # Caches for FK lookups during migration
        self._location_cache: dict[str, int] = {}  # name_normalized -> location_id
        self._role_cache: dict[str, int] = {}  # name_normalized -> role_id

    def migrate(self, batch_size: int = 10000) -> dict[str, int]:
        """
        Run the full migration.

        Args:
            batch_size: Number of records per batch commit

        Returns:
            Dict with migration statistics
        """
        if self.resume and self.target_path.exists():
            logger.info(f"Resuming migration from {self.source_path} to {self.target_path}")
        else:
            logger.info(f"Starting migration from {self.source_path} to {self.target_path}")

        # Open connections
        source_conn = sqlite3.connect(str(self.source_path))
        source_conn.row_factory = sqlite3.Row

        # Load sqlite-vec for source (needed to read embedding virtual tables)
        source_conn.enable_load_extension(True)
        sqlite_vec.load(source_conn)
        source_conn.enable_load_extension(False)

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        target_conn = sqlite3.connect(str(self.target_path))
        target_conn.row_factory = sqlite3.Row

        # Load sqlite-vec for target
        target_conn.enable_load_extension(True)
        sqlite_vec.load(target_conn)
        target_conn.enable_load_extension(False)

        try:
            stats = self._run_migration(source_conn, target_conn, batch_size)
        finally:
            source_conn.close()
            target_conn.close()

        logger.info(f"Migration complete: {stats}")
        return stats

    def _run_migration(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
        batch_size: int,
    ) -> dict[str, int]:
        """Run all migration steps."""
        stats: dict[str, int] = {}

        # Determine which step to start from
        start_step = 1
        if self.resume and self.target_path.exists():
            start_step = self._detect_completed_step(target)
            logger.info(f"Resuming from step {start_step}")

        # Step 1: Create v2 schema
        if start_step <= 1:
            logger.info("Step 1: Creating v2 schema...")
            create_all_tables(target, self.embedding_dim)
        else:
            logger.info("Step 1: Skipped (schema already exists)")

        # Step 2: Seed enum tables
        if start_step <= 2:
            logger.info("Step 2: Seeding enum tables...")
            enum_stats = seed_all_enums(target)
            stats.update({f"seed_{k}": v for k, v in enum_stats.items()})
        else:
            logger.info("Step 2: Skipped (enums already seeded)")

        # Step 3: Import pycountry countries into locations
        if start_step <= 3:
            logger.info("Step 3: Importing pycountry countries...")
            stats["locations_pycountry"] = seed_pycountry_locations(target)
        else:
            logger.info("Step 3: Skipped (pycountry already imported)")

        # Build location lookup cache from imported countries
        self._build_location_cache(target)

        # Step 4: Migrate qid_labels
        if start_step <= 4:
            logger.info("Step 4: Migrating qid_labels...")
            stats["qid_labels"] = self._migrate_qid_labels(source, target)
        else:
            logger.info("Step 4: Skipped (qid_labels already migrated)")

        # Step 5: Migrate organizations
        if start_step <= 5:
            logger.info("Step 5: Migrating organizations...")
            stats["organizations"] = self._migrate_organizations(source, target, batch_size)
        else:
            logger.info("Step 5: Skipped (organizations already migrated)")
            # Rebuild ID mapping for embedding migration
            self._rebuild_org_id_mapping(source, target)

        # Step 6: Migrate people
        if start_step <= 6:
            logger.info("Step 6: Migrating people...")
            stats["people"] = self._migrate_people(source, target, batch_size)
        else:
            logger.info("Step 6: Skipped (people already migrated)")
            # Rebuild ID mapping for embedding migration
            self._rebuild_person_id_mapping(source, target)

        # Step 7: Migrate organization embeddings
        if start_step <= 7:
            logger.info("Step 7: Migrating organization embeddings...")
            stats["org_embeddings"] = self._migrate_org_embeddings(source, target, batch_size)
        else:
            logger.info("Step 7: Skipped (organization embeddings already migrated)")

        # Step 8: Migrate person embeddings
        if start_step <= 8:
            logger.info("Step 8: Migrating person embeddings...")
            stats["person_embeddings"] = self._migrate_person_embeddings(source, target, batch_size)
        else:
            logger.info("Step 8: Skipped (person embeddings already migrated)")

        # Vacuum to optimize
        logger.info("Step 9: Optimizing database...")
        target.execute("VACUUM")

        return stats

    def _detect_completed_step(self, target: sqlite3.Connection) -> int:
        """
        Detect the first incomplete migration step.

        Returns:
            Step number to resume from (1-9)
        """
        # Check if organization_embeddings has data
        try:
            cursor = target.execute("SELECT COUNT(*) FROM organization_embeddings")
            if cursor.fetchone()[0] > 0:
                # Check person embeddings
                cursor = target.execute("SELECT COUNT(*) FROM person_embeddings")
                if cursor.fetchone()[0] > 0:
                    return 9  # All done, just vacuum
                return 8  # Person embeddings pending
            # Org embeddings empty, check if organizations exist
            cursor = target.execute("SELECT COUNT(*) FROM organizations")
            if cursor.fetchone()[0] > 0:
                return 7  # Org embeddings pending
        except sqlite3.OperationalError:
            pass

        # Check if organizations table has data
        try:
            cursor = target.execute("SELECT COUNT(*) FROM organizations")
            if cursor.fetchone()[0] > 0:
                # Check if people exist
                cursor = target.execute("SELECT COUNT(*) FROM people")
                if cursor.fetchone()[0] > 0:
                    return 7  # Ready for embeddings
                return 6  # People pending
        except sqlite3.OperationalError:
            pass

        # Check if qid_labels has data
        try:
            cursor = target.execute("SELECT COUNT(*) FROM qid_labels")
            if cursor.fetchone()[0] > 0:
                return 5  # Organizations pending
        except sqlite3.OperationalError:
            pass

        # Check if locations has data
        try:
            cursor = target.execute("SELECT COUNT(*) FROM locations")
            if cursor.fetchone()[0] > 0:
                return 4  # qid_labels pending
        except sqlite3.OperationalError:
            pass

        # Check if source_types has data
        try:
            cursor = target.execute("SELECT COUNT(*) FROM source_types")
            if cursor.fetchone()[0] > 0:
                return 3  # pycountry import pending
        except sqlite3.OperationalError:
            pass

        # Check if organizations table exists at all
        try:
            target.execute("SELECT 1 FROM organizations LIMIT 1")
            return 2  # Schema exists, enum seeding pending
        except sqlite3.OperationalError:
            pass

        return 1  # Start from beginning

    def _rebuild_org_id_mapping(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
    ) -> None:
        """Rebuild organization ID mapping for embedding migration when resuming."""
        logger.info("Rebuilding organization ID mapping...")

        self._org_id_mapping = {}

        # Get all source organizations with their IDs and source_ids
        source_cursor = source.execute(
            "SELECT id, source_id FROM organizations"
        )

        for row in source_cursor:
            old_id = row["id"]
            source_identifier = row["source_id"]

            if source_identifier:
                # Look up in target by source_identifier
                target_cursor = target.execute(
                    "SELECT id FROM organizations WHERE source_identifier = ?",
                    (source_identifier,)
                )
                target_row = target_cursor.fetchone()
                if target_row:
                    self._org_id_mapping[old_id] = target_row["id"]

        logger.info(f"Rebuilt mapping for {len(self._org_id_mapping)} organizations")

    def _rebuild_person_id_mapping(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
    ) -> None:
        """Rebuild person ID mapping for embedding migration when resuming."""
        logger.info("Rebuilding person ID mapping...")

        self._person_id_mapping = {}

        # Get all source people with their IDs and source_ids
        source_cursor = source.execute(
            "SELECT id, source_id FROM people"
        )

        for row in source_cursor:
            old_id = row["id"]
            source_identifier = row["source_id"]

            if source_identifier:
                # Look up in target by source_identifier
                target_cursor = target.execute(
                    "SELECT id FROM people WHERE source_identifier = ?",
                    (source_identifier,)
                )
                target_row = target_cursor.fetchone()
                if target_row:
                    self._person_id_mapping[old_id] = target_row["id"]

        logger.info(f"Rebuilt mapping for {len(self._person_id_mapping)} people")

    def _build_location_cache(self, conn: sqlite3.Connection) -> None:
        """Build location lookup cache from existing locations."""
        cursor = conn.execute("SELECT id, name_normalized, source_identifier FROM locations")
        for row in cursor:
            # Cache by normalized name
            self._location_cache[row["name_normalized"]] = row["id"]
            # Also cache by source_identifier (e.g., "US", "GB")
            if row["source_identifier"]:
                self._location_cache[row["source_identifier"].lower()] = row["id"]

    def _resolve_region_to_location(
        self,
        conn: sqlite3.Connection,
        region: str,
    ) -> Optional[int]:
        """
        Resolve a region string to a location ID.

        Args:
            conn: Target database connection
            region: Region string (country code, name, or QID)

        Returns:
            Location ID or None if not resolved
        """
        if not region:
            return None

        # Check cache first
        region_lower = region.lower().strip()
        if region_lower in self._location_cache:
            return self._location_cache[region_lower]

        # Try to resolve via pycountry
        location_id = self._resolve_via_pycountry(conn, region)
        if location_id:
            self._location_cache[region_lower] = location_id
            return location_id

        # If it's a QID, try to look up or create
        if region.startswith("Q") and region[1:].isdigit():
            # We can't resolve QIDs to locations without more data
            # Return None for now
            return None

        return None

    def _resolve_via_pycountry(
        self,
        conn: sqlite3.Connection,
        region: str,
    ) -> Optional[int]:
        """Try to resolve region via pycountry."""
        region_clean = region.strip()
        if not region_clean:
            return None

        alpha_2 = None

        # Try as 2-letter code
        if len(region_clean) == 2:
            country = pycountry.countries.get(alpha_2=region_clean.upper())
            if country:
                alpha_2 = country.alpha_2

        # Try as 3-letter code
        if not alpha_2 and len(region_clean) == 3:
            country = pycountry.countries.get(alpha_3=region_clean.upper())
            if country:
                alpha_2 = country.alpha_2

        # Try fuzzy search
        if not alpha_2:
            try:
                matches = pycountry.countries.search_fuzzy(region_clean)
                if matches:
                    alpha_2 = matches[0].alpha_2
            except LookupError:
                pass

        # Look up by alpha_2 in cache
        if alpha_2:
            return self._location_cache.get(alpha_2.lower())

        return None

    def _get_or_create_role(
        self,
        conn: sqlite3.Connection,
        role_name: str,
        source_id: int = 4,  # wikidata
    ) -> int:
        """
        Get or create a role record.

        Args:
            conn: Target database connection
            role_name: Role/title name
            source_id: Source type ID (default: wikidata)

        Returns:
            Role ID
        """
        if not role_name:
            raise ValueError("Role name cannot be empty")

        name_normalized = normalize_name_for_lookup(role_name)

        # Check cache
        if name_normalized in self._role_cache:
            return self._role_cache[name_normalized]

        # Check database
        cursor = conn.execute(
            "SELECT id FROM roles WHERE name_normalized = ? AND source_id = ?",
            (name_normalized, source_id)
        )
        row = cursor.fetchone()
        if row:
            self._role_cache[name_normalized] = row["id"]
            return row["id"]

        # Create new role
        cursor = conn.execute(
            """
            INSERT INTO roles (name, name_normalized, source_id, record)
            VALUES (?, ?, ?, '{}')
            """,
            (role_name, name_normalized, source_id)
        )
        role_id = cursor.lastrowid
        assert role_id is not None
        conn.commit()

        self._role_cache[name_normalized] = role_id
        return role_id

    def _migrate_qid_labels(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
    ) -> int:
        """Migrate qid_labels table, converting TEXT QIDs to INTEGER."""
        # Check if source has qid_labels table
        cursor = source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='qid_labels'"
        )
        if not cursor.fetchone():
            logger.info("No qid_labels table in source, skipping")
            return 0

        cursor = source.execute("SELECT qid, label FROM qid_labels")
        count = 0

        for row in cursor:
            qid_int = parse_qid(row["qid"])
            if qid_int is not None:
                target.execute(
                    "INSERT OR IGNORE INTO qid_labels (qid, label) VALUES (?, ?)",
                    (qid_int, row["label"])
                )
                count += 1

        target.commit()
        logger.info(f"Migrated {count} QID labels")
        return count

    def _migrate_organizations(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
        batch_size: int,
    ) -> int:
        """Migrate organizations table with FK conversion."""
        # Check source schema
        cursor = source.execute("PRAGMA table_info(organizations)")
        columns = {row["name"] for row in cursor}

        if "source_id" in columns and "source" not in columns:
            logger.info("Source appears to already be v2 schema")
            return 0

        cursor = source.execute("""
            SELECT id, name, name_normalized, source, source_id, region,
                   entity_type, from_date, to_date, record, canon_id, canon_size
            FROM organizations
        """)

        count = 0
        id_mapping: dict[int, int] = {}  # old_id -> new_id

        for row in cursor:
            # Convert source to source_id FK
            source_name = row["source"]
            # Map "wikipedia" to "wikidata"
            if source_name == "wikipedia":
                source_name = "wikidata"
            source_type_id = SOURCE_NAME_TO_ID.get(source_name, 4)  # default to wikidata

            # Convert entity_type to entity_type_id FK
            entity_type_name = row["entity_type"] or "unknown"
            entity_type_id = ORG_TYPE_NAME_TO_ID.get(entity_type_name, 17)  # default to unknown

            # Resolve region to location_id
            region_id = self._resolve_region_to_location(target, row["region"] or "")

            # Extract QID from source_id if it's a Q code
            qid = None
            old_source_id = row["source_id"]
            if old_source_id and old_source_id.startswith("Q"):
                qid = parse_qid(old_source_id)

            # Insert into target (use OR IGNORE to handle region normalization duplicates)
            cursor2 = target.execute(
                """
                INSERT OR IGNORE INTO organizations
                (qid, name, name_normalized, source_id, source_identifier, region_id,
                 entity_type_id, from_date, to_date, record, canon_id, canon_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    qid,
                    row["name"],
                    row["name_normalized"],
                    source_type_id,
                    old_source_id,  # Keep original source_id as source_identifier
                    region_id,
                    entity_type_id,
                    row["from_date"] or None,
                    row["to_date"] or None,
                    row["record"],
                    None,  # Reset canon_id for fresh canonicalization
                    1,     # Reset canon_size
                )
            )

            new_id = cursor2.lastrowid
            if new_id and cursor2.rowcount > 0:
                id_mapping[row["id"]] = new_id
                count += 1
            else:
                # Duplicate - look up the existing record's ID for embedding mapping
                existing = target.execute(
                    "SELECT id FROM organizations WHERE source_identifier = ? AND source_id = ?",
                    (old_source_id, source_type_id)
                ).fetchone()
                if existing:
                    id_mapping[row["id"]] = existing["id"]

            if count % batch_size == 0:
                target.commit()
                logger.info(f"  Migrated {count} organizations...")

        target.commit()
        logger.info(f"Migrated {count} organizations")

        # Store ID mapping for embedding migration
        self._org_id_mapping = id_mapping
        return count

    def _migrate_people(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
        batch_size: int,
    ) -> int:
        """Migrate people table with FK conversion."""
        # Check if source has people table
        cursor = source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
        )
        if not cursor.fetchone():
            logger.info("No people table in source, skipping")
            return 0

        # Check source schema
        cursor = source.execute("PRAGMA table_info(people)")
        columns = {row["name"] for row in cursor}

        if "source_id" in columns and "source" not in columns:
            logger.info("People table appears to already be v2 schema")
            return 0

        cursor = source.execute("""
            SELECT id, name, name_normalized, source, source_id, country,
                   person_type, known_for_role, known_for_org, known_for_org_id,
                   from_date, to_date, birth_date, death_date, record,
                   canon_id, canon_size
            FROM people
        """)

        count = 0
        id_mapping: dict[int, int] = {}  # old_id -> new_id

        for row in cursor:
            # Convert source to source_id FK
            source_name = row["source"] or "wikidata"
            source_type_id = SOURCE_NAME_TO_ID.get(source_name, 4)

            # Convert person_type to person_type_id FK
            person_type_name = row["person_type"] or "unknown"
            person_type_id = PEOPLE_TYPE_NAME_TO_ID.get(person_type_name, 15)

            # Resolve country to location_id
            country_id = self._resolve_region_to_location(target, row["country"] or "")

            # Get or create role_id for known_for_role
            role_id = None
            if row["known_for_role"]:
                role_id = self._get_or_create_role(target, row["known_for_role"], source_type_id)

            # Map known_for_org_id to new org ID
            old_org_id = row["known_for_org_id"]
            new_org_id = None
            if old_org_id and hasattr(self, "_org_id_mapping"):
                new_org_id = self._org_id_mapping.get(old_org_id)

            # Extract QID from source_id if it's a Q code
            qid = None
            old_source_id = row["source_id"]
            if old_source_id and old_source_id.startswith("Q"):
                qid = parse_qid(old_source_id)

            # Insert into target (use OR IGNORE to handle duplicates from normalization)
            cursor2 = target.execute(
                """
                INSERT OR IGNORE INTO people
                (qid, name, name_normalized, source_id, source_identifier, country_id,
                 person_type_id, known_for_role_id, known_for_org, known_for_org_id,
                 from_date, to_date, birth_date, death_date, record, canon_id, canon_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    qid,
                    row["name"],
                    row["name_normalized"],
                    source_type_id,
                    old_source_id,
                    country_id,
                    person_type_id,
                    role_id,
                    row["known_for_org"] or "",
                    new_org_id,
                    row["from_date"] or None,
                    row["to_date"] or None,
                    row["birth_date"] or None,
                    row["death_date"] or None,
                    row["record"],
                    None,  # Reset canon_id
                    1,     # Reset canon_size
                )
            )

            new_id = cursor2.lastrowid
            if new_id and cursor2.rowcount > 0:
                id_mapping[row["id"]] = new_id
                count += 1
            else:
                # Duplicate - look up existing record for embedding mapping
                existing = target.execute(
                    """SELECT id FROM people
                       WHERE source_identifier = ? AND source_id = ?
                       AND known_for_role_id IS ? AND known_for_org_id IS ?""",
                    (old_source_id, source_type_id, role_id, new_org_id)
                ).fetchone()
                if existing:
                    id_mapping[row["id"]] = existing["id"]

            if count % batch_size == 0:
                target.commit()
                logger.info(f"  Migrated {count} people...")

        target.commit()
        logger.info(f"Migrated {count} people")

        # Store ID mapping for embedding migration
        self._person_id_mapping = id_mapping
        return count

    def _migrate_org_embeddings(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
        batch_size: int,
    ) -> int:
        """Migrate organization embeddings using ID mapping."""
        # Check if source has embeddings table
        cursor = source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='organization_embeddings'"
        )
        if not cursor.fetchone():
            logger.info("No organization_embeddings table in source, skipping")
            return 0

        if not hasattr(self, "_org_id_mapping"):
            logger.warning("No org ID mapping available, skipping embedding migration")
            return 0

        cursor = source.execute("SELECT org_id, embedding FROM organization_embeddings")
        count = 0

        for row in cursor:
            old_id = row["org_id"]
            new_id = self._org_id_mapping.get(old_id)

            if new_id is not None:
                target.execute(
                    "INSERT OR REPLACE INTO organization_embeddings (org_id, embedding) VALUES (?, ?)",
                    (new_id, row["embedding"])
                )
                count += 1

                if count % batch_size == 0:
                    target.commit()
                    logger.info(f"  Migrated {count} organization embeddings...")

        target.commit()
        logger.info(f"Migrated {count} organization embeddings")
        return count

    def _migrate_person_embeddings(
        self,
        source: sqlite3.Connection,
        target: sqlite3.Connection,
        batch_size: int,
    ) -> int:
        """Migrate person embeddings using ID mapping."""
        # Check if source has embeddings table
        cursor = source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='person_embeddings'"
        )
        if not cursor.fetchone():
            logger.info("No person_embeddings table in source, skipping")
            return 0

        if not hasattr(self, "_person_id_mapping"):
            logger.warning("No person ID mapping available, skipping embedding migration")
            return 0

        cursor = source.execute("SELECT person_id, embedding FROM person_embeddings")
        count = 0

        for row in cursor:
            old_id = row["person_id"]
            new_id = self._person_id_mapping.get(old_id)

            if new_id is not None:
                target.execute(
                    "INSERT OR REPLACE INTO person_embeddings (person_id, embedding) VALUES (?, ?)",
                    (new_id, row["embedding"])
                )
                count += 1

                if count % batch_size == 0:
                    target.commit()
                    logger.info(f"  Migrated {count} person embeddings...")

        target.commit()
        logger.info(f"Migrated {count} person embeddings")
        return count


def migrate_database(
    source_path: str | Path,
    target_path: str | Path,
    embedding_dim: int = 768,
    batch_size: int = 10000,
    resume: bool = False,
) -> dict[str, int]:
    """
    Migrate a v1 database to v2 normalized schema.

    Args:
        source_path: Path to v1 database
        target_path: Path for v2 database (will be created)
        embedding_dim: Embedding dimension
        batch_size: Batch size for commits
        resume: If True, resume from last completed step

    Returns:
        Migration statistics
    """
    migrator = DatabaseMigrator(source_path, target_path, embedding_dim, resume=resume)
    return migrator.migrate(batch_size)
