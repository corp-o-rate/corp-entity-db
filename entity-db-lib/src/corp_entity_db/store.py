"""
Entity/Organization database with USearch for vector search.

Uses a hybrid approach:
1. Text-based filtering to narrow candidates (Levenshtein-like)
2. USearch HNSW index for fast approximate nearest neighbor search
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pycountry

try:
    import sqlite_vec
    _has_sqlite_vec = True
except ImportError:
    _has_sqlite_vec = False

from .models import (
    CompanyRecord,
    DatabaseStats,
    EntityType,
    LocationRecord,
    PersonRecord,
    PersonType,
    RoleRecord,
    SimplifiedLocationType,
)
from .seed_data import (
    LOCATION_TYPE_NAME_TO_ID,
    LOCATION_TYPE_QID_TO_ID,
    LOCATION_TYPE_TO_SIMPLIFIED,
    ORG_TYPE_ID_TO_NAME,
    ORG_TYPE_NAME_TO_ID,
    PEOPLE_TYPE_ID_TO_NAME,
    PEOPLE_TYPE_NAME_TO_ID,
    SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME,
    SOURCE_ID_TO_NAME,
    SOURCE_NAME_TO_ID,
    seed_all_enums,
    seed_pycountry_locations,
)

logger = logging.getLogger(__name__)

from .hub import DB_VERSION, DEFAULT_DB_FULL_FILENAME

# Default database location
DEFAULT_DB_PATH = Path.home() / ".cache" / "corp-extractor" / DEFAULT_DB_FULL_FILENAME

# Module-level shared connections by path (both databases share the same connection)
_shared_connections: dict[str, sqlite3.Connection] = {}

# Module-level shared read-only connections
_shared_readonly_connections: dict[str, sqlite3.Connection] = {}

# Module-level singleton for OrganizationDatabase to prevent multiple loads
_database_instances: dict[str, "OrganizationDatabase"] = {}

# Module-level singleton for PersonDatabase
_person_database_instances: dict[str, "PersonDatabase"] = {}


def _apply_pragmas(conn: sqlite3.Connection, readonly: bool) -> None:
    """Apply performance PRAGMAs to a SQLite connection.

    Uses 256MB mmap, in-memory temp store, and 500MB page cache.
    USearch loads its index natively into memory, so we only need moderate
    SQLite cache for record lookups and text filtering.
    """
    cache_kb = 500 * 1024  # 500 MB

    conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB memory-mapped I/O
    conn.execute(f"PRAGMA cache_size = -{cache_kb}")
    conn.execute("PRAGMA temp_store = MEMORY")
    logger.debug(f"Applied PRAGMAs: mmap_size=256MB, cache_size={cache_kb // 1024}MB, temp_store=MEMORY")

    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
        logger.debug("Enabled WAL journal mode")


def _get_shared_connection(
    db_path: Path, embedding_dim: int = 768, readonly: bool = False
) -> sqlite3.Connection:
    """Get or create a shared database connection for the given path."""
    path_key = str(db_path)

    # Use separate pools for read-only vs read-write connections
    if readonly:
        if path_key not in _shared_readonly_connections:
            # Open in immutable mode for read-only access (avoids locking)
            conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
            conn.row_factory = sqlite3.Row

            # Load sqlite-vec extension if available (only needed for build operations)
            if _has_sqlite_vec:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

            _apply_pragmas(conn, readonly=True)

            _shared_readonly_connections[path_key] = conn
            logger.debug(f"Created shared read-only database connection for {path_key}")

        return _shared_readonly_connections[path_key]

    if path_key not in _shared_connections:
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension if available (only needed for build operations)
        if _has_sqlite_vec:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        _apply_pragmas(conn, readonly=False)

        _shared_connections[path_key] = conn
        logger.debug(f"Created shared database connection for {path_key}")

    return _shared_connections[path_key]


def close_shared_connection(db_path: Optional[Path] = None) -> None:
    """Close a shared database connection."""
    path_key = str(db_path or DEFAULT_DB_PATH)
    if path_key in _shared_connections:
        _shared_connections[path_key].close()
        del _shared_connections[path_key]
        logger.debug(f"Closed shared database connection for {path_key}")

# Comprehensive set of corporate legal suffixes (international)
COMPANY_SUFFIXES: set[str] = {
    'A/S', 'AB', 'AG', 'AO', 'AG & Co', 'AG &', 'AG & CO.', 'AG & CO. KG', 'AG & CO. KGaA',
    'AG & KG', 'AG & KGaA', 'AG & PARTNER', 'ATE', 'ASA', 'B.V.', 'BV', 'Class A', 'Class B',
    'Class C', 'Class D', 'Class E', 'Class F', 'Class G', 'CO', 'Co', 'Co.', 'Company',
    'Corp', 'Corp.', 'Corporation', 'DAC', 'GmbH', 'Inc', 'Inc.', 'Incorporated', 'KGaA',
    'Limited', 'LLC', 'LLP', 'LP', 'Ltd', 'Ltd.', 'N.V.', 'NV', 'Plc', 'PC', 'plc', 'PLC',
    'Pty Ltd', 'Pty', 'Pty. Ltd.', 'S.A.', 'S.A.B. de C.V.', 'SAB de CV', 'S.A.B.', 'S.A.P.I.',
    'NV/SA', 'SDI', 'SpA', 'S.L.', 'S.p.A.', 'SA', 'SE', 'Tbk PT', 'U.A.',
    # Additional common suffixes
    'Group', 'Holdings', 'Holding', 'Partners', 'Trust', 'Fund', 'Bank', 'N.A.', 'The',
}

# Source priority for organization canonicalization (lower = higher priority)
SOURCE_PRIORITY: dict[str, int] = {
    "gleif": 1,       # Gold standard LEI - globally unique legal entity identifier
    "sec_edgar": 2,   # Vetted US filers with CIK + ticker
    "companies_house": 3,  # Official UK registry
    "wikipedia": 4,   # Crowdsourced, less authoritative
}

# Source priority for people canonicalization (lower = higher priority)
PERSON_SOURCE_PRIORITY: dict[str, int] = {
    "wikidata": 1,       # Curated, has rich biographical data and Q codes
    "sec_edgar": 2,      # Vetted US filers (Form 4 officers/directors)
    "companies_house": 3,  # UK company officers
}

# Suffix expansions for canonical name matching
SUFFIX_EXPANSIONS: dict[str, str] = {
    " ltd": " limited",
    " corp": " corporation",
    " inc": " incorporated",
    " co": " company",
    " intl": " international",
    " natl": " national",
}


class UnionFind:
    """Simple Union-Find (Disjoint Set Union) data structure for canonicalization."""

    def __init__(self, elements: list[int]):
        """Initialize with list of element IDs."""
        self.parent: dict[int, int] = {e: e for e in elements}
        self.rank: dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return dict of root -> list of members."""
        result: dict[int, list[int]] = {}
        for e in self.parent:
            root = self.find(e)
            result.setdefault(root, []).append(e)
        return result


# Common region aliases not handled well by pycountry fuzzy search
REGION_ALIASES: dict[str, str] = {
    "uk": "GB",
    "u.k.": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "usa": "US",
    "u.s.a.": "US",
    "u.s.": "US",
    "united states of america": "US",
    "america": "US",
}

# Cache for region normalization lookups
_region_cache: dict[str, str] = {}


def _normalize_region(region: str) -> str:
    """
    Normalize a region string to ISO 3166-1 alpha-2 country code.

    Handles:
    - Country codes (2-letter, 3-letter)
    - Country names (with fuzzy matching)
    - US state codes (CA, NY) -> US
    - US state names (California, New York) -> US
    - Common aliases (UK, USA, England) -> proper codes

    Returns empty string if region cannot be normalized.
    """
    if not region:
        return ""

    # Check cache first
    cache_key = region.lower().strip()
    if cache_key in _region_cache:
        return _region_cache[cache_key]

    result = _normalize_region_uncached(region)
    _region_cache[cache_key] = result
    return result


def _normalize_region_uncached(region: str) -> str:
    """Uncached region normalization logic."""
    region_clean = region.strip()

    # Empty after stripping = empty result
    if not region_clean:
        return ""

    region_lower = region_clean.lower()
    region_upper = region_clean.upper()

    # Check common aliases first
    if region_lower in REGION_ALIASES:
        return REGION_ALIASES[region_lower]

    # For 2-letter codes, check country first, then US state
    # This means ambiguous codes like "CA" (Canada vs California) prefer country
    # But unambiguous codes like "NY" (not a country) will match as US state
    if len(region_clean) == 2:
        # Try as country alpha-2 first
        country = pycountry.countries.get(alpha_2=region_upper)
        if country:
            return country.alpha_2

        # If not a country, try as US state code
        subdivision = pycountry.subdivisions.get(code=f"US-{region_upper}")
        if subdivision:
            return "US"

    # Try alpha-3 lookup
    if len(region_clean) == 3:
        country = pycountry.countries.get(alpha_3=region_upper)
        if country:
            return country.alpha_2

    # Try as US state name (e.g., "California", "New York")
    try:
        subdivisions = list(pycountry.subdivisions.search_fuzzy(region_clean))
        if subdivisions:
            # Check if it's a US state
            if subdivisions[0].code.startswith("US-"):
                return "US"
            # Return the parent country code
            return subdivisions[0].country_code
    except LookupError:
        pass

    # Try country fuzzy search
    try:
        countries = pycountry.countries.search_fuzzy(region_clean)
        if countries:
            return countries[0].alpha_2
    except LookupError:
        pass

    # Return empty if we can't normalize
    return ""


def _regions_match(region1: str, region2: str) -> bool:
    """
    Check if two regions match after normalization.

    Empty regions match anything (lenient matching for incomplete data).
    """
    norm1 = _normalize_region(region1)
    norm2 = _normalize_region(region2)

    # Empty regions match anything
    if not norm1 or not norm2:
        return True

    return norm1 == norm2


def _normalize_for_canon(name: str) -> str:
    """Normalize name for canonical matching (simpler than search normalization)."""
    # Lowercase
    result = name.lower()
    # Remove trailing dots
    result = result.rstrip(".")
    # Remove extra whitespace
    result = " ".join(result.split())
    return result


def _expand_suffix(name: str) -> str:
    """Expand known suffix abbreviations."""
    result = name.lower().rstrip(".")
    for abbrev, full in SUFFIX_EXPANSIONS.items():
        if result.endswith(abbrev):
            result = result[:-len(abbrev)] + full
            break  # Only expand one suffix
    return result


def _names_match_for_canon(name1: str, name2: str) -> bool:
    """Check if two names match for canonicalization."""
    n1 = _normalize_for_canon(name1)
    n2 = _normalize_for_canon(name2)

    # Exact match after normalization
    if n1 == n2:
        return True

    # Try with suffix expansion
    if _expand_suffix(n1) == _expand_suffix(n2):
        return True

    return False

# Pre-compile the suffix pattern for performance
_SUFFIX_PATTERN = re.compile(
    r'\s+(' + '|'.join(re.escape(suffix) for suffix in COMPANY_SUFFIXES) + r')\.?$',
    re.IGNORECASE
)


def _clean_org_name(name: str | None) -> str:
    """
    Remove special characters and formatting from organization name.

    Removes brackets, parentheses, quotes, and other formatting artifacts.
    """
    if not name:
        return ""
    # Remove special characters, keeping only alphanumeric and spaces
    cleaned = re.sub(r'[•;:\'"\[\](){}<>`~!@#$%^&*\-_=+\\|/?!`~]+', ' ', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Recurse if changes were made (handles nested special chars)
    return _clean_org_name(cleaned) if cleaned != name else cleaned


def _remove_suffix(name: str) -> str:
    """
    Remove corporate legal suffixes from company name.

    Iteratively removes suffixes until no more are found.
    Also removes possessive 's and trailing punctuation.
    """
    cleaned = name.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove possessive 's (e.g., "Amazon's" -> "Amazon")
    cleaned = re.sub(r"'s\b", "", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    while True:
        new_name = _SUFFIX_PATTERN.sub('', cleaned)
        # Remove trailing punctuation
        new_name = re.sub(r'[ .,;&\n\t/)]$', '', new_name)

        if new_name == cleaned:
            break
        cleaned = new_name.strip()

    return cleaned.strip()


def _normalize_name(name: str) -> str:
    """
    Normalize company name for text matching.

    1. Remove possessive 's (before cleaning removes apostrophe)
    2. Clean special characters
    3. Remove legal suffixes
    4. Lowercase
    5. If result is empty, use cleaned lowercase original

    Always returns a non-empty string for valid input.
    """
    if not name:
        return ""
    # Remove possessive 's first (before cleaning removes the apostrophe)
    normalized = re.sub(r"'s\b", "", name)
    # Clean special characters
    cleaned = _clean_org_name(normalized)
    # Remove legal suffixes
    normalized = _remove_suffix(cleaned)
    # Lowercase for matching
    normalized = normalized.lower()
    # If normalized is empty (e.g., name was just "Ltd"), use the cleaned name
    if not normalized:
        normalized = cleaned.lower() if cleaned else name.lower()
    return normalized


def _extract_search_terms(query: str) -> list[str]:
    """
    Extract search terms from a query for SQL LIKE matching.

    Returns list of terms to search for, ordered by length (longest first).
    """
    # Split into words
    words = query.split()

    # Filter out very short words (< 3 chars) unless it's the only word
    if len(words) > 1:
        words = [w for w in words if len(w) >= 3]

    # Sort by length descending (longer words are more specific)
    words.sort(key=len, reverse=True)

    return words[:3]  # Limit to top 3 terms


# Person name normalization patterns
_PERSON_PREFIXES = {
    "dr.", "dr", "prof.", "prof", "professor",
    "mr.", "mr", "mrs.", "mrs", "ms.", "ms", "miss",
    "sir", "dame", "lord", "lady",
    "rev.", "rev", "reverend",
    "hon.", "hon", "honorable",
    "gen.", "gen", "general",
    "col.", "col", "colonel",
    "capt.", "capt", "captain",
    "lt.", "lt", "lieutenant",
    "sgt.", "sgt", "sergeant",
}

_PERSON_SUFFIXES = {
    "jr.", "jr", "junior",
    "sr.", "sr", "senior",
    "ii", "iii", "iv", "v",
    "2nd", "3rd", "4th", "5th",
    "phd", "ph.d.", "ph.d",
    "md", "m.d.", "m.d",
    "esq", "esq.",
    "mba", "m.b.a.",
    "cpa", "c.p.a.",
    "jd", "j.d.",
}


def _invert_date_str(date_str: str) -> str:
    """Invert a date string for descending sort (most recent first).

    Works by replacing each digit d with (9-d). Empty strings sort last
    because the caller places them after entries with dates.
    """
    if not date_str:
        return ""
    return "".join(chr(ord("9") - ord(c) + ord("0")) if c.isdigit() else c for c in date_str)


def _normalize_person_name(name: str) -> str:
    """
    Normalize person name for text matching.

    1. Remove honorific prefixes (Dr., Prof., Mr., etc.)
    2. Remove generational suffixes (Jr., Sr., III, PhD, etc.)
    3. Keep name particles (von, van, de, al-, etc.)
    4. Lowercase and strip

    Always returns a non-empty string for valid input.
    """
    if not name:
        return ""

    # Lowercase for matching
    normalized = name.lower().strip()

    # Split into words
    words = normalized.split()
    if not words:
        return ""

    # Remove prefix if first word is a title
    while words and words[0].rstrip(".") in _PERSON_PREFIXES:
        words.pop(0)
        if not words:
            return name.lower().strip()  # Fallback if name was just a title

    # Remove suffix if last word is a suffix
    while words and words[-1].rstrip(".") in _PERSON_SUFFIXES:
        words.pop()
        if not words:
            return name.lower().strip()  # Fallback if name was just suffixes

    # Rejoin remaining words
    normalized = " ".join(words)

    # Clean up extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized if normalized else name.lower().strip()


# ---------------------------------------------------------------------------
# PCA spatial pre-filtering helpers
# ---------------------------------------------------------------------------



def build_hnsw_index(
    conn: sqlite3.Connection,
    entity_type: str,
    embedding_dim: int = 768,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 200,
    progress_callback: Optional[Any] = None,
) -> int:
    """Build HNSW index for fast approximate nearest neighbor search.

    Args:
        conn: Read-only database connection
        entity_type: 'people' or 'organizations'
        embedding_dim: Dimension of embeddings (default 768)
        M: Number of connections per node (default 32)
        ef_construction: Size of dynamic candidate list during construction (default 200)
        ef_search: Size of dynamic candidate list during search (default 200)
        progress_callback: Optional callable(processed, total) for progress

    Returns:
        Number of vectors indexed
    """
    from usearch.index import Index

    # Get database path for saving index
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    cache_dir = db_path.parent

    if entity_type == "people":
        embedding_table = "person_embeddings_scalar"
        id_column = "person_id"
        index_path = cache_dir / "people_usearch.bin"
    elif entity_type == "organizations":
        embedding_table = "organization_embeddings_scalar"
        id_column = "org_id"
        index_path = cache_dir / "organizations_usearch.bin"
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

    # Count total vectors
    total_count = conn.execute(f"SELECT COUNT(*) FROM {embedding_table}").fetchone()[0]
    if total_count == 0:
        logger.warning(f"No embeddings found in {embedding_table}")
        return 0

    logger.info(f"Building USearch index for {total_count:,} {entity_type} vectors")
    logger.info(f"Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    logger.info(f"Estimated time: ~{total_count / 1000 / 60:.0f}-{total_count / 500 / 60:.0f} minutes")

    # Initialize USearch index with int8 quantization
    index = Index(
        ndim=embedding_dim,
        metric='cos',  # cosine similarity
        dtype='i8',  # int8 quantization
        connectivity=M,
        expansion_add=ef_construction,
        expansion_search=ef_search,
    )

    # Fetch all embeddings
    logger.info("Fetching embeddings from database...")
    cursor = conn.execute(f"SELECT {id_column}, embedding FROM {embedding_table} ORDER BY {id_column}")

    ids_list = []
    vectors_list = []

    for i, row in enumerate(cursor):
        entity_id = row[id_column]
        embedding_blob = row["embedding"]
        # Keep as int8 for usearch
        vec = np.frombuffer(embedding_blob, dtype=np.int8)

        ids_list.append(entity_id)
        vectors_list.append(vec)

        if progress_callback and (i + 1) % 100000 == 0:
            logger.info(f"Loaded {i + 1:,}/{total_count:,} vectors...")
            progress_callback(i + 1, total_count)

    logger.info(f"Loaded {len(ids_list):,} vectors, building index...")

    # Convert to numpy arrays
    ids_array = np.array(ids_list, dtype=np.int64)  # usearch requires int64 keys
    vectors_array = np.stack(vectors_list)

    # Add to index (usearch API: add(keys, vectors))
    index.add(ids_array, vectors_array, threads=4)

    # Save index
    logger.info(f"Saving index to {index_path.name}...")
    index.save(str(index_path))

    index_size_mb = index_path.stat().st_size / 1024**2
    logger.info(f"USearch index built successfully: {len(ids_array):,} vectors")
    logger.info(f"  Index file: {index_path.name} ({index_size_mb:.1f} MB)")

    return len(ids_array)


def get_database(db_path: Optional[str | Path] = None, embedding_dim: int = 768, readonly: bool = True) -> "OrganizationDatabase":
    """
    Get a singleton OrganizationDatabase instance for the given path.

    Args:
        db_path: Path to database file
        embedding_dim: Dimension of embeddings
        readonly: If True (default), open in read-only mode.

    Returns:
        Shared OrganizationDatabase instance (readonly by default)
    """
    path_key = str(db_path or DEFAULT_DB_PATH) + (":ro" if readonly else ":rw")
    if path_key not in _database_instances:
        logger.debug(f"Creating new OrganizationDatabase instance for {path_key}")
        _database_instances[path_key] = OrganizationDatabase(db_path=db_path, embedding_dim=embedding_dim, readonly=readonly)
    return _database_instances[path_key]


class OrganizationDatabase:
    """
    SQLite database with sqlite-vec for organization vector search.

    Uses hybrid text + vector search:
    1. Text filtering with Levenshtein distance to reduce candidates
    2. sqlite-vec for semantic similarity ranking
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_dim: int = 768,  # Default for embeddinggemma-300m
        readonly: bool = True,
    ):
        """
        Initialize the organization database.

        Args:
            db_path: Path to database file (creates if not exists)
            embedding_dim: Dimension of embeddings to store
            readonly: If True (default), open in read-only mode (avoids locking).
                      Set to False for import operations.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._embedding_dim = embedding_dim
        self._readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None
        self._is_v2: Optional[bool] = None  # Detected on first connect
        self._schema_version: int = 1  # Updated on connect (1=legacy, 2=normalized FKs, 3=no embeddings in lite)
        # Cached flags
        self._has_scalar_cached: Optional[bool] = None
        # USearch index for fast approximate nearest neighbor search
        self._hnsw_index: Optional[Any] = None  # usearch.index.Index

    def _ensure_dir(self) -> None:
        """Ensure database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, self._embedding_dim, self._readonly)

        # Detect schema version BEFORE creating tables
        # v2 has entity_type_id (FK) instead of entity_type (TEXT)
        if self._is_v2 is None:
            cursor = self._conn.execute("PRAGMA table_info(organizations)")
            columns = {row["name"] for row in cursor}
            self._is_v2 = "entity_type_id" in columns
            if self._is_v2:
                self._schema_version = 2
                logger.debug("Detected v2 schema for organizations")

            # Check for v3 metadata table
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='db_info'"
            )
            if cursor.fetchone():
                row = self._conn.execute(
                    "SELECT value FROM db_info WHERE key = 'schema_version'"
                ).fetchone()
                if row:
                    self._schema_version = int(row[0])
                    logger.debug(f"Detected schema version {self._schema_version} from db_info")

        # Create tables (idempotent) - only for v1 schema or fresh databases
        # v2 databases already have their schema from migration
        # Skip table creation in readonly mode
        if not self._is_v2 and not self._readonly:
            self._create_tables()

        return self._conn

    @property
    def _org_table(self) -> str:
        """Return table/view name for organization queries needing text fields."""
        return "organizations_view" if self._is_v2 else "organizations"

    def _create_tables(self) -> None:
        """Create database tables including sqlite-vec virtual table."""
        conn = self._conn
        assert conn is not None

        # Main organization records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id)
            )
        """)

        # Add region column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN region TEXT NOT NULL DEFAULT ''")
            logger.info("Added region column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add entity_type column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'unknown'")
            logger.info("Added entity_type column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add from_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN from_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added from_date column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add to_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN to_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added to_date column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_id column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN canon_id INTEGER DEFAULT NULL")
            logger.info("Added canon_id column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_size column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE organizations ADD COLUMN canon_size INTEGER DEFAULT 1")
            logger.info("Added canon_size column to organizations table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source ON organizations(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_region ON organizations(region)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_entity_type ON organizations(entity_type)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_name_region_source ON organizations(name, region, source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_canon_id ON organizations(canon_id)")

        # Create sqlite-vec virtual table for embeddings (float32)
        # vec0 is the recommended virtual table type
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS organization_embeddings USING vec0(
                org_id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dim}] distance_metric=cosine
            )
        """)

        # Create sqlite-vec virtual table for scalar embeddings (int8)
        # Provides 75% storage reduction with ~92% recall at top-100
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS organization_embeddings_scalar USING vec0(
                org_id INTEGER PRIMARY KEY,
                embedding int8[{self._embedding_dim}] distance_metric=cosine
            )
        """)

        conn.commit()

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(
        self,
        record: CompanyRecord,
        embedding: np.ndarray,
        scalar_embedding: Optional[np.ndarray] = None,
    ) -> int:
        """
        Insert an organization record with its embedding.

        Args:
            record: Organization record to insert
            embedding: Embedding vector for the organization name (float32)
            scalar_embedding: Optional int8 scalar embedding for compact storage

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_name(record.name)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO organizations
            (name, name_normalized, source, source_id, region, entity_type, from_date, to_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.region,
            record.entity_type.value,
            record.from_date or "",
            record.to_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table (float32)
        # sqlite-vec virtual tables don't support INSERT OR REPLACE, so delete first
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (row_id,))
        conn.execute("""
            INSERT INTO organization_embeddings (org_id, embedding)
            VALUES (?, ?)
        """, (row_id, embedding_blob))

        # Insert scalar embedding if provided (int8)
        if scalar_embedding is not None:
            scalar_blob = scalar_embedding.astype(np.int8).tobytes()
            conn.execute("DELETE FROM organization_embeddings_scalar WHERE org_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO organization_embeddings_scalar (org_id, embedding)
                VALUES (?, vec_int8(?))
            """, (row_id, scalar_blob))

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[CompanyRecord],
        embeddings: np.ndarray,
        batch_size: int = 1000,
        scalar_embeddings: Optional[np.ndarray] = None,
    ) -> int:
        """
        Insert multiple organization records with embeddings.

        Args:
            records: List of organization records
            embeddings: Matrix of embeddings (N x dim) - float32
            batch_size: Commit batch size
            scalar_embeddings: Optional matrix of int8 scalar embeddings (N x dim)

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for i, (record, embedding) in enumerate(zip(records, embeddings)):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_name(record.name)

            if self._is_v2:
                # v2 schema: use FK IDs instead of TEXT columns
                source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
                entity_type_id = ORG_TYPE_NAME_TO_ID.get(record.entity_type.value, 17)  # 17 = unknown

                # Resolve region to location_id if provided
                region_id = None
                if record.region:
                    # Use readonly=False to avoid immutable mode conflicts with write connection
                    locations_db = get_locations_database(db_path=self._db_path, readonly=False)
                    region_id = locations_db.resolve_region_text(record.region)

                cursor = conn.execute("""
                    INSERT OR REPLACE INTO organizations
                    (name, name_normalized, source_id, source_identifier, region_id, entity_type_id, from_date, to_date, record)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.name,
                    name_normalized,
                    source_type_id,
                    record.source_id,
                    region_id,
                    entity_type_id,
                    record.from_date or "",
                    record.to_date or "",
                    record_json,
                ))
            else:
                # v1 schema: use TEXT columns
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO organizations
                    (name, name_normalized, source, source_id, region, entity_type, from_date, to_date, record)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.name,
                    name_normalized,
                    record.source,
                    record.source_id,
                    record.region,
                    record.entity_type.value,
                    record.from_date or "",
                    record.to_date or "",
                    record_json,
                ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding (delete first since sqlite-vec doesn't support REPLACE)
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (row_id, embedding_blob))

            # Insert scalar embedding if provided (int8)
            if scalar_embeddings is not None:
                scalar_blob = scalar_embeddings[i].astype(np.int8).tobytes()
                conn.execute("DELETE FROM organization_embeddings_scalar WHERE org_id = ?", (row_id,))
                conn.execute("""
                    INSERT INTO organization_embeddings_scalar (org_id, embedding)
                    VALUES (?, vec_int8(?))
                """, (row_id, scalar_blob))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} records...")

        conn.commit()
        return count

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        source_filter: Optional[str] = None,
        query_text: Optional[str] = None,
        max_text_candidates: int = 5000,
        rerank_min_candidates: int = 500,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar organizations using hybrid text + vector search.

        Three-stage approach:
        1. If query_text provided, use SQL LIKE to find candidates containing search terms
        2. USearch HNSW index for fast approximate nearest neighbor search
        3. Apply prominence-based re-ranking to boost major companies (SEC filers, tickers)

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)
            query_text: Optional query text for text-based pre-filtering
            max_text_candidates: Max candidates to keep after text filtering
            rerank_min_candidates: Minimum candidates to fetch for re-ranking (default 500)

        Returns:
            List of (CompanyRecord, adjusted_score) tuples sorted by prominence-adjusted score
        """
        start = time.time()
        self._connect()

        # Log search mode
        if query_text:
            logger.info(f"Organization search: hybrid mode (text + embeddings) for '{query_text}'")
        else:
            logger.debug("Organization search: embeddings-only mode (fast KNN)")

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Stage 1: Text filtering (if query_text provided)
        candidate_ids: Optional[set[int]] = None
        query_normalized_text = ""
        if query_text:
            query_normalized_text = _normalize_name(query_text)
            if query_normalized_text:
                candidate_ids = self._text_filter_candidates(
                    query_normalized_text,
                    max_candidates=max_text_candidates,
                    source_filter=source_filter,
                )
                logger.info(f"Text filter: {len(candidate_ids)} candidates for '{query_text}'")
                if len(candidate_ids) == 0:
                    return []

        # Stage 2: Vector ranking via USearch
        query_int8 = np.clip(np.round(query_normalized * 127), -127, 127).astype(np.int8)

        # Fetch extra results if we'll be re-ranking (hybrid mode)
        fetch_k = max(rerank_min_candidates, top_k * 5) if query_normalized_text else top_k

        if not self._load_hnsw_index():
            raise RuntimeError("USearch index not found. Run: corp-extractor db build-index")

        results = self._hnsw_search(query_int8, fetch_k, candidate_ids, source_filter)

        # Stage 3: Prominence-based re-ranking (hybrid mode only)
        if results and query_normalized_text:
            logger.debug(f"Applying prominence re-ranking to {len(results)} candidates")
            results = self._apply_prominence_reranking(results, query_normalized_text, top_k)
        else:
            # No re-ranking, just trim to top_k
            results = results[:top_k]

        elapsed = time.time() - start
        logger.debug(f"Organization search took {elapsed:.3f}s (results={len(results)})")
        return results

    def _calculate_prominence_boost(
        self,
        record: CompanyRecord,
        query_normalized: str,
        canon_sources: Optional[set[str]] = None,
    ) -> float:
        """
        Calculate prominence boost for re-ranking search results.

        Boosts scores based on signals that indicate a major/prominent company:
        - Has ticker symbol (publicly traded)
        - GLEIF source (has LEI)
        - SEC source (vetted US filers)
        - Wikidata source (Wikipedia-notable)
        - Exact normalized name match

        When canon_sources is provided (from a canonical group), boosts are
        applied for ALL sources in the canon group, not just this record's source.

        Args:
            record: The company record to evaluate
            query_normalized: Normalized query text for exact match check
            canon_sources: Optional set of sources in this record's canonical group

        Returns:
            Boost value to add to embedding similarity (0.0 to ~0.21)
        """
        boost = 0.0

        # Get all sources to consider (canon group or just this record)
        sources_to_check = canon_sources or {record.source}

        # Has ticker symbol = publicly traded major company
        # Check if ANY record in canon group has ticker
        if record.record.get("ticker") or (canon_sources and "sec_edgar" in canon_sources):
            boost += 0.08

        # Source-based boosts - accumulate for all sources in canon group
        if "gleif" in sources_to_check:
            boost += 0.05  # Has LEI = verified legal entity
        if "sec_edgar" in sources_to_check:
            boost += 0.03  # SEC filer
        if "wikipedia" in sources_to_check:
            boost += 0.02  # Wikipedia notable

        # Exact normalized name match bonus
        record_normalized = _normalize_name(record.name)
        if query_normalized == record_normalized:
            boost += 0.05

        return boost

    def _apply_prominence_reranking(
        self,
        results: list[tuple[CompanyRecord, float]],
        query_normalized: str,
        top_k: int,
        similarity_weight: float = 0.3,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Apply prominence-based re-ranking to search results with canon group awareness.

        When records have been canonicalized, boosts are applied based on ALL sources
        in the canonical group, not just the matched record's source.

        Args:
            results: List of (record, similarity) from vector search
            query_normalized: Normalized query text
            top_k: Number of results to return after re-ranking
            similarity_weight: Weight for similarity score (0-1), lower = prominence matters more

        Returns:
            Re-ranked list of (record, adjusted_score) tuples
        """
        conn = self._conn
        assert conn is not None

        # Build canon_id -> sources mapping for all results that have canon_id
        canon_sources_map: dict[int, set[str]] = {}
        canon_ids = [
            r.record.get("canon_id")
            for r, _ in results
            if r.record.get("canon_id") is not None
        ]

        if canon_ids:
            # Fetch all sources for each canon_id in one query
            unique_canon_ids = list(set(canon_ids))
            placeholders = ",".join("?" * len(unique_canon_ids))
            rows = conn.execute(f"""
                SELECT canon_id, source
                FROM organizations
                WHERE canon_id IN ({placeholders})
            """, unique_canon_ids).fetchall()

            for row in rows:
                canon_id = row["canon_id"]
                canon_sources_map.setdefault(canon_id, set()).add(row["source"])

        # Calculate boosted scores with canon group awareness
        # Formula: adjusted = (similarity * weight) + boost
        # With weight=0.3, a sim=0.65 SEC+ticker (boost=0.11) beats sim=0.75 no-boost
        boosted_results: list[tuple[CompanyRecord, float, float, float]] = []
        for record, similarity in results:
            canon_id = record.record.get("canon_id")
            # Get all sources in this record's canon group (if any)
            canon_sources = canon_sources_map.get(canon_id) if canon_id else None

            boost = self._calculate_prominence_boost(record, query_normalized, canon_sources)
            adjusted_score = (similarity * similarity_weight) + boost
            boosted_results.append((record, similarity, boost, adjusted_score))

        # Sort by adjusted score (descending)
        boosted_results.sort(key=lambda x: x[3], reverse=True)

        # Log re-ranking details for top results
        logger.debug(f"Prominence re-ranking for '{query_normalized}':")
        for record, sim, boost, adj in boosted_results[:10]:
            ticker = record.record.get("ticker", "")
            ticker_str = f" ticker={ticker}" if ticker else ""
            canon_id = record.record.get("canon_id")
            canon_str = f" canon={canon_id}" if canon_id else ""
            logger.debug(
                f"  {record.name}: sim={sim:.3f} + boost={boost:.3f} = {adj:.3f} "
                f"[{record.source}{ticker_str}{canon_str}]"
            )

        # Return top_k with adjusted scores
        return [(r, adj) for r, _, _, adj in boosted_results[:top_k]]

    def _text_filter_candidates(
        self,
        query_normalized: str,
        max_candidates: int,
        source_filter: Optional[str] = None,
    ) -> set[int]:
        """
        Filter candidates using SQL LIKE for fast text matching.

        This is a generous pre-filter to reduce the embedding search space.
        Returns set of organization IDs that contain any search term.
        Uses `name_normalized` column for consistent matching.
        """
        conn = self._conn
        assert conn is not None

        # Extract search terms from the normalized query
        search_terms = _extract_search_terms(query_normalized)
        if not search_terms:
            return set()

        logger.debug(f"Text filter search terms: {search_terms}")

        # Build OR clause for LIKE matching on any term
        # Use name_normalized for consistent matching (already lowercased, suffixes removed)
        like_clauses = []
        params: list = []
        for term in search_terms:
            like_clauses.append("name_normalized LIKE ?")
            params.append(f"%{term}%")

        where_clause = " OR ".join(like_clauses)

        # Add source filter if specified
        if source_filter:
            query = f"""
                SELECT id FROM organizations
                WHERE ({where_clause}) AND source = ?
                LIMIT ?
            """
            params.append(source_filter)
        else:
            query = f"""
                SELECT id FROM organizations
                WHERE {where_clause}
                LIMIT ?
            """

        params.append(max_candidates)

        cursor = conn.execute(query, params)
        return set(row["id"] for row in cursor)

    def _has_scalar_table(self) -> bool:
        """Check if scalar embedding table exists (cached)."""
        if self._has_scalar_cached is not None:
            return self._has_scalar_cached
        conn = self._conn
        assert conn is not None
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='organization_embeddings_scalar'"
        )
        result = cursor.fetchone() is not None
        self._has_scalar_cached = result
        return result

    # --- USearch approximate nearest neighbor search ---

    def _get_hnsw_index_path(self) -> Path:
        """Get path to USearch index file."""
        return self._db_path.parent / "organizations_usearch.bin"

    def _load_hnsw_index(self) -> bool:
        """
        Load pre-built USearch index from disk.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if self._hnsw_index is not None:
            return True  # Already loaded

        index_path = self._get_hnsw_index_path()
        if not index_path.exists():
            logger.debug(f"USearch index not found: {index_path}")
            return False

        try:
            from usearch.index import Index

            # Load index (IDs are stored in the index itself)
            logger.info(f"Loading USearch index from {index_path.name}...")
            index = Index.restore(str(index_path))
            # USearch doesn't persist expansion_search — restore it for good recall on large indexes
            index.expansion_search = 200
            self._hnsw_index = index
            logger.info(f"Loaded USearch index: {len(index):,} vectors, connectivity={index.connectivity}, ef_search={index.expansion_search}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load USearch index: {e}")
            return False

    def _hnsw_search(
        self,
        query_int8: np.ndarray,
        top_k: int,
        candidate_ids: Optional[set[int]] = None,
        source_filter: Optional[str] = None,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search using USearch index.

        Args:
            query_int8: Query embedding quantized to int8
            top_k: Number of results to return
            candidate_ids: Optional set of candidate IDs to filter results
            source_filter: Optional source filter

        Returns:
            List of (CompanyRecord, similarity) tuples
        """
        if self._hnsw_index is None:
            return []

        # Query USearch index (fetch more if we need to filter)
        fetch_k = top_k * 10 if (candidate_ids or source_filter) else top_k
        fetch_k = min(fetch_k, len(self._hnsw_index))

        matches = self._hnsw_index.search(query_int8, fetch_k)

        # Convert to results
        results: list[tuple[CompanyRecord, float]] = []
        for org_id, dist in zip(matches.keys, matches.distances):
            org_id = int(org_id)

            # Filter if needed
            if candidate_ids and org_id not in candidate_ids:
                continue

            # Convert distance to similarity (cosine distance is 1 - cosine similarity)
            similarity = 1.0 - float(dist)

            record = self._get_record_by_id(org_id)
            if record is not None:
                # Apply source filter if specified
                if source_filter and record.source != source_filter:
                    continue
                results.append((record, similarity))

            if len(results) >= top_k:
                break

        return results

    def _get_record_by_id(self, org_id: int) -> Optional[CompanyRecord]:
        """Get an organization record by ID, including db_id and canon_id in record dict."""
        conn = self._conn
        assert conn is not None

        if self._is_v2:
            # v2 schema: use view for text fields, but need record from base table
            cursor = conn.execute("""
                SELECT v.id, v.name, v.source, v.source_identifier, v.region, v.entity_type, v.canon_id, o.record
                FROM organizations_view v
                JOIN organizations o ON v.id = o.id
                WHERE v.id = ?
            """, (org_id,))
        else:
            cursor = conn.execute("""
                SELECT id, name, source, source_id, region, entity_type, record, canon_id
                FROM organizations WHERE id = ?
            """, (org_id,))

        row = cursor.fetchone()
        if row:
            record_data = json.loads(row["record"])
            # Add db_id and canon_id to record dict for canon-aware search
            record_data["db_id"] = row["id"]
            record_data["canon_id"] = row["canon_id"]
            source_id_field = "source_identifier" if self._is_v2 else "source_id"
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row[source_id_field],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=record_data,
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[CompanyRecord]:
        """Get an organization record by source and source_id."""
        conn = self._connect()

        if self._is_v2:
            # v2 schema: join view with base table for record
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute("""
                SELECT v.name, v.source, v.source_identifier, v.region, v.entity_type, o.record
                FROM organizations_view v
                JOIN organizations o ON v.id = o.id
                WHERE o.source_id = ? AND o.source_identifier = ?
            """, (source_type_id, source_id))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, region, entity_type, record
                FROM organizations
                WHERE source = ? AND source_id = ?
            """, (source, source_id))

        row = cursor.fetchone()
        if row:
            source_id_field = "source_identifier" if self._is_v2 else "source_id"
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row[source_id_field],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=json.loads(row["record"]),
            )
        return None

    def get_id_by_source_id(self, source: str, source_id: str) -> Optional[int]:
        """Get the internal database ID for an organization by source and source_id."""
        conn = self._connect()

        if self._is_v2:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute("""
                SELECT id FROM organizations
                WHERE source_id = ? AND source_identifier = ?
            """, (source_type_id, source_id))
        else:
            cursor = conn.execute("""
                SELECT id FROM organizations
                WHERE source = ? AND source_id = ?
            """, (source, source_id))

        row = cursor.fetchone()
        if row:
            return row["id"]
        return None

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM organizations")
        total = cursor.fetchone()[0]

        # Count by source - handle both v1 and v2 schema
        if self._is_v2:
            # v2 schema - join with source_types
            cursor = conn.execute("""
                SELECT st.name as source, COUNT(*) as cnt
                FROM organizations o
                JOIN source_types st ON o.source_id = st.id
                GROUP BY o.source_id
            """)
        else:
            # v1 schema
            cursor = conn.execute("SELECT source, COUNT(*) as cnt FROM organizations GROUP BY source")
        by_source = {row["source"]: row["cnt"] for row in cursor}

        # Database file size
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return DatabaseStats(
            total_records=total,
            by_source=by_source,
            embedding_dimension=self._embedding_dim,
            database_size_bytes=db_size,
        )

    def get_all_source_ids(self, source: Optional[str] = None) -> set[str]:
        """
        Get all source_ids from the organizations table.

        Useful for resume operations to skip already-imported records.

        Args:
            source: Optional source filter (e.g., "wikidata" for Wikidata orgs)

        Returns:
            Set of source_id strings (e.g., Q codes for Wikidata)
        """
        conn = self._connect()

        if self._is_v2:
            id_col = "source_identifier"
            if source:
                source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
                cursor = conn.execute(
                    f"SELECT DISTINCT {id_col} FROM organizations WHERE source_id = ?",
                    (source_type_id,)
                )
            else:
                cursor = conn.execute(f"SELECT DISTINCT {id_col} FROM organizations")
        else:
            if source:
                cursor = conn.execute(
                    "SELECT DISTINCT source_id FROM organizations WHERE source = ?",
                    (source,)
                )
            else:
                cursor = conn.execute("SELECT DISTINCT source_id FROM organizations")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[CompanyRecord]:
        """Iterate over all records, optionally filtered by source."""
        conn = self._connect()

        if self._is_v2:
            if source:
                source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
                cursor = conn.execute("""
                    SELECT v.name, v.source, v.source_identifier, v.region, v.entity_type, o.record
                    FROM organizations_view v
                    JOIN organizations o ON v.id = o.id
                    WHERE o.source_id = ?
                """, (source_type_id,))
            else:
                cursor = conn.execute("""
                    SELECT v.name, v.source, v.source_identifier, v.region, v.entity_type, o.record
                    FROM organizations_view v
                    JOIN organizations o ON v.id = o.id
                """)
            for row in cursor:
                yield CompanyRecord(
                    name=row["name"],
                    source=row["source"],
                    source_id=row["source_identifier"],
                    region=row["region"] or "",
                    entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                    record=json.loads(row["record"]),
                )
        else:
            if source:
                cursor = conn.execute("""
                    SELECT name, source, source_id, region, entity_type, record
                    FROM organizations
                    WHERE source = ?
                """, (source,))
            else:
                cursor = conn.execute("""
                    SELECT name, source, source_id, region, entity_type, record
                    FROM organizations
                """)
            for row in cursor:
                yield CompanyRecord(
                    name=row["name"],
                    source=row["source"],
                    source_id=row["source_id"],
                    region=row["region"] or "",
                    entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                    record=json.loads(row["record"]),
                )

    def canonicalize(self, batch_size: int = 10000) -> dict[str, int]:
        """
        Canonicalize all organizations by linking equivalent records.

        Records are considered equivalent if they match by:
        1. Same LEI (GLEIF source_id or Wikidata P1278) - globally unique, no region check
        2. Same ticker symbol - globally unique, no region check
        3. Same CIK - globally unique, no region check
        4. Same normalized name AND same normalized region
        5. Name match with suffix expansion AND same region

        Region normalization uses pycountry to handle:
        - Country codes/names (GB, United Kingdom, Great Britain -> GB)
        - US state codes/names (CA, California -> US)
        - Common aliases (UK -> GB, USA -> US)

        For each group of equivalent records, the highest-priority source
        (gleif > sec_edgar > companies_house > wikipedia) becomes canonical.

        Args:
            batch_size: Commit batch size for updates

        Returns:
            Dict with stats: total_records, groups_found, records_updated
        """
        conn = self._connect()
        logger.info("Starting canonicalization...")

        # Phase 1: Load all organization data and build indexes
        logger.info("Phase 1: Building indexes...")

        lei_index: dict[str, list[int]] = {}
        ticker_index: dict[str, list[int]] = {}
        cik_index: dict[str, list[int]] = {}
        # Name indexes now keyed by (normalized_name, normalized_region)
        # Region-less matching only applies for identifier-based matching
        name_region_index: dict[tuple[str, str], list[int]] = {}
        expanded_name_region_index: dict[tuple[str, str], list[int]] = {}

        sources: dict[int, str] = {}  # org_id -> source
        all_org_ids: list[int] = []

        if self._is_v2:
            cursor = conn.execute("""
                SELECT o.id, s.name as source, o.source_identifier as source_id, o.name, l.name as region, o.record
                FROM organizations o
                JOIN source_types s ON o.source_id = s.id
                LEFT JOIN locations l ON o.region_id = l.id
            """)
        else:
            cursor = conn.execute("""
                SELECT id, source, source_id, name, region, record
                FROM organizations
            """)

        count = 0
        for row in cursor:
            org_id = row["id"]
            source = row["source"]
            name = row["name"]
            region = row["region"] or ""
            record = json.loads(row["record"])

            all_org_ids.append(org_id)
            sources[org_id] = source

            # Index by LEI (GLEIF source_id or Wikidata's P1278)
            # LEI is globally unique - no region check needed
            if source == "gleif":
                lei = row["source_id"]
            else:
                lei = record.get("lei")
            if lei:
                lei_index.setdefault(lei.upper(), []).append(org_id)

            # Index by ticker - globally unique, no region check
            ticker = record.get("ticker")
            if ticker:
                ticker_index.setdefault(ticker.upper(), []).append(org_id)

            # Index by CIK - globally unique, no region check
            if source == "sec_edgar":
                cik = row["source_id"]
            else:
                cik = record.get("cik")
            if cik:
                cik_index.setdefault(str(cik), []).append(org_id)

            # Index by (normalized_name, normalized_region)
            # Same name in different regions = different legal entities
            norm_name = _normalize_for_canon(name)
            norm_region = _normalize_region(region)
            if norm_name:
                key = (norm_name, norm_region)
                name_region_index.setdefault(key, []).append(org_id)

            # Index by (expanded_name, normalized_region)
            expanded_name = _expand_suffix(name)
            if expanded_name and expanded_name != norm_name:
                key = (expanded_name, norm_region)
                expanded_name_region_index.setdefault(key, []).append(org_id)

            count += 1
            if count % 100000 == 0:
                logger.info(f"  Indexed {count} organizations...")

        logger.info(f"  Indexed {count} organizations total")
        logger.info(f"  LEI index: {len(lei_index)} unique LEIs")
        logger.info(f"  Ticker index: {len(ticker_index)} unique tickers")
        logger.info(f"  CIK index: {len(cik_index)} unique CIKs")
        logger.info(f"  Name+region index: {len(name_region_index)} unique (name, region) pairs")
        logger.info(f"  Expanded name+region index: {len(expanded_name_region_index)} unique pairs")

        # Phase 2: Build equivalence groups using Union-Find
        logger.info("Phase 2: Building equivalence groups...")

        uf = UnionFind(all_org_ids)

        # Merge by LEI (globally unique identifier)
        for _lei, ids in lei_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by ticker (globally unique identifier)
        for _ticker, ids in ticker_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by CIK (globally unique identifier)
        for _cik, ids in cik_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by (normalized_name, normalized_region)
        for _name_region, ids in name_region_index.items():
            for i in range(1, len(ids)):
                uf.union(ids[0], ids[i])

        # Merge by (expanded_name, normalized_region)
        # This connects "Amazon Ltd" with "Amazon Limited" in same region
        for key, expanded_ids in expanded_name_region_index.items():
            # Find org_ids with the expanded form as their normalized name in same region
            if key in name_region_index:
                # Link first expanded_id to first name_id
                uf.union(expanded_ids[0], name_region_index[key][0])

        groups = uf.groups()
        logger.info(f"  Found {len(groups)} equivalence groups")

        # Count groups with multiple records
        multi_record_groups = sum(1 for ids in groups.values() if len(ids) > 1)
        logger.info(f"  Groups with multiple records: {multi_record_groups}")

        # Phase 3: Select canonical record for each group and update database
        logger.info("Phase 3: Updating database...")

        updated_count = 0
        batch_updates: list[tuple[int, int, int]] = []  # (org_id, canon_id, canon_size)

        for _root, group_ids in groups.items():
            if len(group_ids) == 1:
                # Single record - canonical to itself
                batch_updates.append((group_ids[0], group_ids[0], 1))
            else:
                # Multiple records - find highest priority source
                best_id = min(
                    group_ids,
                    key=lambda oid: (SOURCE_PRIORITY.get(sources[oid], 99), oid)
                )
                group_size = len(group_ids)

                # All records in group point to the best one
                for oid in group_ids:
                    # canon_size is only set on the canonical record
                    size = group_size if oid == best_id else 1
                    batch_updates.append((oid, best_id, size))

            # Commit batch
            if len(batch_updates) >= batch_size:
                self._apply_canon_updates(batch_updates)
                updated_count += len(batch_updates)
                logger.info(f"  Updated {updated_count} records...")
                batch_updates = []

        # Final batch
        if batch_updates:
            self._apply_canon_updates(batch_updates)
            updated_count += len(batch_updates)

        conn.commit()
        logger.info(f"Canonicalization complete: {updated_count} records updated, {multi_record_groups} multi-record groups")

        return {
            "total_records": count,
            "groups_found": len(groups),
            "multi_record_groups": multi_record_groups,
            "records_updated": updated_count,
        }

    def _apply_canon_updates(self, updates: list[tuple[int, int, int]]) -> None:
        """Apply batch of canon updates: (org_id, canon_id, canon_size)."""
        conn = self._conn
        assert conn is not None

        for org_id, canon_id, canon_size in updates:
            conn.execute(
                "UPDATE organizations SET canon_id = ?, canon_size = ? WHERE id = ?",
                (canon_id, canon_size, org_id)
            )

        conn.commit()

    def get_canon_stats(self) -> dict[str, int]:
        """Get statistics about canonicalization status."""
        conn = self._connect()

        # Total records
        cursor = conn.execute("SELECT COUNT(*) FROM organizations")
        total = cursor.fetchone()[0]

        # Records with canon_id set
        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE canon_id IS NOT NULL")
        canonicalized = cursor.fetchone()[0]

        # Number of canonical groups (unique canon_ids)
        cursor = conn.execute("SELECT COUNT(DISTINCT canon_id) FROM organizations WHERE canon_id IS NOT NULL")
        groups = cursor.fetchone()[0]

        # Multi-record groups (canon_size > 1)
        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE canon_size > 1")
        multi_record_groups = cursor.fetchone()[0]

        # Records in multi-record groups
        cursor = conn.execute("""
            SELECT COUNT(*) FROM organizations o1
            WHERE EXISTS (SELECT 1 FROM organizations o2 WHERE o2.id = o1.canon_id AND o2.canon_size > 1)
        """)
        records_in_multi = cursor.fetchone()[0]

        return {
            "total_records": total,
            "canonicalized_records": canonicalized,
            "canonical_groups": groups,
            "multi_record_groups": multi_record_groups,
            "records_in_multi_groups": records_in_multi,
        }

    def migrate_name_normalized(self, batch_size: int = 50000) -> int:
        """
        Populate the name_normalized column for all records.

        This is a one-time migration for databases that don't have
        normalized names populated.

        Args:
            batch_size: Number of records to process per batch

        Returns:
            Number of records updated
        """
        conn = self._connect()

        # Check how many need migration (empty, null, or placeholder "-")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM organizations WHERE name_normalized = '' OR name_normalized IS NULL OR name_normalized = '-'"
        )
        empty_count = cursor.fetchone()[0]

        if empty_count == 0:
            logger.info("All records already have name_normalized populated")
            return 0

        logger.info(f"Populating name_normalized for {empty_count} records...")

        updated = 0
        last_id = 0

        while True:
            # Get batch of records that need normalization, ordered by ID
            cursor = conn.execute("""
                SELECT id, name FROM organizations
                WHERE id > ? AND (name_normalized = '' OR name_normalized IS NULL OR name_normalized = '-')
                ORDER BY id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            # Update each record
            for row in rows:
                # _normalize_name now always returns non-empty for valid input
                normalized = _normalize_name(row["name"])
                conn.execute(
                    "UPDATE organizations SET name_normalized = ? WHERE id = ?",
                    (normalized, row["id"])
                )
                last_id = row["id"]

            conn.commit()
            updated += len(rows)
            logger.info(f"  Updated {updated}/{empty_count} records...")

        logger.info(f"Migration complete: {updated} name_normalized values populated")
        return updated

    def migrate_to_sqlite_vec(self, batch_size: int = 10000) -> int:
        """
        Migrate embeddings from BLOB column to sqlite-vec virtual table.

        This is a one-time migration for databases created before sqlite-vec support.

        Args:
            batch_size: Number of records to process per batch

        Returns:
            Number of embeddings migrated
        """
        conn = self._connect()

        # Check if migration is needed
        cursor = conn.execute("SELECT COUNT(*) FROM organization_embeddings")
        vec_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM organizations WHERE embedding IS NOT NULL")
        blob_count = cursor.fetchone()[0]

        if vec_count >= blob_count:
            logger.info(f"Migration not needed: sqlite-vec has {vec_count} embeddings, BLOB has {blob_count}")
            return 0

        logger.info(f"Migrating {blob_count} embeddings from BLOB to sqlite-vec...")

        # Get IDs that need migration (in sqlite-vec but not in organizations)
        cursor = conn.execute("""
            SELECT c.id, c.embedding
            FROM organizations c
            LEFT JOIN organization_embeddings e ON c.id = e.org_id
            WHERE c.embedding IS NOT NULL AND e.org_id IS NULL
        """)

        migrated = 0
        batch = []

        for row in cursor:
            org_id = row["id"]
            embedding_blob = row["embedding"]

            if embedding_blob:
                batch.append((org_id, embedding_blob))

            if len(batch) >= batch_size:
                self._insert_vec_batch(batch)
                migrated += len(batch)
                logger.info(f"  Migrated {migrated}/{blob_count} embeddings...")
                batch = []

        # Insert remaining batch
        if batch:
            self._insert_vec_batch(batch)
            migrated += len(batch)

        logger.info(f"Migration complete: {migrated} embeddings migrated to sqlite-vec")
        return migrated

    def _insert_vec_batch(self, batch: list[tuple[int, bytes]]) -> None:
        """Insert a batch of embeddings into sqlite-vec table."""
        conn = self._conn
        assert conn is not None

        for org_id, embedding_blob in batch:
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, embedding_blob))

        conn.commit()

    def delete_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        conn = self._connect()

        if self._is_v2:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            # First get IDs to delete from vec table
            cursor = conn.execute("SELECT id FROM organizations WHERE source_id = ?", (source_type_id,))
            ids_to_delete = [row["id"] for row in cursor]

            # Delete from vec table
            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(f"DELETE FROM organization_embeddings WHERE org_id IN ({placeholders})", ids_to_delete)

            # Delete from main table
            cursor = conn.execute("DELETE FROM organizations WHERE source_id = ?", (source_type_id,))
        else:
            # First get IDs to delete from vec table
            cursor = conn.execute("SELECT id FROM organizations WHERE source = ?", (source,))
            ids_to_delete = [row["id"] for row in cursor]

            # Delete from vec table
            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                conn.execute(f"DELETE FROM organization_embeddings WHERE org_id IN ({placeholders})", ids_to_delete)

            # Delete from main table
            cursor = conn.execute("DELETE FROM organizations WHERE source = ?", (source,))

        deleted = cursor.rowcount

        conn.commit()

        logger.info(f"Deleted {deleted} records from source '{source}'")
        return deleted

    def migrate_from_legacy_schema(self) -> dict[str, str]:
        """
        Migrate database from legacy schema (companies/company_embeddings tables)
        to new schema (organizations/organization_embeddings tables).

        This handles:
        - Renaming 'companies' table to 'organizations'
        - Renaming 'company_embeddings' table to 'organization_embeddings'
        - Renaming 'company_id' column to 'org_id' in embeddings table
        - Updating indexes to use new naming

        Returns:
            Dict of migrations performed (table_name -> action)
        """
        conn = self._connect()
        migrations = {}

        # Check what tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor}

        has_companies = "companies" in existing_tables
        has_organizations = "organizations" in existing_tables
        has_company_embeddings = "company_embeddings" in existing_tables
        has_org_embeddings = "organization_embeddings" in existing_tables

        if not has_companies and not has_company_embeddings:
            if has_organizations and has_org_embeddings:
                logger.info("Database already uses new schema, no migration needed")
                return {}
            else:
                logger.info("No legacy tables found, database will use new schema")
                return {}

        logger.info("Migrating database from legacy schema...")
        conn.execute("BEGIN")

        try:
            # Migrate companies -> organizations
            if has_companies:
                if has_organizations:
                    # Both exist - merge data from companies into organizations
                    logger.info("Merging companies table into organizations...")
                    conn.execute("""
                        INSERT OR IGNORE INTO organizations
                        (name, name_normalized, source, source_id, region, entity_type, record)
                        SELECT name, name_normalized, source, source_id,
                               COALESCE(region, ''), COALESCE(entity_type, 'unknown'), record
                        FROM companies
                    """)
                    conn.execute("DROP TABLE companies")
                    migrations["companies"] = "merged_into_organizations"
                else:
                    # Just rename
                    logger.info("Renaming companies table to organizations...")
                    conn.execute("ALTER TABLE companies RENAME TO organizations")
                    migrations["companies"] = "renamed_to_organizations"

                # Update indexes
                for old_idx in ["idx_companies_name", "idx_companies_name_normalized",
                               "idx_companies_source", "idx_companies_source_id",
                               "idx_companies_region", "idx_companies_entity_type",
                               "idx_companies_name_region_source"]:
                    try:
                        conn.execute(f"DROP INDEX IF EXISTS {old_idx}")
                    except Exception:
                        pass

                # Create new indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source ON organizations(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source, source_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_region ON organizations(region)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_entity_type ON organizations(entity_type)")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orgs_name_region_source ON organizations(name, region, source)")

            # Migrate company_embeddings -> organization_embeddings
            if has_company_embeddings:
                if has_org_embeddings:
                    # Both exist - merge
                    logger.info("Merging company_embeddings into organization_embeddings...")
                    # Get column info to check for company_id vs org_id
                    cursor = conn.execute("PRAGMA table_info(company_embeddings)")
                    cols = {row[1] for row in cursor}
                    id_col = "company_id" if "company_id" in cols else "org_id"

                    conn.execute(f"""
                        INSERT OR IGNORE INTO organization_embeddings (org_id, embedding)
                        SELECT {id_col}, embedding FROM company_embeddings
                    """)
                    conn.execute("DROP TABLE company_embeddings")
                    migrations["company_embeddings"] = "merged_into_organization_embeddings"
                else:
                    # Need to recreate with new column name
                    logger.info("Migrating company_embeddings to organization_embeddings...")

                    # Check if it has company_id or org_id column
                    cursor = conn.execute("PRAGMA table_info(company_embeddings)")
                    cols = {row[1] for row in cursor}
                    id_col = "company_id" if "company_id" in cols else "org_id"

                    # Create new virtual table
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE organization_embeddings USING vec0(
                            org_id INTEGER PRIMARY KEY,
                            embedding float[{self._embedding_dim}] distance_metric=cosine
                        )
                    """)

                    # Copy data
                    conn.execute(f"""
                        INSERT INTO organization_embeddings (org_id, embedding)
                        SELECT {id_col}, embedding FROM company_embeddings
                    """)

                    # Drop old table
                    conn.execute("DROP TABLE company_embeddings")
                    migrations["company_embeddings"] = "renamed_to_organization_embeddings"

            conn.execute("COMMIT")
            logger.info(f"Migration complete: {migrations}")

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Migration failed: {e}")
            raise

        # Vacuum to clean up - outside try block since COMMIT already succeeded
        try:
            conn.execute("VACUUM")
        except Exception as e:
            logger.warning(f"VACUUM failed (migration was successful): {e}")

        return migrations

    def get_missing_embedding_count(self) -> int:
        """Get count of organizations without embeddings in organization_embeddings table."""
        conn = self._connect()

        cursor = conn.execute("""
            SELECT COUNT(*) FROM organizations c
            LEFT JOIN organization_embeddings e ON c.id = e.org_id
            WHERE e.org_id IS NULL
        """)
        return cursor.fetchone()[0]

    def get_organizations_without_embeddings(
        self,
        batch_size: int = 1000,
        source: Optional[str] = None,
    ) -> Iterator[tuple[int, str]]:
        """
        Iterate over organizations that don't have embeddings.

        Args:
            batch_size: Number of records per batch
            source: Optional source filter

        Yields:
            Tuples of (org_id, name)
        """
        conn = self._connect()

        last_id = 0
        while True:
            if source:
                cursor = conn.execute("""
                    SELECT c.id, c.name FROM organizations c
                    LEFT JOIN organization_embeddings e ON c.id = e.org_id
                    WHERE e.org_id IS NULL AND c.id > ? AND c.source = ?
                    ORDER BY c.id
                    LIMIT ?
                """, (last_id, source, batch_size))
            else:
                cursor = conn.execute("""
                    SELECT c.id, c.name FROM organizations c
                    LEFT JOIN organization_embeddings e ON c.id = e.org_id
                    WHERE e.org_id IS NULL AND c.id > ?
                    ORDER BY c.id
                    LIMIT ?
                """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield (row[0], row[1])
                last_id = row[0]

    def insert_embeddings_batch(
        self,
        org_ids: list[int],
        embeddings: np.ndarray,
    ) -> int:
        """
        Insert embeddings for existing organizations.

        Args:
            org_ids: List of organization IDs
            embeddings: Matrix of embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for org_id, embedding in zip(org_ids, embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, embedding_blob))
            count += 1

        conn.commit()
        return count

    def ensure_scalar_table_exists(self) -> None:
        """Create scalar embedding table if it doesn't exist."""
        conn = self._connect()
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS organization_embeddings_scalar USING vec0(
                org_id INTEGER PRIMARY KEY,
                embedding int8[{self._embedding_dim}] distance_metric=cosine
            )
        """)
        conn.commit()
        logger.info("Ensured organization_embeddings_scalar table exists")

    def rebuild_vec_tables(self) -> dict[str, int]:
        """
        Rebuild vec0 tables with distance_metric=cosine for indexed KNN search.

        Drops and recreates each vec0 table, copying embeddings through a temp table.
        Required once for databases created before the cosine metric was added.

        Returns:
            Dict with table names and row counts after rebuild.
        """
        conn = self._connect()
        stats: dict[str, int] = {}

        for table, id_col, dtype, vec_wrap in [
            ("organization_embeddings", "org_id", f"float[{self._embedding_dim}]", None),
            ("organization_embeddings_scalar", "org_id", f"int8[{self._embedding_dim}]", "vec_int8"),
        ]:
            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            if not cursor.fetchone():
                logger.info(f"Table {table} does not exist, skipping")
                stats[table] = 0
                continue

            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"Rebuilding {table} ({count} rows) with distance_metric=cosine...")

            # Dump to temp table
            conn.execute(f"CREATE TEMP TABLE _rebuild AS SELECT {id_col}, embedding FROM {table}")
            conn.execute(f"DROP TABLE {table}")

            # Recreate with cosine metric
            conn.execute(f"""
                CREATE VIRTUAL TABLE {table} USING vec0(
                    {id_col} INTEGER PRIMARY KEY,
                    embedding {dtype} distance_metric=cosine
                )
            """)

            # Copy back
            if vec_wrap:
                conn.execute(f"""
                    INSERT INTO {table} ({id_col}, embedding)
                    SELECT {id_col}, {vec_wrap}(embedding) FROM _rebuild
                """)
            else:
                conn.execute(f"""
                    INSERT INTO {table} ({id_col}, embedding)
                    SELECT {id_col}, embedding FROM _rebuild
                """)

            conn.execute("DROP TABLE _rebuild")
            stats[table] = count
            logger.info(f"Rebuilt {table}: {count} rows")

        conn.commit()
        return stats

    def get_missing_scalar_embedding_ids(self, batch_size: int = 1000) -> Iterator[list[int]]:
        """
        Yield batches of org IDs that have float32 but missing scalar embeddings.

        Args:
            batch_size: Number of IDs per batch

        Yields:
            Lists of org_ids needing scalar embeddings
        """
        conn = self._connect()

        # Ensure scalar table exists before querying
        self.ensure_scalar_table_exists()

        last_id = 0
        while True:
            cursor = conn.execute("""
                SELECT e.org_id FROM organization_embeddings e
                LEFT JOIN organization_embeddings_scalar s ON e.org_id = s.org_id
                WHERE s.org_id IS NULL AND e.org_id > ?
                ORDER BY e.org_id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            ids = [row["org_id"] for row in rows]
            yield ids
            last_id = ids[-1]

    def get_embeddings_by_ids(self, org_ids: list[int]) -> dict[int, np.ndarray]:
        """
        Fetch float32 embeddings for given org IDs.

        Args:
            org_ids: List of organization IDs

        Returns:
            Dict mapping org_id to float32 embedding array
        """
        conn = self._connect()

        if not org_ids:
            return {}

        placeholders = ",".join("?" * len(org_ids))
        cursor = conn.execute(f"""
            SELECT org_id, embedding FROM organization_embeddings
            WHERE org_id IN ({placeholders})
        """, org_ids)

        result = {}
        for row in cursor:
            embedding_blob = row["embedding"]
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            result[row["org_id"]] = embedding
        return result

    def insert_scalar_embeddings_batch(self, org_ids: list[int], embeddings: np.ndarray) -> int:
        """
        Insert scalar (int8) embeddings for existing orgs.

        Args:
            org_ids: List of organization IDs
            embeddings: Matrix of int8 embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for org_id, embedding in zip(org_ids, embeddings):
            scalar_blob = embedding.astype(np.int8).tobytes()
            conn.execute("DELETE FROM organization_embeddings_scalar WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings_scalar (org_id, embedding)
                VALUES (?, vec_int8(?))
            """, (org_id, scalar_blob))
            count += 1

        conn.commit()
        return count

    def get_scalar_embedding_count(self) -> int:
        """Get count of scalar embeddings."""
        conn = self._connect()
        if not self._has_scalar_table():
            return 0
        cursor = conn.execute("SELECT COUNT(*) FROM organization_embeddings_scalar")
        return cursor.fetchone()[0]

    def get_float32_embedding_count(self) -> int:
        """Get count of float32 embeddings."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM organization_embeddings")
        return cursor.fetchone()[0]

    def get_missing_all_embedding_ids(self, batch_size: int = 1000) -> Iterator[list[tuple[int, str]]]:
        """
        Yield batches of (org_id, name) tuples for records missing both float32 and scalar embeddings.

        Args:
            batch_size: Number of IDs per batch

        Yields:
            Lists of (org_id, name) tuples needing embeddings generated from scratch
        """
        conn = self._connect()

        # Ensure scalar table exists
        self.ensure_scalar_table_exists()

        last_id = 0
        while True:
            cursor = conn.execute("""
                SELECT o.id, o.name FROM organizations o
                LEFT JOIN organization_embeddings e ON o.id = e.org_id
                WHERE e.org_id IS NULL AND o.id > ?
                ORDER BY o.id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            results = [(row["id"], row["name"]) for row in rows]
            yield results
            last_id = results[-1][0]

    def insert_both_embeddings_batch(
        self,
        org_ids: list[int],
        fp32_embeddings: np.ndarray,
        int8_embeddings: np.ndarray,
    ) -> int:
        """
        Insert both float32 and int8 embeddings for existing orgs.

        Args:
            org_ids: List of organization IDs
            fp32_embeddings: Matrix of float32 embeddings (N x dim)
            int8_embeddings: Matrix of int8 embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for org_id, fp32, int8 in zip(org_ids, fp32_embeddings, int8_embeddings):
            # Insert float32
            fp32_blob = fp32.astype(np.float32).tobytes()
            conn.execute("DELETE FROM organization_embeddings WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings (org_id, embedding)
                VALUES (?, ?)
            """, (org_id, fp32_blob))

            # Insert int8
            int8_blob = int8.astype(np.int8).tobytes()
            conn.execute("DELETE FROM organization_embeddings_scalar WHERE org_id = ?", (org_id,))
            conn.execute("""
                INSERT INTO organization_embeddings_scalar (org_id, embedding)
                VALUES (?, vec_int8(?))
            """, (org_id, int8_blob))

            count += 1

        conn.commit()
        return count

    def resolve_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> tuple[int, int]:
        """
        Update organization records that have QIDs instead of labels in region field.

        If resolving would create a duplicate of an existing record with
        resolved labels, the QID version is deleted instead.

        Args:
            label_map: Mapping of QID -> label for resolution
            batch_size: Commit batch size

        Returns:
            Tuple of (records updated, duplicates deleted)
        """
        conn = self._connect()

        # v2 schema uses region_id (FK to locations) instead of region text - this method doesn't apply
        if self._is_v2:
            logger.debug("Skipping resolve_qid_labels for v2 schema (region stored as FK)")
            return 0, 0

        # Find records with QIDs in region field (starts with 'Q' followed by digits)
        region_updates = 0
        cursor = conn.execute("""
            SELECT id, region FROM organizations
            WHERE region LIKE 'Q%' AND region GLOB 'Q[0-9]*'
        """)
        rows = cursor.fetchall()

        duplicates_deleted = 0
        for row in rows:
            org_id = row["id"]
            qid = row["region"]
            if qid in label_map:
                resolved_region = label_map[qid]
                # Check if this update would create a duplicate
                # Get the name and source of the current record
                org_cursor = conn.execute(
                    "SELECT name, source FROM organizations WHERE id = ?",
                    (org_id,)
                )
                org_row = org_cursor.fetchone()
                if org_row is None:
                    continue

                org_name = org_row["name"]
                org_source = org_row["source"]

                # Check if a record with the resolved region already exists
                existing_cursor = conn.execute(
                    "SELECT id FROM organizations WHERE name = ? AND region = ? AND source = ? AND id != ?",
                    (org_name, resolved_region, org_source, org_id)
                )
                existing = existing_cursor.fetchone()

                if existing is not None:
                    # Duplicate would be created - delete the QID-based record
                    conn.execute("DELETE FROM organizations WHERE id = ?", (org_id,))
                    duplicates_deleted += 1
                else:
                    # Safe to update
                    conn.execute(
                        "UPDATE organizations SET region = ? WHERE id = ?",
                        (resolved_region, org_id)
                    )
                    region_updates += 1

                if (region_updates + duplicates_deleted) % batch_size == 0:
                    conn.commit()
                    logger.info(f"Resolved QID labels: {region_updates} updates, {duplicates_deleted} deletes...")

        conn.commit()
        logger.info(f"Resolved QID labels: {region_updates} organization regions, {duplicates_deleted} duplicates deleted")
        return region_updates, duplicates_deleted

    def get_unresolved_qids(self) -> set[str]:
        """
        Get all QIDs that still need resolution in the organizations table.

        Returns:
            Set of QIDs (starting with 'Q') found in region field
        """
        # v2 schema uses region_id (FK to locations) instead of region text - no QIDs to resolve
        if self._is_v2:
            return set()

        conn = self._connect()
        qids: set[str] = set()

        cursor = conn.execute("""
            SELECT DISTINCT region FROM organizations
            WHERE region LIKE 'Q%' AND region GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["region"])

        return qids


def get_person_database(
    db_path: Optional[str | Path] = None, embedding_dim: int = 768, readonly: bool = True
) -> "PersonDatabase":
    """
    Get a singleton PersonDatabase instance for the given path.

    Args:
        db_path: Path to database file
        embedding_dim: Dimension of embeddings
        readonly: If True (default), open in read-only mode.
                  For write operations, create a PersonDatabase directly with readonly=False.

    Returns:
        Shared PersonDatabase instance (readonly by default)
    """
    path_key = str(db_path or DEFAULT_DB_PATH) + (":ro" if readonly else ":rw")
    if path_key not in _person_database_instances:
        logger.debug(f"Creating new PersonDatabase instance for {path_key}")
        _person_database_instances[path_key] = PersonDatabase(
            db_path=db_path, embedding_dim=embedding_dim, readonly=readonly
        )
    return _person_database_instances[path_key]


class PersonDatabase:
    """
    SQLite database with sqlite-vec for person vector search.

    Uses hybrid text + vector search:
    1. Text filtering with LIKE to reduce candidates
    2. sqlite-vec for semantic similarity ranking

    Stores people from sources like Wikidata with role/org context.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_dim: int = 768,  # Default for embeddinggemma-300m
        readonly: bool = True,
    ):
        """
        Initialize the person database.

        Args:
            db_path: Path to database file (creates if not exists)
            embedding_dim: Dimension of embeddings to store
            readonly: If True (default), open in read-only mode (avoids locking).
                      Set to False for import operations.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._embedding_dim = embedding_dim
        self._readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None
        self._is_v2: Optional[bool] = None  # Detected on first connect
        self._schema_version: int = 1  # Updated on connect (1=legacy, 2=normalized FKs, 3=no embeddings in lite)
        # Cached flags
        self._has_scalar_cached: Optional[bool] = None
        # USearch index for fast approximate nearest neighbor search
        self._hnsw_index: Optional[Any] = None  # usearch.index.Index

    def _ensure_dir(self) -> None:
        """Ensure database directory exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, self._embedding_dim, self._readonly)

        # Detect schema version BEFORE creating tables
        # v2 has person_type_id (FK) instead of person_type (TEXT)
        if self._is_v2 is None:
            cursor = self._conn.execute("PRAGMA table_info(people)")
            columns = {row["name"] for row in cursor}
            self._is_v2 = "person_type_id" in columns
            if self._is_v2:
                self._schema_version = 2
                logger.debug("Detected v2 schema for people")

            # Check for v3 metadata table
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='db_info'"
            )
            if cursor.fetchone():
                row = self._conn.execute(
                    "SELECT value FROM db_info WHERE key = 'schema_version'"
                ).fetchone()
                if row:
                    self._schema_version = int(row[0])
                    logger.debug(f"Detected schema version {self._schema_version} from db_info")

        # Create tables (idempotent) - only for v1 schema or fresh databases
        # v2 databases already have their schema from migration
        if not self._is_v2 and not self._readonly:
            self._create_tables()

        return self._conn

    @property
    def _people_table(self) -> str:
        """Return table/view name for people queries needing text fields."""
        return "people_view" if self._is_v2 else "people"

    def _create_tables(self) -> None:
        """Create database tables including sqlite-vec virtual table."""
        conn = self._conn
        assert conn is not None

        # Check if we need to migrate from old schema (unique on source+source_id only)
        self._migrate_people_schema_if_needed(conn)

        # Main people records table
        # Unique constraint on source+source_id+role+org allows multiple records
        # for the same person with different role/org combinations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'wikidata',
                source_id TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                person_type TEXT NOT NULL DEFAULT 'unknown',
                known_for_role TEXT NOT NULL DEFAULT '',
                known_for_org TEXT NOT NULL DEFAULT '',
                known_for_org_id INTEGER DEFAULT NULL,
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                birth_date TEXT NOT NULL DEFAULT '',
                death_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id, known_for_role, known_for_org),
                FOREIGN KEY (known_for_org_id) REFERENCES organizations(id)
            )
        """)

        # Create indexes on main table
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_name_normalized ON people(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source ON people(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_source_id ON people(source, source_id, known_for_role, known_for_org)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_people_known_for_org ON people(known_for_org)")

        # Add from_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN from_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added from_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add to_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN to_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added to_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add known_for_org_id column if it doesn't exist (migration for existing DBs)
        # This is a foreign key to the organizations table (nullable)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN known_for_org_id INTEGER DEFAULT NULL")
            logger.info("Added known_for_org_id column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index on known_for_org_id for joins (only if column exists)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_people_known_for_org_id ON people(known_for_org_id)")
        except sqlite3.OperationalError:
            pass  # Column doesn't exist yet (will be added on next connection)

        # Add birth_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN birth_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added birth_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add death_date column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN death_date TEXT NOT NULL DEFAULT ''")
            logger.info("Added death_date column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_id column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN canon_id INTEGER DEFAULT NULL")
            logger.info("Added canon_id column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add canon_size column if it doesn't exist (migration for canonicalization)
        try:
            conn.execute("ALTER TABLE people ADD COLUMN canon_size INTEGER DEFAULT 1")
            logger.info("Added canon_size column to people table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index on canon_id for joins
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_people_canon_id ON people(canon_id)")
        except sqlite3.OperationalError:
            pass  # Column doesn't exist yet

        # Create sqlite-vec virtual table for embeddings (float32)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS person_embeddings USING vec0(
                person_id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dim}] distance_metric=cosine
            )
        """)

        # Create sqlite-vec virtual table for scalar embeddings (int8)
        # Provides 75% storage reduction with ~92% recall at top-100
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS person_embeddings_scalar USING vec0(
                person_id INTEGER PRIMARY KEY,
                embedding int8[{self._embedding_dim}] distance_metric=cosine
            )
        """)

        # Create QID labels lookup table for Wikidata QID -> label mappings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qid_labels (
                qid TEXT PRIMARY KEY,
                label TEXT NOT NULL
            )
        """)

        conn.commit()

    def _migrate_people_schema_if_needed(self, conn: sqlite3.Connection) -> None:
        """Migrate people table from old schema if needed."""
        # Check if people table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
        )
        if not cursor.fetchone():
            return  # Table doesn't exist, no migration needed

        # Check the unique constraint - look at index info
        # Old schema: UNIQUE(source, source_id)
        # New schema: UNIQUE(source, source_id, known_for_role, known_for_org)
        cursor = conn.execute("PRAGMA index_list(people)")
        indexes = cursor.fetchall()

        needs_migration = False
        for idx in indexes:
            idx_name = idx[1]
            if "sqlite_autoindex_people" in idx_name:
                # Check columns in this unique index
                cursor = conn.execute(f"PRAGMA index_info('{idx_name}')")
                cols = [row[2] for row in cursor.fetchall()]
                # Old schema has only 2 columns in unique constraint
                if cols == ["source", "source_id"]:
                    needs_migration = True
                    logger.info("Detected old people schema, migrating to new unique constraint...")
                    break

        if not needs_migration:
            return

        # Migrate: create new table, copy data, drop old, rename new
        logger.info("Migrating people table to new schema with (source, source_id, role, org) unique constraint...")

        conn.execute("""
            CREATE TABLE people_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'wikidata',
                source_id TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                person_type TEXT NOT NULL DEFAULT 'unknown',
                known_for_role TEXT NOT NULL DEFAULT '',
                known_for_org TEXT NOT NULL DEFAULT '',
                known_for_org_id INTEGER DEFAULT NULL,
                from_date TEXT NOT NULL DEFAULT '',
                to_date TEXT NOT NULL DEFAULT '',
                record TEXT NOT NULL,
                UNIQUE(source, source_id, known_for_role, known_for_org),
                FOREIGN KEY (known_for_org_id) REFERENCES organizations(id)
            )
        """)

        # Copy data (old IDs will change, but embeddings table references them)
        # Note: old table may not have from_date/to_date columns, so use defaults
        conn.execute("""
            INSERT INTO people_new (name, name_normalized, source, source_id, country,
                                    person_type, known_for_role, known_for_org, record)
            SELECT name, name_normalized, source, source_id, country,
                   person_type, known_for_role, known_for_org, record
            FROM people
        """)

        # Drop old table and embeddings (IDs changed, embeddings are invalid)
        conn.execute("DROP TABLE IF EXISTS person_embeddings")
        conn.execute("DROP TABLE people")
        conn.execute("ALTER TABLE people_new RENAME TO people")

        # Drop old index if it exists
        conn.execute("DROP INDEX IF EXISTS idx_people_source_id")

        conn.commit()
        logger.info("Migration complete. Note: person embeddings were cleared and need to be regenerated.")

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(
        self,
        record: PersonRecord,
        embedding: np.ndarray,
        scalar_embedding: Optional[np.ndarray] = None,
    ) -> int:
        """
        Insert a person record with its embedding.

        Args:
            record: Person record to insert
            embedding: Embedding vector for the person name (float32)
            scalar_embedding: Optional int8 scalar embedding for compact storage

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_person_name(record.name)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO people
            (name, name_normalized, source, source_id, country, person_type,
             known_for_role, known_for_org, known_for_org_id, from_date, to_date,
             birth_date, death_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            record.source,
            record.source_id,
            record.country,
            record.person_type.value,
            record.known_for_role,
            record.known_for_org,
            record.known_for_org_id,  # Can be None
            record.from_date or "",
            record.to_date or "",
            record.birth_date or "",
            record.death_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        # Insert embedding into vec table (float32)
        embedding_blob = embedding.astype(np.float32).tobytes()
        conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (row_id,))
        conn.execute("""
            INSERT INTO person_embeddings (person_id, embedding)
            VALUES (?, ?)
        """, (row_id, embedding_blob))

        # Insert scalar embedding if provided (int8)
        if scalar_embedding is not None:
            scalar_blob = scalar_embedding.astype(np.int8).tobytes()
            conn.execute("DELETE FROM person_embeddings_scalar WHERE person_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO person_embeddings_scalar (person_id, embedding)
                VALUES (?, vec_int8(?))
            """, (row_id, scalar_blob))

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[PersonRecord],
        embeddings: np.ndarray,
        batch_size: int = 1000,
        scalar_embeddings: Optional[np.ndarray] = None,
    ) -> int:
        """
        Insert multiple person records with embeddings.

        Args:
            records: List of person records
            embeddings: Matrix of embeddings (N x dim) - float32
            batch_size: Commit batch size
            scalar_embeddings: Optional matrix of int8 scalar embeddings (N x dim)

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for i, (record, embedding) in enumerate(zip(records, embeddings)):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_person_name(record.name)

            if self._is_v2:
                # v2 schema: use FK IDs instead of TEXT columns
                source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
                person_type_id = PEOPLE_TYPE_NAME_TO_ID.get(record.person_type.value, 15)  # 15 = unknown

                # Resolve country to location_id if provided
                country_id = None
                if record.country:
                    # Use readonly=False to avoid immutable mode conflicts with write connection
                    locations_db = get_locations_database(db_path=self._db_path, readonly=False)
                    country_id = locations_db.resolve_region_text(record.country)

                # Resolve known_for_role to role_id if provided
                role_id = None
                if record.known_for_role:
                    roles_db = get_roles_database(db_path=self._db_path, readonly=False)
                    role_id = roles_db.get_or_create(record.known_for_role, source_id=source_type_id)

                cursor = conn.execute("""
                    INSERT OR REPLACE INTO people
                    (name, name_normalized, source_id, source_identifier, country_id, person_type_id,
                     known_for_role_id, known_for_org, known_for_org_id, from_date, to_date,
                     birth_date, death_date, record)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.name,
                    name_normalized,
                    source_type_id,
                    record.source_id,
                    country_id,
                    person_type_id,
                    role_id,
                    record.known_for_org,
                    record.known_for_org_id,  # Can be None
                    record.from_date or "",
                    record.to_date or "",
                    record.birth_date or "",
                    record.death_date or "",
                    record_json,
                ))
            else:
                # v1 schema: use TEXT columns
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO people
                    (name, name_normalized, source, source_id, country, person_type,
                     known_for_role, known_for_org, known_for_org_id, from_date, to_date,
                     birth_date, death_date, record)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.name,
                    name_normalized,
                    record.source,
                    record.source_id,
                    record.country,
                    record.person_type.value,
                    record.known_for_role,
                    record.known_for_org,
                    record.known_for_org_id,  # Can be None
                    record.from_date or "",
                    record.to_date or "",
                    record.birth_date or "",
                    record.death_date or "",
                    record_json,
                ))

            row_id = cursor.lastrowid
            assert row_id is not None

            # Insert embedding (delete first since sqlite-vec doesn't support REPLACE)
            embedding_blob = embedding.astype(np.float32).tobytes()
            conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (row_id,))
            conn.execute("""
                INSERT INTO person_embeddings (person_id, embedding)
                VALUES (?, ?)
            """, (row_id, embedding_blob))

            # Insert scalar embedding if provided (int8)
            if scalar_embeddings is not None:
                scalar_blob = scalar_embeddings[i].astype(np.int8).tobytes()
                conn.execute("DELETE FROM person_embeddings_scalar WHERE person_id = ?", (row_id,))
                conn.execute("""
                    INSERT INTO person_embeddings_scalar (person_id, embedding)
                    VALUES (?, vec_int8(?))
                """, (row_id, scalar_blob))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} person records...")

        conn.commit()
        return count

    def update_dates(self, source: str, source_id: str, from_date: Optional[str], to_date: Optional[str]) -> bool:
        """
        Update the from_date and to_date for a person record.

        Args:
            source: Data source (e.g., 'wikidata')
            source_id: Source identifier (e.g., QID)
            from_date: Start date in ISO format or None
            to_date: End date in ISO format or None

        Returns:
            True if record was updated, False if not found
        """
        conn = self._connect()

        if self._is_v2:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute("""
                UPDATE people SET from_date = ?, to_date = ?
                WHERE source_id = ? AND source_identifier = ?
            """, (from_date or "", to_date or "", source_type_id, source_id))
        else:
            cursor = conn.execute("""
                UPDATE people SET from_date = ?, to_date = ?
                WHERE source = ? AND source_id = ?
            """, (from_date or "", to_date or "", source, source_id))

        conn.commit()
        return cursor.rowcount > 0

    def update_role_org(
        self,
        source: str,
        source_id: str,
        known_for_role: str,
        known_for_org: str,
        known_for_org_id: Optional[int],
        new_embedding: np.ndarray,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> bool:
        """
        Update the role/org/dates data for a person record and re-embed.

        Args:
            source: Data source (e.g., 'wikidata')
            source_id: Source identifier (e.g., QID)
            known_for_role: Role/position title
            known_for_org: Organization name
            known_for_org_id: Organization internal ID (FK) or None
            new_embedding: New embedding vector based on updated data
            from_date: Start date in ISO format or None
            to_date: End date in ISO format or None

        Returns:
            True if record was updated, False if not found
        """
        conn = self._connect()

        # First get the person's internal ID
        if self._is_v2:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            row = conn.execute(
                "SELECT id FROM people WHERE source_id = ? AND source_identifier = ?",
                (source_type_id, source_id)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM people WHERE source = ? AND source_id = ?",
                (source, source_id)
            ).fetchone()

        if not row:
            return False

        person_id = row[0]

        # Update the person record (including dates)
        conn.execute("""
            UPDATE people SET
                known_for_role = ?, known_for_org = ?, known_for_org_id = ?,
                from_date = COALESCE(?, from_date, ''),
                to_date = COALESCE(?, to_date, '')
            WHERE id = ?
        """, (known_for_role, known_for_org, known_for_org_id, from_date, to_date, person_id))

        # Update the embedding
        embedding_bytes = new_embedding.astype(np.float32).tobytes()
        conn.execute("""
            UPDATE people_vec SET embedding = ?
            WHERE rowid = ?
        """, (embedding_bytes, person_id))

        conn.commit()
        return True

    def backfill_known_for_org(
        self, qid_to_orgs: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]]
    ) -> int:
        """
        Backfill known_for_org (and dates) for people using QID→org mappings.

        Only updates records where known_for_org is currently empty. Uses the
        first reverse mapping entry for each person QID to fill in org label,
        role, and from/to dates.

        Args:
            qid_to_orgs: Mapping of person Wikidata QID →
                [(org_label, role_description, start_date, end_date), ...]

        Returns:
            Number of records updated
        """
        if not qid_to_orgs:
            return 0

        conn = self._connect()
        updated = 0
        for person_qid, entries in qid_to_orgs.items():
            if not entries:
                continue
            # Use the first entry with an org label
            org_label = ""
            role = ""
            start_date: Optional[str] = None
            end_date: Optional[str] = None
            for entry_org, entry_role, entry_start, entry_end in entries:
                if entry_org:
                    org_label = entry_org
                    role = entry_role
                    start_date = entry_start
                    end_date = entry_end
                    break
            if not org_label:
                continue

            # Strip Q prefix if present for integer QID column
            qid_int = int(person_qid.lstrip("Q")) if person_qid.startswith("Q") else None
            if qid_int is None:
                continue

            # Resolve role to role_id for v2 schema
            role_id = None
            if role and self._is_v2:
                roles_db = get_roles_database(db_path=self._db_path, readonly=False)
                role_id = roles_db.get_or_create(role, source_id=4)  # 4 = wikidata

            # Update records missing known_for_org, also backfill dates and role
            if self._is_v2:
                result = conn.execute(
                    """UPDATE people SET known_for_org = ?,
                       known_for_role_id = COALESCE(known_for_role_id, ?),
                       from_date = CASE WHEN from_date IS NULL OR from_date = '' THEN ? ELSE from_date END,
                       to_date = CASE WHEN to_date IS NULL OR to_date = '' THEN ? ELSE to_date END
                    WHERE qid = ? AND (known_for_org = '' OR known_for_org IS NULL)""",
                    (org_label, role_id, start_date or "", end_date or "", qid_int),
                )
            else:
                result = conn.execute(
                    """UPDATE people SET known_for_org = ?,
                       known_for_role = CASE WHEN known_for_role = '' OR known_for_role IS NULL THEN ? ELSE known_for_role END,
                       from_date = CASE WHEN from_date IS NULL OR from_date = '' THEN ? ELSE from_date END,
                       to_date = CASE WHEN to_date IS NULL OR to_date = '' THEN ? ELSE to_date END
                    WHERE source_id = ? AND source = 'wikidata' AND (known_for_org = '' OR known_for_org IS NULL)""",
                    (org_label, role, start_date or "", end_date or "", person_qid),
                )
            updated += result.rowcount
            if updated % 1000 == 0 and updated > 0:
                conn.commit()
        conn.commit()
        return updated

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        query_text: Optional[str] = None,
        max_text_candidates: int = 5000,
    ) -> list[tuple[PersonRecord, float]]:
        """
        Search for similar people using hybrid text + vector search.

        Two-stage approach:
        1. If query_text provided, use SQL LIKE to find candidates containing search terms
        2. USearch HNSW index for fast approximate nearest neighbor search

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            query_text: Optional query text for text-based pre-filtering
            max_text_candidates: Max candidates to keep after text filtering

        Returns:
            List of (PersonRecord, similarity_score) tuples
        """
        start = time.time()
        self._connect()

        # Log search mode
        if query_text:
            logger.info(f"Person search: hybrid mode (text + embeddings) for '{query_text}'")
        else:
            logger.debug("Person search: embeddings-only mode (fast KNN)")

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Stage 1: Text filtering (if query_text provided)
        candidate_ids: Optional[set[int]] = None
        if query_text:
            query_normalized_text = _normalize_person_name(query_text)
            if query_normalized_text:
                candidate_ids = self._text_filter_candidates(
                    query_normalized_text,
                    max_candidates=max_text_candidates,
                )
                logger.info(f"Text filter: {len(candidate_ids)} candidates for '{query_text}'")
                if len(candidate_ids) == 0:
                    return []

        # Stage 2: Vector ranking via USearch
        query_int8 = np.clip(np.round(query_normalized * 127), -127, 127).astype(np.int8)

        if not self._load_hnsw_index():
            raise RuntimeError("USearch index not found. Run: corp-extractor db build-index")

        results = self._hnsw_search(query_int8, top_k, candidate_ids)

        elapsed = time.time() - start
        logger.debug(f"Person search took {elapsed:.3f}s (results={len(results)})")
        return results

    def _text_filter_candidates(
        self,
        query_normalized: str,
        max_candidates: int,
    ) -> set[int]:
        """
        Filter candidates using SQL LIKE for fast text matching.

        Uses `name_normalized` column for consistent matching.
        """
        conn = self._conn
        assert conn is not None

        # Extract search terms from the normalized query
        search_terms = _extract_search_terms(query_normalized)
        if not search_terms:
            return set()

        logger.debug(f"Person text filter search terms: {search_terms}")

        # Build OR clause for LIKE matching on any term
        like_clauses = []
        params: list = []
        for term in search_terms:
            like_clauses.append("name_normalized LIKE ?")
            params.append(f"%{term}%")

        where_clause = " OR ".join(like_clauses)

        query = f"""
            SELECT id FROM people
            WHERE {where_clause}
            LIMIT ?
        """

        params.append(max_candidates)

        cursor = conn.execute(query, params)
        return set(row["id"] for row in cursor)

    def _has_scalar_table(self) -> bool:
        """Check if scalar embedding table exists (cached)."""
        if self._has_scalar_cached is not None:
            return self._has_scalar_cached
        conn = self._conn
        assert conn is not None
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='person_embeddings_scalar'"
        )
        result = cursor.fetchone() is not None
        self._has_scalar_cached = result
        return result

    # --- USearch approximate nearest neighbor search ---

    def _get_hnsw_index_path(self) -> Path:
        """Get path to USearch index file."""
        return self._db_path.parent / "people_usearch.bin"

    def _load_hnsw_index(self) -> bool:
        """
        Load pre-built USearch index from disk.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if self._hnsw_index is not None:
            return True  # Already loaded

        index_path = self._get_hnsw_index_path()
        if not index_path.exists():
            logger.debug(f"USearch index not found: {index_path}")
            return False

        try:
            from usearch.index import Index

            # Load index (IDs are stored in the index itself)
            logger.info(f"Loading USearch index from {index_path.name}...")
            index = Index.restore(str(index_path))
            # USearch doesn't persist expansion_search — restore it for good recall on large indexes
            index.expansion_search = 200
            self._hnsw_index = index
            logger.info(f"Loaded USearch index: {len(index):,} vectors, connectivity={index.connectivity}, ef_search={index.expansion_search}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load USearch index: {e}")
            return False

    def _hnsw_search(
        self,
        query_int8: np.ndarray,
        top_k: int,
        candidate_ids: Optional[set[int]] = None,
    ) -> list[tuple[PersonRecord, float]]:
        """
        Search using USearch index.

        Args:
            query_int8: Query embedding quantized to int8
            top_k: Number of results to return
            candidate_ids: Optional set of candidate IDs to filter results

        Returns:
            List of (PersonRecord, similarity) tuples
        """
        if self._hnsw_index is None:
            return []

        # Query USearch index (fetch more if we need to filter)
        fetch_k = top_k * 10 if candidate_ids else top_k
        fetch_k = min(fetch_k, len(self._hnsw_index))

        matches = self._hnsw_index.search(query_int8, fetch_k)

        # Convert to results
        results: list[tuple[PersonRecord, float]] = []
        for person_id, dist in zip(matches.keys, matches.distances):
            person_id = int(person_id)

            # Filter if needed
            if candidate_ids and person_id not in candidate_ids:
                continue

            # Convert distance to similarity (cosine distance is 1 - cosine similarity)
            similarity = 1.0 - float(dist)

            record = self._get_record_by_id(person_id)
            if record is not None:
                results.append((record, similarity))

            if len(results) >= top_k:
                break

        return results

    def _get_record_by_id(self, person_id: int) -> Optional[PersonRecord]:
        """Get a person record by ID."""
        conn = self._conn
        assert conn is not None

        if self._is_v2:
            # v2 schema: join view with base table for record
            cursor = conn.execute("""
                SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                       v.known_for_role, v.known_for_org, v.known_for_org_id,
                       v.birth_date, v.death_date, p.record
                FROM people_view v
                JOIN people p ON v.id = p.id
                WHERE v.id = ?
            """, (person_id,))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                FROM people WHERE id = ?
            """, (person_id,))

        row = cursor.fetchone()
        if row:
            source_id_field = "source_identifier" if self._is_v2 else "source_id"
            return PersonRecord(
                name=row["name"],
                source=row["source"],
                source_id=row[source_id_field],
                country=row["country"] or "",
                person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                known_for_role=row["known_for_role"] or "",
                known_for_org=row["known_for_org"] or "",
                known_for_org_id=row["known_for_org_id"],  # Can be None
                birth_date=row["birth_date"] or "",
                death_date=row["death_date"] or "",
                record=json.loads(row["record"]),
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[PersonRecord]:
        """Get a person record by source and source_id."""
        conn = self._connect()

        if self._is_v2:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute("""
                SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                       v.known_for_role, v.known_for_org, v.known_for_org_id,
                       v.birth_date, v.death_date, p.record
                FROM people_view v
                JOIN people p ON v.id = p.id
                WHERE p.source_id = ? AND p.source_identifier = ?
            """, (source_type_id, source_id))
        else:
            cursor = conn.execute("""
                SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                FROM people
                WHERE source = ? AND source_id = ?
            """, (source, source_id))

        row = cursor.fetchone()
        if row:
            source_id_field = "source_identifier" if self._is_v2 else "source_id"
            return PersonRecord(
                name=row["name"],
                source=row["source"],
                source_id=row[source_id_field],
                country=row["country"] or "",
                person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                known_for_role=row["known_for_role"] or "",
                known_for_org=row["known_for_org"] or "",
                known_for_org_id=row["known_for_org_id"],  # Can be None
                birth_date=row["birth_date"] or "",
                death_date=row["death_date"] or "",
                record=json.loads(row["record"]),
            )
        return None

    def get_stats(self) -> dict:
        """Get database statistics for people table."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM people")
        total = cursor.fetchone()[0]

        # Count by person_type - handle both v1 and v2 schema
        if self._is_v2:
            # v2 schema - join with people_types
            cursor = conn.execute("""
                SELECT pt.name as person_type, COUNT(*) as cnt
                FROM people p
                JOIN people_types pt ON p.person_type_id = pt.id
                GROUP BY p.person_type_id
            """)
        else:
            # v1 schema
            cursor = conn.execute("SELECT person_type, COUNT(*) as cnt FROM people GROUP BY person_type")
        by_type = {row["person_type"]: row["cnt"] for row in cursor}

        # Count by source - handle both v1 and v2 schema
        if self._is_v2:
            # v2 schema - join with source_types
            cursor = conn.execute("""
                SELECT st.name as source, COUNT(*) as cnt
                FROM people p
                JOIN source_types st ON p.source_id = st.id
                GROUP BY p.source_id
            """)
        else:
            # v1 schema
            cursor = conn.execute("SELECT source, COUNT(*) as cnt FROM people GROUP BY source")
        by_source = {row["source"]: row["cnt"] for row in cursor}

        return {
            "total_records": total,
            "by_type": by_type,
            "by_source": by_source,
        }

    def ensure_scalar_table_exists(self) -> None:
        """Create scalar embedding table if it doesn't exist."""
        conn = self._connect()
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS person_embeddings_scalar USING vec0(
                person_id INTEGER PRIMARY KEY,
                embedding int8[{self._embedding_dim}] distance_metric=cosine
            )
        """)
        conn.commit()
        logger.info("Ensured person_embeddings_scalar table exists")

    def rebuild_vec_tables(self) -> dict[str, int]:
        """
        Rebuild vec0 tables with distance_metric=cosine for indexed KNN search.

        Drops and recreates each vec0 table, copying embeddings through a temp table.
        Required once for databases created before the cosine metric was added.

        Returns:
            Dict with table names and row counts after rebuild.
        """
        conn = self._connect()
        stats: dict[str, int] = {}

        for table, id_col, dtype, vec_wrap in [
            ("person_embeddings", "person_id", f"float[{self._embedding_dim}]", None),
            ("person_embeddings_scalar", "person_id", f"int8[{self._embedding_dim}]", "vec_int8"),
        ]:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            if not cursor.fetchone():
                logger.info(f"Table {table} does not exist, skipping")
                stats[table] = 0
                continue

            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"Rebuilding {table} ({count} rows) with distance_metric=cosine...")

            conn.execute(f"CREATE TEMP TABLE _rebuild AS SELECT {id_col}, embedding FROM {table}")
            conn.execute(f"DROP TABLE {table}")

            conn.execute(f"""
                CREATE VIRTUAL TABLE {table} USING vec0(
                    {id_col} INTEGER PRIMARY KEY,
                    embedding {dtype} distance_metric=cosine
                )
            """)

            if vec_wrap:
                conn.execute(f"""
                    INSERT INTO {table} ({id_col}, embedding)
                    SELECT {id_col}, {vec_wrap}(embedding) FROM _rebuild
                """)
            else:
                conn.execute(f"""
                    INSERT INTO {table} ({id_col}, embedding)
                    SELECT {id_col}, embedding FROM _rebuild
                """)

            conn.execute("DROP TABLE _rebuild")
            stats[table] = count
            logger.info(f"Rebuilt {table}: {count} rows")

        conn.commit()
        return stats

    def get_missing_scalar_embedding_ids(self, batch_size: int = 1000) -> Iterator[list[int]]:
        """
        Yield batches of person IDs that have float32 but missing scalar embeddings.

        Args:
            batch_size: Number of IDs per batch

        Yields:
            Lists of person_ids needing scalar embeddings
        """
        conn = self._connect()

        # Ensure scalar table exists before querying
        self.ensure_scalar_table_exists()

        last_id = 0
        while True:
            cursor = conn.execute("""
                SELECT e.person_id FROM person_embeddings e
                LEFT JOIN person_embeddings_scalar s ON e.person_id = s.person_id
                WHERE s.person_id IS NULL AND e.person_id > ?
                ORDER BY e.person_id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            ids = [row["person_id"] for row in rows]
            yield ids
            last_id = ids[-1]

    def get_embeddings_by_ids(self, person_ids: list[int]) -> dict[int, np.ndarray]:
        """
        Fetch float32 embeddings for given person IDs.

        Args:
            person_ids: List of person IDs

        Returns:
            Dict mapping person_id to float32 embedding array
        """
        conn = self._connect()

        if not person_ids:
            return {}

        placeholders = ",".join("?" * len(person_ids))
        cursor = conn.execute(f"""
            SELECT person_id, embedding FROM person_embeddings
            WHERE person_id IN ({placeholders})
        """, person_ids)

        result = {}
        for row in cursor:
            embedding_blob = row["embedding"]
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            result[row["person_id"]] = embedding
        return result

    def insert_scalar_embeddings_batch(self, person_ids: list[int], embeddings: np.ndarray) -> int:
        """
        Insert scalar (int8) embeddings for existing people.

        Args:
            person_ids: List of person IDs
            embeddings: Matrix of int8 embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for person_id, embedding in zip(person_ids, embeddings):
            scalar_blob = embedding.astype(np.int8).tobytes()
            conn.execute("DELETE FROM person_embeddings_scalar WHERE person_id = ?", (person_id,))
            conn.execute("""
                INSERT INTO person_embeddings_scalar (person_id, embedding)
                VALUES (?, vec_int8(?))
            """, (person_id, scalar_blob))
            count += 1

        conn.commit()
        return count

    def get_scalar_embedding_count(self) -> int:
        """Get count of scalar embeddings."""
        conn = self._connect()
        if not self._has_scalar_table():
            return 0
        cursor = conn.execute("SELECT COUNT(*) FROM person_embeddings_scalar")
        return cursor.fetchone()[0]

    def get_float32_embedding_count(self) -> int:
        """Get count of float32 embeddings."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM person_embeddings")
        return cursor.fetchone()[0]

    def get_missing_all_embedding_ids(self, batch_size: int = 1000) -> Iterator[list[tuple[int, str]]]:
        """
        Yield batches of (person_id, name) tuples for records missing both float32 and scalar embeddings.

        Args:
            batch_size: Number of IDs per batch

        Yields:
            Lists of (person_id, name) tuples needing embeddings generated from scratch
        """
        conn = self._connect()

        # Ensure scalar table exists
        self.ensure_scalar_table_exists()

        last_id = 0
        while True:
            cursor = conn.execute("""
                SELECT p.id, p.name FROM people p
                LEFT JOIN person_embeddings e ON p.id = e.person_id
                WHERE e.person_id IS NULL AND p.id > ?
                ORDER BY p.id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            results = [(row["id"], row["name"]) for row in rows]
            yield results
            last_id = results[-1][0]

    def insert_both_embeddings_batch(
        self,
        person_ids: list[int],
        fp32_embeddings: np.ndarray,
        int8_embeddings: np.ndarray,
    ) -> int:
        """
        Insert both float32 and int8 embeddings for existing people.

        Args:
            person_ids: List of person IDs
            fp32_embeddings: Matrix of float32 embeddings (N x dim)
            int8_embeddings: Matrix of int8 embeddings (N x dim)

        Returns:
            Number of embeddings inserted
        """
        conn = self._connect()
        count = 0

        for person_id, fp32, int8 in zip(person_ids, fp32_embeddings, int8_embeddings):
            # Insert float32
            fp32_blob = fp32.astype(np.float32).tobytes()
            conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (person_id,))
            conn.execute("""
                INSERT INTO person_embeddings (person_id, embedding)
                VALUES (?, ?)
            """, (person_id, fp32_blob))

            # Insert int8
            int8_blob = int8.astype(np.int8).tobytes()
            conn.execute("DELETE FROM person_embeddings_scalar WHERE person_id = ?", (person_id,))
            conn.execute("""
                INSERT INTO person_embeddings_scalar (person_id, embedding)
                VALUES (?, vec_int8(?))
            """, (person_id, int8_blob))

            count += 1

        conn.commit()
        return count

    def get_all_source_ids(self, source: Optional[str] = None) -> set[str]:
        """
        Get all source_ids from the people table.

        Useful for resume operations to skip already-imported records.

        Args:
            source: Optional source filter (e.g., "wikidata")

        Returns:
            Set of source_id strings (e.g., Q codes for Wikidata)
        """
        conn = self._connect()

        if self._is_v2:
            id_col = "source_identifier"
            if source:
                source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
                cursor = conn.execute(
                    f"SELECT DISTINCT {id_col} FROM people WHERE source_id = ?",
                    (source_type_id,)
                )
            else:
                cursor = conn.execute(f"SELECT DISTINCT {id_col} FROM people")
        else:
            if source:
                cursor = conn.execute(
                    "SELECT DISTINCT source_id FROM people WHERE source = ?",
                    (source,)
                )
            else:
                cursor = conn.execute("SELECT DISTINCT source_id FROM people")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[PersonRecord]:
        """Iterate over all person records, optionally filtered by source."""
        conn = self._connect()

        if self._is_v2:
            if source:
                source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
                cursor = conn.execute("""
                    SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                           v.known_for_role, v.known_for_org, v.known_for_org_id,
                           v.birth_date, v.death_date, p.record
                    FROM people_view v
                    JOIN people p ON v.id = p.id
                    WHERE p.source_id = ?
                """, (source_type_id,))
            else:
                cursor = conn.execute("""
                    SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                           v.known_for_role, v.known_for_org, v.known_for_org_id,
                           v.birth_date, v.death_date, p.record
                    FROM people_view v
                    JOIN people p ON v.id = p.id
                """)
            for row in cursor:
                yield PersonRecord(
                    name=row["name"],
                    source=row["source"],
                    source_id=row["source_identifier"],
                    country=row["country"] or "",
                    person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                    known_for_role=row["known_for_role"] or "",
                    known_for_org=row["known_for_org"] or "",
                    known_for_org_id=row["known_for_org_id"],  # Can be None
                    birth_date=row["birth_date"] or "",
                    death_date=row["death_date"] or "",
                    record=json.loads(row["record"]),
                )
        else:
            if source:
                cursor = conn.execute("""
                    SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                    FROM people
                    WHERE source = ?
                """, (source,))
            else:
                cursor = conn.execute("""
                    SELECT name, source, source_id, country, person_type, known_for_role, known_for_org, known_for_org_id, birth_date, death_date, record
                    FROM people
                """)

            for row in cursor:
                yield PersonRecord(
                    name=row["name"],
                    source=row["source"],
                    source_id=row["source_id"],
                    country=row["country"] or "",
                    person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
                    known_for_role=row["known_for_role"] or "",
                    known_for_org=row["known_for_org"] or "",
                    known_for_org_id=row["known_for_org_id"],  # Can be None
                    birth_date=row["birth_date"] or "",
                    death_date=row["death_date"] or "",
                    record=json.loads(row["record"]),
                )

    def resolve_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> tuple[int, int]:
        """
        Update records that have QIDs instead of labels.

        This is called after dump import to resolve any QIDs that were
        stored because labels weren't available in the cache at import time.

        If resolving would create a duplicate of an existing record with
        resolved labels, the QID version is deleted instead.

        Args:
            label_map: Mapping of QID -> label for resolution
            batch_size: Commit batch size

        Returns:
            Tuple of (updates, deletes)
        """
        conn = self._connect()

        # v2 schema stores QIDs as integers, not text - this method doesn't apply
        if self._is_v2:
            logger.debug("Skipping resolve_qid_labels for v2 schema (QIDs stored as integers)")
            return 0, 0

        # Find all records with QIDs in any field (role or org - these are in unique constraint)
        # Country is not part of unique constraint so can be updated directly
        cursor = conn.execute("""
            SELECT id, source, source_id, country, known_for_role, known_for_org
            FROM people
            WHERE (country LIKE 'Q%' AND country GLOB 'Q[0-9]*')
               OR (known_for_role LIKE 'Q%' AND known_for_role GLOB 'Q[0-9]*')
               OR (known_for_org LIKE 'Q%' AND known_for_org GLOB 'Q[0-9]*')
        """)
        rows = cursor.fetchall()

        updates = 0
        deletes = 0

        for row in rows:
            person_id = row["id"]
            source = row["source"]
            source_id = row["source_id"]
            country = row["country"]
            role = row["known_for_role"]
            org = row["known_for_org"]

            # Resolve QIDs to labels
            new_country = label_map.get(country, country) if country.startswith("Q") and country[1:].isdigit() else country
            new_role = label_map.get(role, role) if role.startswith("Q") and role[1:].isdigit() else role
            new_org = label_map.get(org, org) if org.startswith("Q") and org[1:].isdigit() else org

            # Skip if nothing changed
            if new_country == country and new_role == role and new_org == org:
                continue

            # Check if resolved values would duplicate an existing record
            # (unique constraint is on source, source_id, known_for_role, known_for_org)
            if new_role != role or new_org != org:
                cursor2 = conn.execute("""
                    SELECT id FROM people
                    WHERE source = ? AND source_id = ? AND known_for_role = ? AND known_for_org = ?
                    AND id != ?
                """, (source, source_id, new_role, new_org, person_id))
                existing = cursor2.fetchone()

                if existing:
                    # Duplicate would exist - delete the QID version
                    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
                    conn.execute("DELETE FROM person_embeddings WHERE person_id = ?", (person_id,))
                    deletes += 1
                    logger.debug(f"Deleted duplicate QID record {person_id} (source_id={source_id})")
                    continue

            # No duplicate - update in place
            conn.execute("""
                UPDATE people SET country = ?, known_for_role = ?, known_for_org = ?
                WHERE id = ?
            """, (new_country, new_role, new_org, person_id))
            updates += 1

            if (updates + deletes) % batch_size == 0:
                conn.commit()
                logger.info(f"Resolved QID labels: {updates} updates, {deletes} deletes...")

        conn.commit()
        logger.info(f"Resolved QID labels: {updates} updates, {deletes} deletes")
        return updates, deletes

    def get_unresolved_qids(self) -> set[str]:
        """
        Get all QIDs that still need resolution in the database.

        Returns:
            Set of QIDs (starting with 'Q') found in country, role, or org fields
        """
        conn = self._connect()

        # v2 schema stores QIDs as integers, not text - this method doesn't apply
        if self._is_v2:
            return set()

        qids: set[str] = set()

        # Get QIDs from country field
        cursor = conn.execute("""
            SELECT DISTINCT country FROM people
            WHERE country LIKE 'Q%' AND country GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["country"])

        # Get QIDs from known_for_role field
        cursor = conn.execute("""
            SELECT DISTINCT known_for_role FROM people
            WHERE known_for_role LIKE 'Q%' AND known_for_role GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["known_for_role"])

        # Get QIDs from known_for_org field
        cursor = conn.execute("""
            SELECT DISTINCT known_for_org FROM people
            WHERE known_for_org LIKE 'Q%' AND known_for_org GLOB 'Q[0-9]*'
        """)
        for row in cursor:
            qids.add(row["known_for_org"])

        return qids

    def insert_qid_labels(
        self,
        label_map: dict[str, str],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert QID -> label mappings into the lookup table.

        Args:
            label_map: Mapping of QID -> label
            batch_size: Commit batch size

        Returns:
            Number of labels inserted/updated
        """
        conn = self._connect()
        count = 0
        skipped = 0

        for qid, label in label_map.items():
            # Skip non-Q IDs (e.g., property IDs like P19)
            if not qid.startswith("Q"):
                skipped += 1
                continue

            # v2 schema stores QID as integer without Q prefix
            if self._is_v2:
                try:
                    qid_val: str | int = int(qid[1:])
                except ValueError:
                    skipped += 1
                    continue
            else:
                qid_val = qid

            conn.execute(
                "INSERT OR REPLACE INTO qid_labels (qid, label) VALUES (?, ?)",
                (qid_val, label)
            )
            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.debug(f"Inserted {count} QID labels...")

        conn.commit()
        logger.info(f"Inserted {count} QID labels into lookup table")
        return count

    def get_qid_label(self, qid: str) -> Optional[str]:
        """
        Get the label for a QID from the lookup table.

        Args:
            qid: Wikidata QID (e.g., 'Q30')

        Returns:
            Label string or None if not found
        """
        conn = self._connect()

        # v2 schema stores QID as integer without Q prefix
        if self._is_v2:
            qid_val: str | int = int(qid[1:]) if qid.startswith("Q") else int(qid)
        else:
            qid_val = qid

        cursor = conn.execute(
            "SELECT label FROM qid_labels WHERE qid = ?",
            (qid_val,)
        )
        row = cursor.fetchone()
        return row["label"] if row else None

    def get_all_qid_labels(self) -> dict[str, str]:
        """
        Get all QID -> label mappings from the lookup table.

        Returns:
            Dict mapping QID -> label
        """
        conn = self._connect()
        cursor = conn.execute("SELECT qid, label FROM qid_labels")
        return {row["qid"]: row["label"] for row in cursor}

    def get_qid_labels_count(self) -> int:
        """Get the number of QID labels in the lookup table."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM qid_labels")
        return cursor.fetchone()[0]

    def canonicalize(self, batch_size: int = 10000) -> dict[str, int]:
        """
        Canonicalize person records by linking equivalent entries across sources.

        Uses a multi-phase approach:
        1. Match by normalized name + same organization (org canonical group)
        2. Match by normalized name + overlapping date ranges

        Source priority (lower = more authoritative):
        - wikidata: 1 (curated, has Q codes)
        - sec_edgar: 2 (US insider filings)
        - companies_house: 3 (UK officers)

        Args:
            batch_size: Number of records to process before committing

        Returns:
            Stats dict with counts for each matching type
        """
        conn = self._connect()
        stats = {
            "total_records": 0,
            "matched_by_org": 0,
            "matched_by_date": 0,
            "canonical_groups": 0,
            "records_in_groups": 0,
        }

        logger.info("Phase 1: Building person index...")

        # Load all people with their normalized names and org info
        if self._is_v2:
            cursor = conn.execute("""
                SELECT p.id, p.name, p.name_normalized, s.name as source, p.source_identifier as source_id,
                       p.known_for_org, p.known_for_org_id, p.from_date, p.to_date
                FROM people p
                JOIN source_types s ON p.source_id = s.id
            """)
        else:
            cursor = conn.execute("""
                SELECT id, name, name_normalized, source, source_id,
                       known_for_org, known_for_org_id, from_date, to_date
                FROM people
            """)

        people: list[dict] = []
        for row in cursor:
            people.append({
                "id": row["id"],
                "name": row["name"],
                "name_normalized": row["name_normalized"],
                "source": row["source"],
                "source_id": row["source_id"],
                "known_for_org": row["known_for_org"],
                "known_for_org_id": row["known_for_org_id"],
                "from_date": row["from_date"],
                "to_date": row["to_date"],
            })

        stats["total_records"] = len(people)
        logger.info(f"Loaded {len(people)} person records")

        if len(people) == 0:
            return stats

        # Initialize Union-Find
        person_ids = [p["id"] for p in people]
        uf = UnionFind(person_ids)

        # Build indexes for efficient matching
        # Index by normalized name
        name_to_people: dict[str, list[dict]] = {}
        for p in people:
            name_norm = p["name_normalized"]
            name_to_people.setdefault(name_norm, []).append(p)

        logger.info("Phase 2: Matching by normalized name + organization...")

        # Match people with same normalized name and same organization
        for name_norm, same_name in name_to_people.items():
            if len(same_name) < 2:
                continue

            # Group by organization (using known_for_org_id if available, else known_for_org)
            org_groups: dict[str, list[dict]] = {}
            for p in same_name:
                org_key = str(p["known_for_org_id"]) if p["known_for_org_id"] else p["known_for_org"]
                if org_key:  # Only group if they have an org
                    org_groups.setdefault(org_key, []).append(p)

            # Union people with same name + same org
            for org_key, org_people in org_groups.items():
                if len(org_people) >= 2:
                    first_id = org_people[0]["id"]
                    for p in org_people[1:]:
                        uf.union(first_id, p["id"])
                        stats["matched_by_org"] += 1

        logger.info(f"Phase 2 complete: {stats['matched_by_org']} matches by org")

        logger.info("Phase 3: Matching by normalized name + overlapping dates...")

        # Match people with same normalized name and overlapping date ranges
        for name_norm, same_name in name_to_people.items():
            if len(same_name) < 2:
                continue

            # Skip if already all unified
            roots = set(uf.find(p["id"]) for p in same_name)
            if len(roots) == 1:
                continue

            # Check for overlapping date ranges
            for i, p1 in enumerate(same_name):
                for p2 in same_name[i+1:]:
                    # Skip if already in same group
                    if uf.find(p1["id"]) == uf.find(p2["id"]):
                        continue

                    # Check date overlap (if both have dates)
                    if p1["from_date"] and p2["from_date"]:
                        # Simple overlap check: if either from_date is before other's to_date
                        p1_from = p1["from_date"]
                        p1_to = p1["to_date"] or "9999-12-31"
                        p2_from = p2["from_date"]
                        p2_to = p2["to_date"] or "9999-12-31"

                        # Overlap if: p1_from <= p2_to AND p2_from <= p1_to
                        if p1_from <= p2_to and p2_from <= p1_to:
                            uf.union(p1["id"], p2["id"])
                            stats["matched_by_date"] += 1

        logger.info(f"Phase 3 complete: {stats['matched_by_date']} matches by date")

        logger.info("Phase 4: Applying canonical updates...")

        # Get all groups and select canonical record for each
        groups = uf.groups()

        # Build id -> source and id -> from_date mappings
        id_to_source = {p["id"]: p["source"] for p in people}
        id_to_from_date = {p["id"]: p["from_date"] or "" for p in people}

        batch_updates: list[tuple[int, int, int]] = []  # (person_id, canon_id, canon_size)

        for _root, group_ids in groups.items():
            group_size = len(group_ids)

            if group_size == 1:
                # Single record is its own canonical
                person_id = group_ids[0]
                batch_updates.append((person_id, person_id, 1))
            else:
                # Multiple records - pick highest priority source as canonical
                # Sort by: source priority asc, has-date first, from_date desc (most recent), id asc
                sorted_ids = sorted(
                    group_ids,
                    key=lambda pid: (
                        PERSON_SOURCE_PRIORITY.get(id_to_source[pid], 99),
                        0 if id_to_from_date[pid] else 1,  # entries with dates first
                        _invert_date_str(id_to_from_date[pid]),  # most recent date first
                        pid,
                    )
                )
                canon_id = sorted_ids[0]
                stats["canonical_groups"] += 1
                stats["records_in_groups"] += group_size

                for person_id in group_ids:
                    batch_updates.append((person_id, canon_id, group_size if person_id == canon_id else 1))

            # Commit in batches
            if len(batch_updates) >= batch_size:
                self._apply_person_canon_updates(batch_updates)
                conn.commit()
                logger.info(f"Applied {len(batch_updates)} canon updates...")
                batch_updates = []

        # Final batch
        if batch_updates:
            self._apply_person_canon_updates(batch_updates)
            conn.commit()

        logger.info(f"Canonicalization complete: {stats['canonical_groups']} groups, "
                   f"{stats['records_in_groups']} records in multi-record groups")

        return stats

    def _apply_person_canon_updates(self, updates: list[tuple[int, int, int]]) -> None:
        """Apply batch of canon updates: (person_id, canon_id, canon_size)."""
        conn = self._conn
        assert conn is not None

        for person_id, canon_id, canon_size in updates:
            conn.execute(
                "UPDATE people SET canon_id = ?, canon_size = ? WHERE id = ?",
                (canon_id, canon_size, person_id)
            )


# =============================================================================
# Module-level singletons for new v2 databases
# =============================================================================

_roles_database_instances: dict[str, "RolesDatabase"] = {}
_locations_database_instances: dict[str, "LocationsDatabase"] = {}


def get_roles_database(db_path: Optional[str | Path] = None, readonly: bool = True) -> "RolesDatabase":
    """
    Get a singleton RolesDatabase instance for the given path.

    Args:
        db_path: Path to database file
        readonly: If True (default), open in read-only mode.

    Returns:
        Shared RolesDatabase instance (readonly by default)
    """
    path_key = str(db_path or DEFAULT_DB_PATH) + (":ro" if readonly else ":rw")
    if path_key not in _roles_database_instances:
        logger.debug(f"Creating new RolesDatabase instance for {path_key}")
        _roles_database_instances[path_key] = RolesDatabase(db_path=db_path, readonly=readonly)
    return _roles_database_instances[path_key]


def get_locations_database(db_path: Optional[str | Path] = None, readonly: bool = True) -> "LocationsDatabase":
    """
    Get a singleton LocationsDatabase instance for the given path.

    Args:
        db_path: Path to database file
        readonly: If True (default), open in read-only mode.

    Returns:
        Shared LocationsDatabase instance (readonly by default)
    """
    path_key = str(db_path or DEFAULT_DB_PATH) + (":ro" if readonly else ":rw")
    if path_key not in _locations_database_instances:
        logger.debug(f"Creating new LocationsDatabase instance for {path_key}")
        _locations_database_instances[path_key] = LocationsDatabase(db_path=db_path, readonly=readonly)
    return _locations_database_instances[path_key]


# =============================================================================
# ROLES DATABASE (v2)
# =============================================================================


class RolesDatabase:
    """
    SQLite database for job titles/roles.

    Stores normalized role records with source tracking and supports
    canonicalization to group equivalent roles (e.g., CEO, Chief Executive).
    """

    def __init__(self, db_path: Optional[str | Path] = None, readonly: bool = True):
        """
        Initialize the roles database.

        Args:
            db_path: Path to database file (creates if not exists)
            readonly: If True (default), open in read-only mode (avoids locking).
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None
        self._role_cache: dict[str, int] = {}  # name_normalized -> role_id

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, readonly=self._readonly)
        if not self._readonly:
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        """Create roles table and indexes."""
        conn = self._conn
        assert conn is not None

        # Check if enum tables exist, create and seed if not
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='source_types'"
        )
        if not cursor.fetchone():
            logger.info("Creating enum tables for v2 schema...")
            from .schema_v2 import (
                CREATE_SOURCE_TYPES,
                CREATE_PEOPLE_TYPES,
                CREATE_ORGANIZATION_TYPES,
                CREATE_SIMPLIFIED_LOCATION_TYPES,
                CREATE_LOCATION_TYPES,
            )
            conn.execute(CREATE_SOURCE_TYPES)
            conn.execute(CREATE_PEOPLE_TYPES)
            conn.execute(CREATE_ORGANIZATION_TYPES)
            conn.execute(CREATE_SIMPLIFIED_LOCATION_TYPES)
            conn.execute(CREATE_LOCATION_TYPES)
            seed_all_enums(conn)

        # Create roles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qid INTEGER,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source_id INTEGER NOT NULL DEFAULT 4,
                source_identifier TEXT,
                record TEXT NOT NULL DEFAULT '{}',
                canon_id INTEGER DEFAULT NULL,
                canon_size INTEGER DEFAULT 1,
                UNIQUE(name_normalized, source_id)
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_name_normalized ON roles(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_qid ON roles(qid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_source_id ON roles(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_canon_id ON roles(canon_id)")

        conn.commit()

    def close(self) -> None:
        """Clear connection reference."""
        self._conn = None

    def get_or_create(
        self,
        name: str,
        source_id: int = 4,  # wikidata
        qid: Optional[int] = None,
        source_identifier: Optional[str] = None,
    ) -> int:
        """
        Get or create a role record.

        Args:
            name: Role/title name
            source_id: FK to source_types table
            qid: Optional Wikidata QID as integer
            source_identifier: Optional source-specific identifier

        Returns:
            Role ID
        """
        if not name:
            raise ValueError("Role name cannot be empty")

        conn = self._connect()
        name_normalized = name.lower().strip()

        # Check cache
        cache_key = f"{name_normalized}:{source_id}"
        if cache_key in self._role_cache:
            return self._role_cache[cache_key]

        # Check database
        cursor = conn.execute(
            "SELECT id FROM roles WHERE name_normalized = ? AND source_id = ?",
            (name_normalized, source_id)
        )
        row = cursor.fetchone()
        if row:
            role_id = row["id"]
            self._role_cache[cache_key] = role_id
            return role_id

        # Create new role
        cursor = conn.execute(
            """
            INSERT INTO roles (name, name_normalized, source_id, qid, source_identifier)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, name_normalized, source_id, qid, source_identifier)
        )
        role_id = cursor.lastrowid
        assert role_id is not None
        conn.commit()

        self._role_cache[cache_key] = role_id
        return role_id

    def get_by_id(self, role_id: int) -> Optional[RoleRecord]:
        """Get a role record by ID."""
        conn = self._connect()

        cursor = conn.execute(
            "SELECT id, qid, name, source_id, source_identifier, record FROM roles WHERE id = ?",
            (role_id,)
        )
        row = cursor.fetchone()
        if row:
            source_name = SOURCE_ID_TO_NAME.get(row["source_id"], "wikidata")
            return RoleRecord(
                name=row["name"],
                source=source_name,
                source_id=row["source_identifier"],
                qid=row["qid"],
                record=json.loads(row["record"]) if row["record"] else {},
            )
        return None

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, str, float]]:
        """
        Search for roles by name.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of (role_id, role_name, score) tuples
        """
        conn = self._connect()
        query_normalized = query.lower().strip()

        # Exact match first
        cursor = conn.execute(
            "SELECT id, name FROM roles WHERE name_normalized = ? LIMIT 1",
            (query_normalized,)
        )
        row = cursor.fetchone()
        if row:
            return [(row["id"], row["name"], 1.0)]

        # LIKE match
        cursor = conn.execute(
            """
            SELECT id, name FROM roles
            WHERE name_normalized LIKE ?
            ORDER BY length(name)
            LIMIT ?
            """,
            (f"%{query_normalized}%", top_k)
        )

        results = []
        for row in cursor:
            # Simple score based on match quality
            name_normalized = row["name"].lower()
            if query_normalized == name_normalized:
                score = 1.0
            elif name_normalized.startswith(query_normalized):
                score = 0.9
            else:
                score = 0.7
            results.append((row["id"], row["name"], score))

        return results

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the roles table."""
        conn = self._connect()

        cursor = conn.execute("SELECT COUNT(*) FROM roles")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM roles WHERE canon_id IS NOT NULL")
        canonicalized = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(DISTINCT canon_id) FROM roles WHERE canon_id IS NOT NULL")
        groups = cursor.fetchone()[0]

        return {
            "total_roles": total,
            "canonicalized": canonicalized,
            "canonical_groups": groups,
        }


# =============================================================================
# LOCATIONS DATABASE (v2)
# =============================================================================


class LocationsDatabase:
    """
    SQLite database for geopolitical locations.

    Stores countries, states, cities with hierarchical relationships
    and type classification. Supports pycountry integration.
    """

    def __init__(self, db_path: Optional[str | Path] = None, readonly: bool = True):
        """
        Initialize the locations database.

        Args:
            db_path: Path to database file (creates if not exists)
            readonly: If True (default), open in read-only mode (avoids locking).
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None
        self._location_cache: dict[str, int] = {}  # lookup_key -> location_id
        self._location_type_cache: dict[str, int] = {}  # type_name -> type_id
        self._location_type_qid_cache: dict[int, int] = {}  # qid -> type_id

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection using shared connection pool."""
        if self._conn is not None:
            return self._conn

        self._conn = _get_shared_connection(self._db_path, readonly=self._readonly)
        if not self._readonly:
            self._create_tables()
        self._build_caches()
        return self._conn

    def _create_tables(self) -> None:
        """Create locations table and indexes."""
        conn = self._conn
        assert conn is not None

        # Check if enum tables exist, create and seed if not
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='source_types'"
        )
        if not cursor.fetchone():
            logger.info("Creating enum tables for v2 schema...")
            from .schema_v2 import (
                CREATE_SOURCE_TYPES,
                CREATE_PEOPLE_TYPES,
                CREATE_ORGANIZATION_TYPES,
                CREATE_SIMPLIFIED_LOCATION_TYPES,
                CREATE_LOCATION_TYPES,
            )
            conn.execute(CREATE_SOURCE_TYPES)
            conn.execute(CREATE_PEOPLE_TYPES)
            conn.execute(CREATE_ORGANIZATION_TYPES)
            conn.execute(CREATE_SIMPLIFIED_LOCATION_TYPES)
            conn.execute(CREATE_LOCATION_TYPES)
            seed_all_enums(conn)

        # Create locations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qid INTEGER,
                name TEXT NOT NULL,
                name_normalized TEXT NOT NULL,
                source_id INTEGER NOT NULL DEFAULT 4,
                source_identifier TEXT,
                parent_ids TEXT,
                location_type_id INTEGER NOT NULL DEFAULT 2,
                record TEXT NOT NULL DEFAULT '{}',
                from_date TEXT DEFAULT NULL,
                to_date TEXT DEFAULT NULL,
                canon_id INTEGER DEFAULT NULL,
                canon_size INTEGER DEFAULT 1,
                UNIQUE(source_identifier, source_id)
            )
        """)

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_name ON locations(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_name_normalized ON locations(name_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_qid ON locations(qid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_source_id ON locations(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_location_type_id ON locations(location_type_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_locations_canon_id ON locations(canon_id)")

        conn.commit()

    def _build_caches(self) -> None:
        """Build lookup caches from database and seed data."""
        # Load location type caches from seed data
        self._location_type_cache = dict(LOCATION_TYPE_NAME_TO_ID)
        self._location_type_qid_cache = dict(LOCATION_TYPE_QID_TO_ID)

        # Load existing locations into cache
        conn = self._conn
        if conn:
            cursor = conn.execute(
                "SELECT id, name_normalized, source_identifier FROM locations"
            )
            for row in cursor:
                # Cache by normalized name
                self._location_cache[row["name_normalized"]] = row["id"]
                # Also cache by source_identifier
                if row["source_identifier"]:
                    self._location_cache[row["source_identifier"].lower()] = row["id"]

    def close(self) -> None:
        """Clear connection reference."""
        self._conn = None

    def get_or_create(
        self,
        name: str,
        location_type_id: int,
        source_id: int = 4,  # wikidata
        qid: Optional[int] = None,
        source_identifier: Optional[str] = None,
        parent_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Get or create a location record.

        Args:
            name: Location name
            location_type_id: FK to location_types table
            source_id: FK to source_types table
            qid: Optional Wikidata QID as integer
            source_identifier: Optional source-specific identifier (e.g., "US", "CA")
            parent_ids: Optional list of parent location IDs

        Returns:
            Location ID
        """
        if not name:
            raise ValueError("Location name cannot be empty")

        conn = self._connect()
        name_normalized = name.lower().strip()

        # Check cache by source_identifier first (more specific)
        if source_identifier:
            cache_key = source_identifier.lower()
            if cache_key in self._location_cache:
                return self._location_cache[cache_key]

        # Check cache by normalized name
        if name_normalized in self._location_cache:
            return self._location_cache[name_normalized]

        # Check database
        if source_identifier:
            cursor = conn.execute(
                "SELECT id FROM locations WHERE source_identifier = ? AND source_id = ?",
                (source_identifier, source_id)
            )
        else:
            cursor = conn.execute(
                "SELECT id FROM locations WHERE name_normalized = ? AND source_id = ?",
                (name_normalized, source_id)
            )

        row = cursor.fetchone()
        if row:
            location_id = row["id"]
            self._location_cache[name_normalized] = location_id
            if source_identifier:
                self._location_cache[source_identifier.lower()] = location_id
            return location_id

        # Create new location
        parent_ids_json = json.dumps(parent_ids) if parent_ids else None
        cursor = conn.execute(
            """
            INSERT INTO locations
            (name, name_normalized, source_id, source_identifier, qid, location_type_id, parent_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, name_normalized, source_id, source_identifier, qid, location_type_id, parent_ids_json)
        )
        location_id = cursor.lastrowid
        assert location_id is not None
        conn.commit()

        self._location_cache[name_normalized] = location_id
        if source_identifier:
            self._location_cache[source_identifier.lower()] = location_id
        return location_id

    def get_or_create_by_qid(
        self,
        name: str,
        wikidata_type_qid: int,
        source_id: int = 4,
        entity_qid: Optional[int] = None,
        source_identifier: Optional[str] = None,
        parent_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Get or create a location using Wikidata P31 type QID.

        Args:
            name: Location name
            wikidata_type_qid: Wikidata instance-of QID (e.g., 515 for city)
            source_id: FK to source_types table
            entity_qid: Wikidata QID of the entity itself
            source_identifier: Optional source-specific identifier
            parent_ids: Optional list of parent location IDs

        Returns:
            Location ID
        """
        location_type_id = self.get_location_type_id_from_qid(wikidata_type_qid)
        return self.get_or_create(
            name=name,
            location_type_id=location_type_id,
            source_id=source_id,
            qid=entity_qid,
            source_identifier=source_identifier,
            parent_ids=parent_ids,
        )

    def get_by_id(self, location_id: int) -> Optional[LocationRecord]:
        """Get a location record by ID."""
        conn = self._connect()

        cursor = conn.execute(
            """
            SELECT id, qid, name, source_id, source_identifier, location_type_id,
                   parent_ids, from_date, to_date, record
            FROM locations WHERE id = ?
            """,
            (location_id,)
        )
        row = cursor.fetchone()
        if row:
            source_name = SOURCE_ID_TO_NAME.get(row["source_id"], "wikidata")
            location_type_id = row["location_type_id"]
            location_type_name = self._get_location_type_name(location_type_id)
            simplified_id = LOCATION_TYPE_TO_SIMPLIFIED.get(location_type_id, 7)
            simplified_name = SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME.get(simplified_id, "other")

            parent_ids = json.loads(row["parent_ids"]) if row["parent_ids"] else []

            return LocationRecord(
                name=row["name"],
                source=source_name,
                source_id=row["source_identifier"],
                qid=row["qid"],
                location_type=location_type_name,
                simplified_type=SimplifiedLocationType(simplified_name),
                parent_ids=parent_ids,
                from_date=row["from_date"],
                to_date=row["to_date"],
                record=json.loads(row["record"]) if row["record"] else {},
            )
        return None

    def _get_location_type_name(self, type_id: int) -> str:
        """Get location type name from ID."""
        # Reverse lookup in cache
        for name, id_ in self._location_type_cache.items():
            if id_ == type_id:
                return name
        return "other"

    def get_location_type_id(self, type_name: str) -> int:
        """Get location_type_id for a type name."""
        return self._location_type_cache.get(type_name, 36)  # default to "other"

    def get_location_type_id_from_qid(self, wikidata_qid: int) -> int:
        """Get location_type_id from Wikidata P31 QID."""
        return self._location_type_qid_cache.get(wikidata_qid, 36)  # default to "other"

    def get_simplified_type(self, location_type_id: int) -> str:
        """Get simplified type name for a location_type_id."""
        simplified_id = LOCATION_TYPE_TO_SIMPLIFIED.get(location_type_id, 7)
        return SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME.get(simplified_id, "other")

    def resolve_region_text(self, text: str) -> Optional[int]:
        """
        Resolve a region/country text to a location ID.

        Uses pycountry for country resolution, then falls back to search.

        Args:
            text: Region text (country code, name, or QID)

        Returns:
            Location ID or None if not resolved
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # Check cache first
        if text_lower in self._location_cache:
            return self._location_cache[text_lower]

        # Try pycountry resolution
        alpha_2 = self._resolve_via_pycountry(text)
        if alpha_2:
            alpha_2_lower = alpha_2.lower()
            if alpha_2_lower in self._location_cache:
                location_id = self._location_cache[alpha_2_lower]
                self._location_cache[text_lower] = location_id  # Cache the input too
                return location_id

            # Country not in database yet, import it
            try:
                country = pycountry.countries.get(alpha_2=alpha_2)
                if country:
                    country_type_id = self._location_type_cache.get("country", 2)
                    location_id = self.get_or_create(
                        name=country.name,
                        location_type_id=country_type_id,
                        source_id=4,  # wikidata
                        source_identifier=alpha_2,
                    )
                    self._location_cache[text_lower] = location_id
                    return location_id
            except Exception:
                pass

        return None

    def _resolve_via_pycountry(self, region: str) -> Optional[str]:
        """Try to resolve region via pycountry."""
        region_clean = region.strip()
        if not region_clean:
            return None

        # Try as 2-letter code
        if len(region_clean) == 2:
            country = pycountry.countries.get(alpha_2=region_clean.upper())
            if country:
                return country.alpha_2

        # Try as 3-letter code
        if len(region_clean) == 3:
            country = pycountry.countries.get(alpha_3=region_clean.upper())
            if country:
                return country.alpha_2

        # Try fuzzy search
        try:
            matches = pycountry.countries.search_fuzzy(region_clean)
            if matches:
                return matches[0].alpha_2
        except LookupError:
            pass

        return None

    def import_from_pycountry(self) -> int:
        """
        Import all countries from pycountry.

        Returns:
            Number of locations imported
        """
        conn = self._connect()
        country_type_id = self._location_type_cache.get("country", 2)
        count = 0

        for country in pycountry.countries:
            name = country.name
            alpha_2 = country.alpha_2
            name_normalized = name.lower()

            # Check if already exists
            if alpha_2.lower() in self._location_cache:
                continue

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO locations
                (name, name_normalized, source_id, source_identifier, location_type_id)
                VALUES (?, ?, 4, ?, ?)
                """,
                (name, name_normalized, alpha_2, country_type_id)
            )

            if cursor.lastrowid:
                self._location_cache[name_normalized] = cursor.lastrowid
                self._location_cache[alpha_2.lower()] = cursor.lastrowid
                count += 1

        conn.commit()
        logger.info(f"Imported {count} countries from pycountry")
        return count

    def search(
        self,
        query: str,
        top_k: int = 10,
        simplified_type: Optional[str] = None,
    ) -> list[tuple[int, str, float]]:
        """
        Search for locations by name.

        Args:
            query: Search query
            top_k: Maximum results to return
            simplified_type: Optional filter by simplified type (e.g., "country", "city")

        Returns:
            List of (location_id, location_name, score) tuples
        """
        conn = self._connect()
        query_normalized = query.lower().strip()

        # Build query with optional type filter
        if simplified_type:
            # Get all location_type_ids for this simplified type
            simplified_id = {
                name: id_ for id_, name in SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME.items()
            }.get(simplified_type)
            if simplified_id:
                type_ids = [
                    type_id for type_id, simp_id in LOCATION_TYPE_TO_SIMPLIFIED.items()
                    if simp_id == simplified_id
                ]
                if type_ids:
                    placeholders = ",".join("?" * len(type_ids))
                    cursor = conn.execute(
                        f"""
                        SELECT id, name FROM locations
                        WHERE name_normalized LIKE ? AND location_type_id IN ({placeholders})
                        ORDER BY length(name)
                        LIMIT ?
                        """,
                        [f"%{query_normalized}%"] + type_ids + [top_k]
                    )
                else:
                    return []
            else:
                return []
        else:
            cursor = conn.execute(
                """
                SELECT id, name FROM locations
                WHERE name_normalized LIKE ?
                ORDER BY length(name)
                LIMIT ?
                """,
                (f"%{query_normalized}%", top_k)
            )

        results = []
        for row in cursor:
            name_normalized = row["name"].lower()
            if query_normalized == name_normalized:
                score = 1.0
            elif name_normalized.startswith(query_normalized):
                score = 0.9
            else:
                score = 0.7
            results.append((row["id"], row["name"], score))

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the locations table."""
        conn = self._connect()

        cursor = conn.execute("SELECT COUNT(*) FROM locations")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM locations WHERE canon_id IS NOT NULL")
        canonicalized = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(DISTINCT canon_id) FROM locations WHERE canon_id IS NOT NULL")
        groups = cursor.fetchone()[0]

        # Count by simplified type
        by_type: dict[str, int] = {}
        cursor = conn.execute("""
            SELECT lt.simplified_id, COUNT(*) as cnt
            FROM locations l
            JOIN location_types lt ON l.location_type_id = lt.id
            GROUP BY lt.simplified_id
        """)
        for row in cursor:
            type_name = SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME.get(row["simplified_id"], "other")
            by_type[type_name] = row["cnt"]

        return {
            "total_locations": total,
            "canonicalized": canonicalized,
            "canonical_groups": groups,
            "by_type": by_type,
        }

    def insert_batch(self, records: list[LocationRecord]) -> int:
        """
        Insert a batch of location records.

        Args:
            records: List of LocationRecord objects to insert

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        conn = self._connect()
        inserted = 0

        for record in records:
            name_normalized = record.name.lower().strip()
            source_identifier = record.source_id  # Q code in source_id field

            # Check cache first
            cache_key = source_identifier.lower() if source_identifier else name_normalized
            if cache_key in self._location_cache:
                continue

            # Get location_type_id from type name
            location_type_id = self._location_type_cache.get(record.location_type, 36)  # default "other"
            source_id = SOURCE_NAME_TO_ID.get(record.source, 4)  # default wikidata

            parent_ids_json = json.dumps(record.parent_ids) if record.parent_ids else None
            record_json = json.dumps(record.record) if record.record else "{}"

            try:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO locations
                    (name, name_normalized, source_id, source_identifier, qid, location_type_id, parent_ids, record, from_date, to_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.name,
                        name_normalized,
                        source_id,
                        source_identifier,
                        record.qid,
                        location_type_id,
                        parent_ids_json,
                        record_json,
                        record.from_date,
                        record.to_date,
                    )
                )
                if cursor.lastrowid:
                    self._location_cache[name_normalized] = cursor.lastrowid
                    if source_identifier:
                        self._location_cache[source_identifier.lower()] = cursor.lastrowid
                    inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert location {record.name}: {e}")

        conn.commit()
        return inserted

    def get_all_source_ids(self, source: str = "wikidata") -> set[str]:
        """
        Get all source_identifiers for a given source.

        Args:
            source: Source name (e.g., "wikidata")

        Returns:
            Set of source_identifiers
        """
        conn = self._connect()
        source_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute(
            "SELECT source_identifier FROM locations WHERE source_id = ? AND source_identifier IS NOT NULL",
            (source_id,)
        )
        return {row["source_identifier"] for row in cursor}
