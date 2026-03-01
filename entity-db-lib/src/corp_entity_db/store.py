"""
Entity/Organization database with USearch for vector search.

Uses USearch HNSW indexes for fast approximate nearest neighbor search.
Embeddings live only in USearch indexes, never in SQLite.
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
from .schema_v2 import create_all_tables
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
)

logger = logging.getLogger(__name__)

from .hub import DB_VERSION, DEFAULT_DB_FULL_FILENAME

# Person types whose identity is defined by their role at an organization
_ORG_DEFINED_TYPE_STRS = {
    "executive", "politician", "government", "military", "legal", "journalist",
    "professional", "academic", "scientist", "entrepreneur",
}

_TYPE_PREPOSITIONS: dict[str, str] = {
    "executive": "of", "politician": "of", "government": "at",
    "military": "of", "legal": "at", "journalist": "at",
    "professional": "at", "academic": "at", "scientist": "at",
    "entrepreneur": "of",
}

_IDENTITY_LABELS: dict[str, str] = {
    "artist": "artist", "media": "media personality",
    "athlete": "athlete", "activist": "activist",
}


def _a_or_an(word: str) -> str:
    """Return 'an' if word starts with a vowel, else 'a'."""
    return "an" if word and word[0].lower() in "aeiou" else "a"


def format_person_query(name: str, person_type: Optional[str] = None,
                        role: Optional[str] = None, org: Optional[str] = None) -> str:
    """Format a person query for embedding, matching stored natural language format.

    Org-defined types: "{name}, a {role} {prep} {org}"
    Identity-defined types: "{name}, a {type_label}"
    Unknown/missing type falls back to role+org with 'at' preposition.
    """
    if person_type and person_type in _ORG_DEFINED_TYPE_STRS:
        prep = _TYPE_PREPOSITIONS[person_type]
        if role and org:
            return f"{name}, {_a_or_an(role)} {role} {prep} {org}"
        elif role:
            return f"{name}, {_a_or_an(role)} {role}"
        elif org:
            return f"{name} {prep} {org}"
        return name

    if person_type and person_type in _IDENTITY_LABELS:
        label = _IDENTITY_LABELS[person_type]
        return f"{name}, {_a_or_an(label)} {label}"

    # Unknown type or no type — use role+org if available
    if role and org:
        return f"{name}, {_a_or_an(role)} {role} at {org}"
    elif role:
        return f"{name}, {_a_or_an(role)} {role}"
    elif org:
        return f"{name} at {org}"
    return name


def format_person_query_variants(name: str, person_type: Optional[str] = None,
                                 role: Optional[str] = None, org: Optional[str] = None) -> list[str]:
    """Generate up to 4 query embedding variants for multi-variant person search.

    Variants (deduplicated):
      [0] name_only:  "Tim Cook"
      [1] name_type:  "Tim Cook, a CEO" (org-defined) / "Taylor Swift, an artist" (identity)
      [2] name_org:   "Tim Cook of Apple" (with type-appropriate preposition)
      [3] name_full:  "Tim Cook, a CEO of Apple" (full context, same as format_person_query)

    When fields are missing, variants degrade to simpler forms. Duplicate texts
    are kept in the list (the caller can deduplicate embeddings).
    """
    # Variant 0: name only
    v_name = name

    # Variant 1: name + type/role
    v_name_type = name
    if person_type and person_type in _ORG_DEFINED_TYPE_STRS:
        if role:
            v_name_type = f"{name}, {_a_or_an(role)} {role}"
    elif person_type and person_type in _IDENTITY_LABELS:
        label = _IDENTITY_LABELS[person_type]
        v_name_type = f"{name}, {_a_or_an(label)} {label}"
    elif role:
        v_name_type = f"{name}, {_a_or_an(role)} {role}"

    # Variant 2: name + org
    v_name_org = name
    if org:
        if person_type and person_type in _ORG_DEFINED_TYPE_STRS:
            prep = _TYPE_PREPOSITIONS[person_type]
        else:
            prep = "at"
        v_name_org = f"{name} {prep} {org}"

    # Variant 3: full context (same as format_person_query)
    v_full = format_person_query(name, person_type=person_type, role=role, org=org)

    return [v_name, v_name_type, v_name_org, v_full]

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

            _apply_pragmas(conn, readonly=True)

            _shared_readonly_connections[path_key] = conn
            logger.debug(f"Created shared read-only database connection for {path_key}")

        return _shared_readonly_connections[path_key]

    if path_key not in _shared_connections:
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        _apply_pragmas(conn, readonly=False)

        # Ensure all v3 schema tables exist (CREATE IF NOT EXISTS is idempotent)
        create_all_tables(conn, embedding_dim)
        seed_all_enums(conn)
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
    embedder: "CompanyEmbedder",
    embedding_dim: int = 768,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 200,
    embed_batch_size: int = 64,
    progress_callback: Optional[Any] = None,
) -> int:
    """Build HNSW index for organizations by generating embeddings on-the-fly.

    Embeddings are generated from organization names in batches, quantized to
    int8, and fed directly into the USearch index. No embeddings are stored in
    the database — this is the only place organization embeddings live.

    For people, use build_people_composite_index() instead.

    Args:
        conn: Database connection
        entity_type: 'organizations' (people no longer supported here)
        embedder: CompanyEmbedder instance for generating embeddings
        embedding_dim: Dimension of embeddings (default 768)
        M: Number of connections per node (default 32)
        ef_construction: Size of dynamic candidate list during construction (default 200)
        ef_search: Size of dynamic candidate list during search (default 200)
        embed_batch_size: Batch size for embedding generation
        progress_callback: Optional callable(processed, total) for progress

    Returns:
        Number of vectors indexed
    """
    from usearch.index import Index

    # Get database path for saving index
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    cache_dir = db_path.parent

    if entity_type != "organizations":
        raise ValueError(f"Unknown entity_type: {entity_type}. For people, use build_people_composite_index().")

    index_path = cache_dir / "organizations_usearch.bin"

    # Count total records
    total_count = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()[0]
    if total_count == 0:
        logger.warning("No organizations found")
        return 0

    logger.info(f"Building USearch index for {total_count:,} organizations (on-the-fly embeddings)")
    logger.info(f"Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

    # Initialize USearch index with int8 quantization
    index = Index(
        ndim=embedding_dim,
        metric='cos',
        dtype='i8',
        connectivity=M,
        expansion_add=ef_construction,
        expansion_search=ef_search,
    )

    # Stream organizations in batches, generate embeddings, add to index
    DB_BATCH_SIZE = 10_000
    total_added = 0
    last_id = 0

    while True:
        cursor = conn.execute(
            "SELECT id, name FROM organizations WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, DB_BATCH_SIZE),
        )
        rows = cursor.fetchall()
        if not rows:
            break

        ids = [row["id"] for row in rows]
        names = [row["name"] for row in rows]

        # Generate embeddings in sub-batches
        for i in range(0, len(ids), embed_batch_size):
            sub_ids = ids[i:i + embed_batch_size]
            sub_names = names[i:i + embed_batch_size]

            fp32_batch = embedder.embed_batch(sub_names, batch_size=embed_batch_size)

            # Quantize to int8 and add to index
            int8_batch = np.clip(np.round(fp32_batch * 127), -127, 127).astype(np.int8)
            ids_array = np.array(sub_ids, dtype=np.int64)
            index.add(ids_array, int8_batch)
            total_added += len(sub_ids)

        last_id = ids[-1]

        if progress_callback:
            progress_callback(total_added, total_count)
        logger.info(f"Indexed {total_added:,}/{total_count:,} organization vectors...")

    # Save index
    logger.info(f"Saving index to {index_path.name}...")
    index.save(str(index_path))

    index_size_mb = index_path.stat().st_size / 1024**2
    logger.info(f"USearch index built successfully: {total_added:,} vectors ({index_size_mb:.1f} MB)")

    return total_added


def build_people_composite_index(
    conn: sqlite3.Connection,
    embedder: "CompanyEmbedder",
    embedding_dim: int = 768,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 200,
    embed_batch_size: int = 64,
    progress_callback: Optional[Any] = None,
) -> int:
    """Build USearch HNSW index for people using composite embeddings.

    Generates composite embeddings (name + role + org) on-the-fly and feeds
    them directly into the USearch index. No embeddings are stored in the
    people table — this is the only place people embeddings live.

    Args:
        conn: Database connection
        embedder: CompanyEmbedder instance for generating embeddings
        embedding_dim: Dimension of composite embeddings (default 768 = 3 × 256)
        M: HNSW connections per node
        ef_construction: Construction quality parameter
        ef_search: Search quality parameter
        embed_batch_size: Batch size for embedding generation
        progress_callback: Optional callable(processed, total) for progress

    Returns:
        Number of vectors indexed
    """
    from usearch.index import Index

    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    cache_dir = db_path.parent
    index_path = cache_dir / "people_usearch.bin"

    total_count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    if total_count == 0:
        logger.warning("No people records found")
        return 0

    logger.info(f"Building composite USearch index for {total_count:,} people")
    logger.info(f"Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

    index = Index(
        ndim=embedding_dim,
        metric='cos',
        dtype='i8',
        connectivity=M,
        expansion_add=ef_construction,
        expansion_search=ef_search,
    )

    # Stream people in batches, generate composite embeddings, add to index
    DB_BATCH_SIZE = 10_000
    total_added = 0
    last_id = 0

    while True:
        cursor = conn.execute("""
            SELECT p.id, p.name, r.name as role_name, kfo.name as org_name
            FROM people p
            LEFT JOIN roles r ON p.known_for_role_id = r.id
            LEFT JOIN organizations kfo ON p.known_for_org_id = kfo.id
            WHERE p.id > ?
            ORDER BY p.id
            LIMIT ?
        """, (last_id, DB_BATCH_SIZE))

        rows = cursor.fetchall()
        if not rows:
            break

        ids = [row["id"] for row in rows]
        names = [row["name"] for row in rows]
        roles = [row["role_name"] for row in rows]
        orgs = [row["org_name"] for row in rows]

        # Generate composite embeddings in sub-batches
        for i in range(0, len(ids), embed_batch_size):
            sub_ids = ids[i:i + embed_batch_size]
            sub_names = names[i:i + embed_batch_size]
            sub_roles = roles[i:i + embed_batch_size]
            sub_orgs = orgs[i:i + embed_batch_size]

            fp32_batch = embedder.embed_composite_person_batch(
                names=sub_names, roles=sub_roles, orgs=sub_orgs,
                batch_size=embed_batch_size,
            )

            # Quantize to int8 and add to index
            int8_batch = np.clip(np.round(fp32_batch * 127), -127, 127).astype(np.int8)
            ids_array = np.array(sub_ids, dtype=np.int64)
            index.add(ids_array, int8_batch)
            total_added += len(sub_ids)

        last_id = ids[-1]

        if progress_callback:
            progress_callback(total_added, total_count)
        logger.info(f"Indexed {total_added:,}/{total_count:,} people vectors...")

    logger.info(f"Saving people index to {index_path.name}...")
    index.save(str(index_path))

    index_size_mb = index_path.stat().st_size / 1024**2
    logger.info(f"People USearch index built: {total_added:,} vectors ({index_size_mb:.1f} MB)")

    return total_added


def build_people_identity_index(
    conn: sqlite3.Connection,
    embedder: "CompanyEmbedder",
    identity_dim: int = 256,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 200,
    embed_batch_size: int = 192,
    progress_callback: Optional[Any] = None,
) -> int:
    """Build USearch HNSW index for people using identity embeddings (Matryoshka 256-dim).

    For each person, builds embedding text using the same logic as
    PersonRecord.get_embedding_text() — identity types get "{name}, a {type_label}",
    org-defined types get their full natural language form.

    The embedding is truncated to `identity_dim` dimensions and L2-normalized
    (Matryoshka truncation), then quantized to int8 for the USearch index.

    Args:
        conn: Database connection
        embedder: CompanyEmbedder instance for generating embeddings
        identity_dim: Dimension of identity embeddings (default 256)
        M: HNSW connections per node
        ef_construction: Construction quality parameter
        ef_search: Search quality parameter
        embed_batch_size: Batch size for embedding generation
        progress_callback: Optional callable(processed, total) for progress

    Returns:
        Number of vectors indexed
    """
    from usearch.index import Index

    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    cache_dir = db_path.parent
    index_path = cache_dir / "people_identity_usearch.bin"

    total_count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    if total_count == 0:
        logger.warning("No people records found")
        return 0

    logger.info(f"Building identity USearch index for {total_count:,} people ({identity_dim}-dim)")
    logger.info(f"Parameters: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

    index = Index(
        ndim=identity_dim,
        metric='cos',
        dtype='i8',
        connectivity=M,
        expansion_add=ef_construction,
        expansion_search=ef_search,
    )

    DB_BATCH_SIZE = 10_000
    total_added = 0
    last_id = 0

    while True:
        cursor = conn.execute("""
            SELECT p.id, p.name, pt.name as person_type,
                   r.name as role_name, kfo.name as org_name
            FROM people p
            JOIN people_types pt ON p.person_type_id = pt.id
            LEFT JOIN roles r ON p.known_for_role_id = r.id
            LEFT JOIN organizations kfo ON p.known_for_org_id = kfo.id
            WHERE p.id > ?
            ORDER BY p.id
            LIMIT ?
        """, (last_id, DB_BATCH_SIZE))

        rows = cursor.fetchall()
        if not rows:
            break

        # Build embedding texts using format_person_query (mirrors PersonRecord.get_embedding_text)
        ids = [row["id"] for row in rows]
        texts: list[str] = []
        for row in rows:
            texts.append(format_person_query(
                row["name"],
                person_type=row["person_type"],
                role=row["role_name"],
                org=row["org_name"],
            ))

        # Embed in sub-batches, truncate to identity_dim, L2-normalize
        for i in range(0, len(ids), embed_batch_size):
            sub_ids = ids[i:i + embed_batch_size]
            sub_texts = texts[i:i + embed_batch_size]

            fp32_batch = embedder.embed_for_identity_index_batch(
                sub_texts, dim=identity_dim, batch_size=embed_batch_size,
            )

            # Quantize to int8 and add to index
            int8_batch = np.clip(np.round(fp32_batch * 127), -127, 127).astype(np.int8)
            ids_array = np.array(sub_ids, dtype=np.int64)
            index.add(ids_array, int8_batch)
            total_added += len(sub_ids)

        last_id = ids[-1]

        if progress_callback:
            progress_callback(total_added, total_count)
        logger.info(f"Indexed {total_added:,}/{total_count:,} identity vectors...")

    logger.info(f"Saving identity index to {index_path.name}...")
    index.save(str(index_path))

    index_size_mb = index_path.stat().st_size / 1024**2
    logger.info(f"People identity index built: {total_added:,} vectors ({index_size_mb:.1f} MB)")

    return total_added


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
    SQLite database with USearch for organization vector search.

    Uses USearch HNSW indexes for fast approximate nearest neighbor search.
    Embeddings live only in USearch indexes, never in SQLite.
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
        self._schema_version: int = 1  # Updated on connect (1=legacy, 2=normalized FKs, 3=normalized FKs+lite, 4=no embedding columns)
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

            # Check for schema version in db_info metadata table
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

        return self._conn

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(
        self,
        record: CompanyRecord,
    ) -> int:
        """
        Insert an organization record.

        Embeddings are generated separately during index building and stored
        only in the USearch index, not in the organizations table.

        Args:
            record: Organization record to insert

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_name(record.name)

        # v2+ schema: use FK IDs instead of TEXT columns
        source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
        entity_type_id = ORG_TYPE_NAME_TO_ID.get(record.entity_type.value, 17)  # 17 = unknown

        # Resolve region to location_id if provided
        region_id = None
        if record.region:
            locations_db = get_locations_database(db_path=self._db_path, readonly=False)
            region_id = locations_db.resolve_region_text(record.region)

        # Parse QID integer from source_id for wikidata entries (e.g. "Q312" -> 312)
        qid = None
        if record.source in ("wikidata", "wikipedia") and record.source_id.startswith("Q"):
            qid_str = record.source_id[1:]
            if qid_str.isdigit():
                qid = int(qid_str)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO organizations
            (name, name_normalized, source_id, source_identifier, qid, region_id, entity_type_id, from_date, to_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            source_type_id,
            record.source_id,
            qid,
            region_id,
            entity_type_id,
            record.from_date or "",
            record.to_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[CompanyRecord],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert multiple organization records.

        Embeddings are generated separately during index building and stored
        only in the USearch index, not in the organizations table.

        Args:
            records: List of organization records
            batch_size: Commit batch size

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for i, record in enumerate(records):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_name(record.name)

            # v2+ schema: use FK IDs instead of TEXT columns
            source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
            entity_type_id = ORG_TYPE_NAME_TO_ID.get(record.entity_type.value, 17)  # 17 = unknown

            # region_id is resolved in pass 2 (resolve_fks) for wikidata imports
            region_id = None

            # Parse QID integer from source_id for wikidata entries (e.g. "Q312" -> 312)
            qid = None
            if record.source in ("wikidata", "wikipedia") and record.source_id.startswith("Q"):
                qid_str = record.source_id[1:]
                if qid_str.isdigit():
                    qid = int(qid_str)

            cursor = conn.execute("""
                INSERT OR REPLACE INTO organizations
                (name, name_normalized, source_id, source_identifier, qid, region_id, entity_type_id, from_date, to_date, record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                source_type_id,
                record.source_id,
                qid,
                region_id,
                entity_type_id,
                record.from_date or "",
                record.to_date or "",
                record_json,
            ))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} records...")

        conn.commit()
        return count

    def resolve_fks(self, qid_fk_data: dict[int, dict]) -> int:
        """
        Pass 2: Resolve cross-table FKs for organizations inserted during pass 1.

        Updates region_id by resolving country QIDs to location IDs.

        Args:
            qid_fk_data: Mapping of QID int → {"country_qid": "Q30"}

        Returns:
            Number of records updated
        """
        if not qid_fk_data:
            return 0

        conn = self._connect()

        # Preload QID → location_id lookup
        cursor = conn.execute("SELECT qid, id FROM locations WHERE qid IS NOT NULL")
        qid_to_location_id: dict[int, int] = {row[0]: row[1] for row in cursor}

        org_null_region = conn.execute("SELECT COUNT(*) FROM organizations WHERE region_id IS NULL").fetchone()[0]
        logger.info(
            f"Org resolve_fks: {len(qid_fk_data):,} FK entries, "
            f"{len(qid_to_location_id):,} location QIDs available, "
            f"{org_null_region:,} orgs needing region_id"
        )
        if org_null_region == 0:
            logger.info("All org region_ids already resolved, nothing to update")
            return 0

        updated = 0
        batch_count = 0
        for org_qid, fk_data in qid_fk_data.items():
            country_qid_str = fk_data.get("country_qid", "")
            if not country_qid_str:
                continue

            # Parse country QID to int
            if country_qid_str.startswith("Q") and country_qid_str[1:].isdigit():
                country_qid_int = int(country_qid_str[1:])
            else:
                continue

            region_id = qid_to_location_id.get(country_qid_int)
            if region_id is None:
                continue

            result = conn.execute(
                "UPDATE organizations SET region_id = ? WHERE qid = ? AND region_id IS NULL",
                (region_id, org_qid),
            )
            updated += result.rowcount
            batch_count += 1

            if batch_count % 10000 == 0:
                conn.commit()
                logger.info(f"Resolved {updated:,} org region FKs...")

        conn.commit()
        logger.info(f"Resolved {updated:,} org region FKs total ({batch_count:,} candidates matched locations)")
        return updated

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        source_filter: Optional[str] = None,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search for similar organizations using USearch ANN search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source (gleif, sec_edgar, etc.)

        Returns:
            List of (CompanyRecord, similarity_score) tuples
        """
        start = time.time()
        self._connect()

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Quantize and search USearch index
        query_int8 = np.clip(np.round(query_normalized * 127), -127, 127).astype(np.int8)

        if not self._load_hnsw_index():
            raise RuntimeError("USearch index not found. Run: corp-entity-db build-index")

        results = self._hnsw_search(query_int8, top_k, source_filter=source_filter)

        elapsed = time.time() - start
        logger.debug(f"Organization search took {elapsed:.3f}s (results={len(results)})")
        return results

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
        source_filter: Optional[str] = None,
    ) -> list[tuple[CompanyRecord, float]]:
        """
        Search using USearch index.

        Args:
            query_int8: Query embedding quantized to int8
            top_k: Number of results to return
            source_filter: Optional source filter

        Returns:
            List of (CompanyRecord, similarity) tuples
        """
        if self._hnsw_index is None:
            return []

        # Query USearch index (fetch more if we need to filter by source)
        fetch_k = top_k * 10 if source_filter else top_k
        fetch_k = min(fetch_k, len(self._hnsw_index))

        matches = self._hnsw_index.search(query_int8, fetch_k)

        # Convert to results
        results: list[tuple[CompanyRecord, float]] = []
        for org_id, dist in zip(matches.keys, matches.distances):
            org_id = int(org_id)

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

        cursor = conn.execute("""
            SELECT v.id, v.name, v.source, v.source_identifier, v.region, v.entity_type, v.canon_id, o.record
            FROM organizations_view v
            JOIN organizations o ON v.id = o.id
            WHERE v.id = ?
        """, (org_id,))

        row = cursor.fetchone()
        if row:
            record_data = json.loads(row["record"])
            # Add db_id and canon_id to record dict for canon-aware search
            record_data["db_id"] = row["id"]
            record_data["canon_id"] = row["canon_id"]
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_identifier"],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=record_data,
            )
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[CompanyRecord]:
        """Get an organization record by source and source_id."""
        conn = self._connect()

        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute("""
            SELECT v.name, v.source, v.source_identifier, v.region, v.entity_type, o.record
            FROM organizations_view v
            JOIN organizations o ON v.id = o.id
            WHERE o.source_id = ? AND o.source_identifier = ?
        """, (source_type_id, source_id))

        row = cursor.fetchone()
        if row:
            return CompanyRecord(
                name=row["name"],
                source=row["source"],
                source_id=row["source_identifier"],
                region=row["region"] or "",
                entity_type=EntityType(row["entity_type"]) if row["entity_type"] else EntityType.UNKNOWN,
                record=json.loads(row["record"]),
            )
        return None

    def get_id_by_source_id(self, source: str, source_id: str) -> Optional[int]:
        """Get the internal database ID for an organization by source and source_id."""
        conn = self._connect()

        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute("""
            SELECT id FROM organizations
            WHERE source_id = ? AND source_identifier = ?
        """, (source_type_id, source_id))

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

        # Count by source - join with source_types
        cursor = conn.execute("""
            SELECT st.name as source, COUNT(*) as cnt
            FROM organizations o
            JOIN source_types st ON o.source_id = st.id
            GROUP BY o.source_id
        """)
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

        if source:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute(
                "SELECT DISTINCT source_identifier FROM organizations WHERE source_id = ?",
                (source_type_id,)
            )
        else:
            cursor = conn.execute("SELECT DISTINCT source_identifier FROM organizations")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[CompanyRecord]:
        """Iterate over all records, optionally filtered by source."""
        conn = self._connect()

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

        cursor = conn.execute("""
            SELECT o.id, s.name as source, o.source_identifier as source_id, o.name, l.name as region, o.record
            FROM organizations o
            JOIN source_types s ON o.source_id = s.id
            LEFT JOIN locations l ON o.region_id = l.id
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

    def delete_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        conn = self._connect()

        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute("DELETE FROM organizations WHERE source_id = ?", (source_type_id,))

        deleted = cursor.rowcount

        conn.commit()

        logger.info(f"Deleted {deleted} records from source '{source}'")
        return deleted




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
    SQLite database with USearch for person vector search.

    Uses USearch HNSW indexes for fast approximate nearest neighbor search.
    Embeddings live only in USearch indexes, never in SQLite.

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
        self._schema_version: int = 1  # Updated on connect (1=legacy, 2=normalized FKs, 3=normalized FKs+lite, 4=no embedding columns)
        # USearch indexes for fast approximate nearest neighbor search
        self._hnsw_index: Optional[Any] = None  # primary composite 768-dim
        self._identity_index: Optional[Any] = None  # secondary identity 256-dim (fallback)

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

            # Check for schema version in db_info metadata table
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

            # Migrate unique constraint to include from_date (allows same role at same org in different periods)
            if self._is_v2 and not self._readonly:
                self._migrate_people_unique_constraint()

        return self._conn

    def _migrate_people_unique_constraint(self) -> None:
        """Migrate people table: add from_date to unique constraint, drop known_for_org_location_id.

        The old constraint UNIQUE(source_identifier, source_id, known_for_role_id, known_for_org_id)
        doesn't allow the same person to hold the same role at the same org in different time periods
        (e.g. Grover Cleveland as US President twice). Adding from_date fixes this.
        Also drops the known_for_org_location_id column (no longer used).
        """
        conn = self._conn
        assert conn is not None

        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='people'"
        ).fetchone()
        if not row:
            return

        create_sql = row[0]
        needs_unique_fix = "from_date)" not in create_sql
        needs_column_drop = "known_for_org_location_id" in create_sql

        if not needs_unique_fix and not needs_column_drop:
            return  # Already migrated

        logger.info("Migrating people table: updating unique constraint and dropping known_for_org_location_id...")

        conn.execute("BEGIN IMMEDIATE")
        try:
            # Drop views that reference the people table before renaming
            conn.execute("DROP VIEW IF EXISTS people_view")

            conn.execute("ALTER TABLE people RENAME TO people_old")
            conn.execute("""
                CREATE TABLE people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qid INTEGER,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    source_identifier TEXT NOT NULL,
                    country_id INTEGER,
                    person_type_id INTEGER NOT NULL DEFAULT 15,
                    known_for_role_id INTEGER,
                    known_for_org_id INTEGER,
                    from_date TEXT DEFAULT NULL,
                    to_date TEXT DEFAULT NULL,
                    birth_date TEXT DEFAULT NULL,
                    death_date TEXT DEFAULT NULL,
                    record TEXT NOT NULL DEFAULT '{}',
                    canon_id INTEGER DEFAULT NULL,
                    canon_size INTEGER DEFAULT 1,
                    FOREIGN KEY (source_id) REFERENCES source_types(id),
                    FOREIGN KEY (country_id) REFERENCES locations(id),
                    FOREIGN KEY (person_type_id) REFERENCES people_types(id),
                    FOREIGN KEY (known_for_role_id) REFERENCES roles(id),
                    FOREIGN KEY (known_for_org_id) REFERENCES organizations(id),
                    UNIQUE(source_identifier, source_id, known_for_role_id, known_for_org_id, from_date)
                )
            """)
            conn.execute("""
                INSERT INTO people
                (id, qid, name, name_normalized, source_id, source_identifier, country_id,
                 person_type_id, known_for_role_id, known_for_org_id, from_date, to_date,
                 birth_date, death_date, record, canon_id, canon_size)
                SELECT id, qid, name, name_normalized, source_id, source_identifier, country_id,
                       person_type_id, known_for_role_id, known_for_org_id, from_date, to_date,
                       birth_date, death_date, record, canon_id, canon_size
                FROM people_old
            """)
            conn.execute("DROP TABLE people_old")

            # Recreate indexes and view
            from .schema_v2 import CREATE_PEOPLE_V2_INDEXES, CREATE_PEOPLE_VIEW
            for stmt in CREATE_PEOPLE_V2_INDEXES.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.execute(CREATE_PEOPLE_VIEW)

            conn.execute("COMMIT")
            logger.info("Migration complete: people unique constraint now includes from_date")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        """Clear connection reference (shared connection remains open)."""
        self._conn = None

    def insert(
        self,
        record: PersonRecord,
    ) -> int:
        """
        Insert a person record.

        Embeddings are generated separately during post-import and stored
        only in the USearch index, not in the people table.

        Args:
            record: Person record to insert

        Returns:
            Row ID of inserted record
        """
        conn = self._connect()

        # Serialize record
        record_json = json.dumps(record.record)
        name_normalized = _normalize_person_name(record.name)

        # v2+ schema: use FK IDs instead of TEXT columns
        source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
        person_type_id = PEOPLE_TYPE_NAME_TO_ID.get(record.person_type.value, 15)  # 15 = unknown

        # Resolve country to location_id if provided
        country_id = None
        if record.country:
            locations_db = get_locations_database(db_path=self._db_path, readonly=False)
            country_id = locations_db.resolve_region_text(record.country)

        # Resolve known_for_role to role_id if provided
        role_id = None
        if record.known_for_role:
            roles_db = get_roles_database(db_path=self._db_path, readonly=False)
            role_id = roles_db.get_or_create(record.known_for_role, source_id=source_type_id)

        # Parse QID integer from source_id for wikidata entries (e.g. "Q265398" -> 265398)
        qid = None
        if record.source == "wikidata" and record.source_id.startswith("Q"):
            qid_str = record.source_id[1:]
            if qid_str.isdigit():
                qid = int(qid_str)

        cursor = conn.execute("""
            INSERT OR REPLACE INTO people
            (name, name_normalized, source_id, source_identifier, qid, country_id, person_type_id,
             known_for_role_id, known_for_org_id, from_date, to_date,
             birth_date, death_date, record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name,
            name_normalized,
            source_type_id,
            record.source_id,
            qid,
            country_id,
            person_type_id,
            role_id,
            record.known_for_org_id,
            record.from_date or "",
            record.to_date or "",
            record.birth_date or "",
            record.death_date or "",
            record_json,
        ))

        row_id = cursor.lastrowid
        assert row_id is not None

        conn.commit()
        return row_id

    def insert_batch(
        self,
        records: list[PersonRecord],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert multiple person records.

        Embeddings are generated separately during post-import and stored
        only in the USearch index, not in the people table.

        Args:
            records: List of person records
            batch_size: Commit batch size

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for i, record in enumerate(records):
            record_json = json.dumps(record.record)
            name_normalized = _normalize_person_name(record.name)

            # v2+ schema: use FK IDs instead of TEXT columns
            source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
            person_type_id = PEOPLE_TYPE_NAME_TO_ID.get(record.person_type.value, 15)  # 15 = unknown

            # country_id is resolved in pass 2 (resolve_fks) for wikidata imports
            country_id = None

            # Resolve known_for_role to role_id if provided
            role_id = None
            if record.known_for_role:
                roles_db = get_roles_database(db_path=self._db_path, readonly=False)
                role_id = roles_db.get_or_create(record.known_for_role, source_id=source_type_id)

            # Parse QID integer from source_id for wikidata entries (e.g. "Q265398" -> 265398)
            qid = None
            if record.source == "wikidata" and record.source_id.startswith("Q"):
                qid_str = record.source_id[1:]
                if qid_str.isdigit():
                    qid = int(qid_str)

            cursor = conn.execute("""
                INSERT OR REPLACE INTO people
                (name, name_normalized, source_id, source_identifier, qid, country_id, person_type_id,
                 known_for_role_id, known_for_org_id, from_date, to_date,
                 birth_date, death_date, record)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.name,
                name_normalized,
                source_type_id,
                record.source_id,
                qid,
                country_id,
                person_type_id,
                role_id,
                record.known_for_org_id,
                record.from_date or "",
                record.to_date or "",
                record.birth_date or "",
                record.death_date or "",
                record_json,
            ))

            count += 1

            if count % batch_size == 0:
                conn.commit()
                logger.info(f"Inserted {count} person records...")

        conn.commit()
        return count

    def resolve_fks(self, qid_fk_data: Optional[dict[int, dict]] = None) -> int:
        """
        Pass 2: Resolve cross-table FKs for people inserted during pass 1.

        Iterates over each person row individually, reads the org_qid from
        its record JSON, and resolves it to the correct org/location FK.
        This ensures each role-record gets its own org rather than all rows
        for the same person getting the same org.

        Country QIDs come from qid_fk_data if provided, otherwise from the
        record JSON column.

        Args:
            qid_fk_data: Optional mapping of QID int → {"country_qid": "Q30"}.
                          If None, country_qid is read from the record column.

        Returns:
            Number of records updated
        """
        conn = self._connect()

        # Preload QID → location_id lookup
        cursor = conn.execute("SELECT qid, id FROM locations WHERE qid IS NOT NULL")
        qid_to_location_id: dict[int, int] = {row[0]: row[1] for row in cursor}

        # Preload QID → org_id lookup
        cursor = conn.execute("SELECT qid, id, region_id FROM organizations WHERE qid IS NOT NULL")
        qid_to_org: dict[int, tuple[int, Optional[int]]] = {
            row[0]: (row[1], row[2]) for row in cursor
        }

        # Preload QID → role_id lookup
        cursor = conn.execute("SELECT qid, id FROM roles WHERE qid IS NOT NULL")
        qid_to_role_id: dict[int, int] = {row[0]: row[1] for row in cursor}
        logger.info(f"Loaded {len(qid_to_role_id):,} role QID→id mappings")

        # Iterate over individual rows, each with its own org_qid in record JSON
        cursor = conn.execute(
            "SELECT id, qid, record FROM people WHERE source_id = 4 AND qid IS NOT NULL"
        )
        rows = cursor.fetchall()
        logger.info(f"Resolving FKs for {len(rows):,} wikidata person rows...")

        updated = 0
        skipped = 0
        for row in rows:
            row_id, person_qid, record_json = row[0], row[1], row[2]
            rec = json.loads(record_json) if record_json else {}

            # Country QID: prefer qid_fk_data if provided, else from record
            if qid_fk_data and person_qid in qid_fk_data:
                country_qid_str = qid_fk_data[person_qid].get("country_qid", "")
            else:
                country_qid_str = rec.get("country_qid", "")

            # Org QID: always from this row's record (each row has its own org)
            org_qid_str = rec.get("org_qid", "")

            # Role QID: from this row's record
            role_qid_str = rec.get("role_qid", "")

            # Resolve country
            country_id = None
            if country_qid_str and country_qid_str.startswith("Q") and country_qid_str[1:].isdigit():
                country_id = qid_to_location_id.get(int(country_qid_str[1:]))

            # Resolve org → org_id
            org_id = None
            if org_qid_str and org_qid_str.startswith("Q") and org_qid_str[1:].isdigit():
                org_info = qid_to_org.get(int(org_qid_str[1:]))
                if org_info:
                    org_id = org_info[0]

            # Resolve role → role_id from P39 position QID.
            # Only fall back to P106 occupations when there is NO role_qid
            # (single-record path with no P39 positions). If a specific position
            # QID doesn't resolve, leave role_id NULL — using the generic
            # occupation would collapse distinct positions into the same role.
            role_id = None
            if role_qid_str and role_qid_str.startswith("Q") and role_qid_str[1:].isdigit():
                role_id = qid_to_role_id.get(int(role_qid_str[1:]))
            elif not role_qid_str:
                for occ_qid in rec.get("occupations", []):
                    if occ_qid.startswith("Q") and occ_qid[1:].isdigit():
                        role_id = qid_to_role_id.get(int(occ_qid[1:]))
                        if role_id is not None:
                            break

            if country_id is None and org_id is None and role_id is None:
                continue

            # Update this specific row by id
            try:
                result = conn.execute(
                    """UPDATE people SET
                        country_id = COALESCE(?, country_id),
                        known_for_org_id = COALESCE(known_for_org_id, ?),
                        known_for_role_id = COALESCE(known_for_role_id, ?)
                    WHERE id = ?""",
                    (country_id, org_id, role_id, row_id),
                )
            except sqlite3.IntegrityError:
                skipped += 1
                logger.error(f"UNIQUE constraint conflict resolving FKs for person Q{person_qid} "
                             f"row {row_id} (org_id={org_id}), skipping")
                continue

            updated += result.rowcount

            if updated % 10000 == 0 and updated > 0:
                conn.commit()
                logger.info(f"Resolved {updated:,} person FKs...")

        conn.commit()
        if skipped:
            logger.warning(f"Resolved {updated:,} person FKs total ({skipped:,} skipped due to UNIQUE conflicts)")
        else:
            logger.info(f"Resolved {updated:,} person FKs total")
        return updated

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

        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute("""
            UPDATE people SET from_date = ?, to_date = ?
            WHERE source_id = ? AND source_identifier = ?
        """, (from_date or "", to_date or "", source_type_id, source_id))

        conn.commit()
        return cursor.rowcount > 0

    def update_role_org(
        self,
        source: str,
        source_id: str,
        known_for_role: str,
        known_for_org_id: Optional[int],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> bool:
        """
        Update the role/org/dates data for a person record.

        Embeddings are regenerated separately during post-import and stored
        only in the USearch index.

        Args:
            source: Data source (e.g., 'wikidata')
            source_id: Source identifier (e.g., QID)
            known_for_role: Role/position title
            known_for_org_id: Organization FK or None
            from_date: Start date in ISO format or None
            to_date: End date in ISO format or None

        Returns:
            True if record was updated, False if not found
        """
        conn = self._connect()

        # First get the person's internal ID
        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        row = conn.execute(
            "SELECT id FROM people WHERE source_id = ? AND source_identifier = ?",
            (source_type_id, source_id)
        ).fetchone()

        if not row:
            return False

        person_id = row[0]

        # Update the person record (including dates)
        conn.execute("""
            UPDATE people SET
                known_for_org_id = ?,
                from_date = COALESCE(?, from_date, ''),
                to_date = COALESCE(?, to_date, '')
            WHERE id = ?
        """, (known_for_org_id, from_date, to_date, person_id))

        conn.commit()
        return True

    def backfill_known_for_org(
        self, qid_to_orgs: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]]
    ) -> int:
        """
        Backfill known_for_org_id for people using QID→org mappings.

        Only updates records where known_for_org_id is NULL.
        Resolves org QIDs to organization FKs (same source: wikidata→wikidata).

        Args:
            qid_to_orgs: Mapping of person Wikidata QID →
                [(org_qid, role_description, start_date, end_date), ...]

        Returns:
            Number of records updated
        """
        if not qid_to_orgs:
            return 0

        conn = self._connect()
        org_database = OrganizationDatabase(db_path=self._db_path, readonly=True)
        locations_db = get_locations_database(db_path=self._db_path, readonly=True)
        updated = 0
        batch_count = 0
        for person_qid, entries in qid_to_orgs.items():
            if not entries:
                continue

            # Strip Q prefix if present for integer QID column
            qid_int = int(person_qid.lstrip("Q")) if person_qid.startswith("Q") else None
            if qid_int is None:
                continue

            # Apply each resolved entry to a separate row for this person
            for entry_org_qid, entry_role, entry_start, entry_end in entries:
                if not entry_org_qid:
                    continue

                # Resolve org QID to org_id
                org_id = org_database.get_id_by_source_id("wikipedia", entry_org_qid)
                if org_id is None:
                    continue

                # Resolve role to role_id
                role_id = None
                if entry_role:
                    roles_db = get_roles_database(db_path=self._db_path, readonly=False)
                    role_id = roles_db.get_or_create(entry_role, source_id=4)  # 4 = wikidata

                # Find one row for this person that still needs an org
                target_id = conn.execute(
                    "SELECT id FROM people WHERE qid = ? AND known_for_org_id IS NULL LIMIT 1",
                    (qid_int,),
                ).fetchone()
                if target_id is None:
                    break  # No more rows without org for this person

                try:
                    result = conn.execute(
                        """UPDATE people SET known_for_org_id = ?,
                           known_for_role_id = COALESCE(known_for_role_id, ?),
                           from_date = CASE WHEN from_date IS NULL OR from_date = '' THEN ? ELSE from_date END,
                           to_date = CASE WHEN to_date IS NULL OR to_date = '' THEN ? ELSE to_date END
                        WHERE id = ?""",
                        (org_id, role_id, entry_start or "", entry_end or "", target_id[0]),
                    )
                    updated += result.rowcount
                except sqlite3.IntegrityError:
                    logger.warning(
                        f"UNIQUE conflict backfilling org for person Q{qid_int} "
                        f"(org_id={org_id}, role_id={role_id}), skipping"
                    )

            batch_count += 1
            if batch_count % 10000 == 0:
                conn.commit()
                logger.info(f"Backfilled {updated:,} known_for_org records...")

        conn.commit()
        logger.info(f"Backfilled {updated:,} known_for_org records total")
        return updated

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        identity_query_embedding: Optional[np.ndarray] = None,
        fallback_threshold: float = 0.75,
    ) -> list[tuple[PersonRecord, float]]:
        """
        Search for similar people using composite embedding + optional identity fallback.

        Two-stage approach:
        1. Primary USearch HNSW index (768-dim composite) for fast ANN search
        2. If identity_query_embedding provided and primary scores are low, also
           search the secondary identity index (256-dim) and merge results

        The identity fallback is used when:
        - identity_query_embedding is provided, AND
        - the best primary score < fallback_threshold

        Args:
            query_embedding: Composite query embedding (768-dim = name|role|org segments)
            top_k: Number of results to return
            identity_query_embedding: Optional 256-dim identity embedding for fallback search
            fallback_threshold: Primary score threshold below which identity fallback is used

        Returns:
            List of (PersonRecord, similarity_score) tuples
        """
        start = time.time()
        self._connect()

        if not self._load_hnsw_index():
            raise RuntimeError("USearch index not found. Run: corp-entity-db build-index")

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm
        query_int8 = np.clip(np.round(query_normalized * 127), -127, 127).astype(np.int8)

        results = self._hnsw_search(query_int8, top_k)

        # Stage 2: Identity fallback — if identity embedding provided and primary scores are weak
        if (
            identity_query_embedding is not None
            and self._identity_index is not None
            and (not results or results[0][1] < fallback_threshold)
        ):
            best_score = results[0][1] if results else 0.0
            logger.info(
                f"Identity fallback: primary best={best_score:.3f}, threshold={fallback_threshold}"
            )
            identity_results = self._identity_search(
                identity_query_embedding, top_k,
            )

            # Merge: deduplicate by person name+source, keep higher score
            seen: dict[int, tuple[PersonRecord, float]] = {}
            for record, score in results:
                key = hash((record.name, record.source, record.source_id))
                seen[key] = (record, score)
            for record, score in identity_results:
                key = hash((record.name, record.source, record.source_id))
                if key not in seen or score > seen[key][1]:
                    seen[key] = (record, score)

            results = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]

        elapsed = time.time() - start
        logger.debug(f"Person search took {elapsed:.3f}s (results={len(results)})")
        return results

    def _identity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[PersonRecord, float]]:
        """Search using the secondary identity USearch index (256-dim)."""
        if self._identity_index is None:
            return []

        # Normalize and quantize
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm
        query_int8 = np.clip(np.round(query_normalized * 127), -127, 127).astype(np.int8)

        fetch_k = min(top_k, len(self._identity_index))

        matches = self._identity_index.search(query_int8, fetch_k)

        results: list[tuple[PersonRecord, float]] = []
        for person_id, dist in zip(matches.keys, matches.distances):
            person_id = int(person_id)
            similarity = 1.0 - float(dist)
            record = self._get_record_by_id(person_id)
            if record is not None:
                results.append((record, similarity))
            if len(results) >= top_k:
                break

        return results

    # --- USearch approximate nearest neighbor search ---

    def _get_hnsw_index_path(self) -> Path:
        """Get path to USearch index file."""
        return self._db_path.parent / "people_usearch.bin"

    def _get_identity_index_path(self) -> Path:
        """Get path to identity USearch index file."""
        return self._db_path.parent / "people_identity_usearch.bin"

    def _load_hnsw_index(self) -> bool:
        """
        Load pre-built USearch indexes from disk.

        Loads both the primary composite index (768-dim) and the secondary
        identity index (256-dim). The identity index is optional — if missing,
        fallback search is simply not available.

        Returns:
            True if primary index successfully loaded, False otherwise.
        """
        if self._hnsw_index is not None:
            return True  # Already loaded

        index_path = self._get_hnsw_index_path()
        if not index_path.exists():
            logger.debug(f"USearch index not found: {index_path}")
            return False

        try:
            from usearch.index import Index

            # Load primary composite index
            logger.info(f"Loading USearch index from {index_path.name}...")
            index = Index.restore(str(index_path))
            # USearch doesn't persist expansion_search — restore it for good recall on large indexes
            index.expansion_search = 200
            self._hnsw_index = index
            logger.info(f"Loaded USearch index: {len(index):,} vectors, connectivity={index.connectivity}, ef_search={index.expansion_search}")

            # Load secondary identity index (non-fatal if missing)
            identity_path = self._get_identity_index_path()
            if identity_path.exists():
                identity_index = Index.restore(str(identity_path))
                identity_index.expansion_search = 200
                self._identity_index = identity_index
                logger.info(f"Loaded identity index: {len(identity_index):,} vectors ({identity_index.ndim}-dim)")
            else:
                logger.debug(f"Identity index not found: {identity_path} (fallback disabled)")

            return True

        except Exception as e:
            logger.warning(f"Failed to load USearch index: {e}")
            return False

    def _hnsw_search(
        self,
        query_int8: np.ndarray,
        top_k: int,
    ) -> list[tuple[PersonRecord, float]]:
        """
        Search using USearch index.

        Args:
            query_int8: Query embedding quantized to int8
            top_k: Number of results to return

        Returns:
            List of (PersonRecord, similarity) tuples
        """
        if self._hnsw_index is None:
            return []

        fetch_k = min(top_k, len(self._hnsw_index))

        matches = self._hnsw_index.search(query_int8, fetch_k)

        # Convert to results
        results: list[tuple[PersonRecord, float]] = []
        for person_id, dist in zip(matches.keys, matches.distances):
            person_id = int(person_id)

            # Convert distance to similarity (cosine distance is 1 - cosine similarity)
            similarity = 1.0 - float(dist)

            record = self._get_record_by_id(person_id)
            if record is not None:
                results.append((record, similarity))

            if len(results) >= top_k:
                break

        return results

    def _row_to_person_record(self, row: sqlite3.Row) -> PersonRecord:
        """Build a PersonRecord from a DB row (view)."""
        source_id_field = "source_identifier"
        # known_for_org comes from COALESCE in people_view (computed from org/location FKs)
        known_for_org_name = row["known_for_org"] or "" if "known_for_org" in row.keys() else ""
        return PersonRecord(
            name=row["name"],
            source=row["source"],
            source_id=row[source_id_field],
            country=row["country"] or "",
            person_type=PersonType(row["person_type"]) if row["person_type"] else PersonType.UNKNOWN,
            known_for_role=row["known_for_role"] or "",
            known_for_org_id=row["known_for_org_id"],
            known_for_org_name=known_for_org_name,
            birth_date=row["birth_date"] or "",
            death_date=row["death_date"] or "",
            record=json.loads(row["record"]),
        )

    def _get_record_by_id(self, person_id: int) -> Optional[PersonRecord]:
        """Get a person record by ID."""
        conn = self._conn
        assert conn is not None

        cursor = conn.execute("""
            SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                   v.known_for_role, v.known_for_org, v.known_for_org_id,
                   v.birth_date, v.death_date, p.record
            FROM people_view v
            JOIN people p ON v.id = p.id
            WHERE v.id = ?
        """, (person_id,))

        row = cursor.fetchone()
        if row:
            return self._row_to_person_record(row)
        return None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[PersonRecord]:
        """Get a person record by source and source_id."""
        conn = self._connect()

        source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute("""
            SELECT v.name, v.source, v.source_identifier, v.country, v.person_type,
                   v.known_for_role, v.known_for_org, v.known_for_org_id,
                   v.birth_date, v.death_date, p.record
            FROM people_view v
            JOIN people p ON v.id = p.id
            WHERE p.source_id = ? AND p.source_identifier = ?
        """, (source_type_id, source_id))

        row = cursor.fetchone()
        if row:
            return self._row_to_person_record(row)
        return None

    def get_stats(self) -> dict:
        """Get database statistics for people table."""
        conn = self._connect()

        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM people")
        total = cursor.fetchone()[0]

        # Count by person_type - join with people_types
        cursor = conn.execute("""
            SELECT pt.name as person_type, COUNT(*) as cnt
            FROM people p
            JOIN people_types pt ON p.person_type_id = pt.id
            GROUP BY p.person_type_id
        """)
        by_type = {row["person_type"]: row["cnt"] for row in cursor}

        # Count by source - join with source_types
        cursor = conn.execute("""
            SELECT st.name as source, COUNT(*) as cnt
            FROM people p
            JOIN source_types st ON p.source_id = st.id
            GROUP BY p.source_id
        """)
        by_source = {row["source"]: row["cnt"] for row in cursor}

        return {
            "total_records": total,
            "by_type": by_type,
            "by_source": by_source,
        }

    def iter_all_for_embedding(self, batch_size: int = 10000) -> Iterator[list[tuple[int, str, str | None, str | None]]]:
        """
        Yield batches of (person_id, name, role_name, org_name) for all people.

        Used by the index builder to generate composite embeddings on-the-fly.
        People embeddings are stored only in the USearch index, not in SQLite.

        Args:
            batch_size: Number of rows per batch

        Yields:
            Lists of (person_id, name, role_name, org_name) tuples
        """
        conn = self._connect()

        last_id = 0
        while True:
            cursor = conn.execute("""
                SELECT p.id, p.name, r.name as role_name,
                       kfo.name as org_name
                FROM people p
                LEFT JOIN roles r ON p.known_for_role_id = r.id
                LEFT JOIN organizations kfo ON p.known_for_org_id = kfo.id
                WHERE p.id > ?
                ORDER BY p.id
                LIMIT ?
            """, (last_id, batch_size))

            rows = cursor.fetchall()
            if not rows:
                break

            results = [
                (row["id"], row["name"], row["role_name"], row["org_name"])
                for row in rows
            ]
            yield results
            last_id = results[-1][0]

    def get_people_count(self) -> int:
        """Get total count of people records."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM people")
        return cursor.fetchone()[0]

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

        if source:
            source_type_id = SOURCE_NAME_TO_ID.get(source, 4)
            cursor = conn.execute(
                "SELECT DISTINCT source_identifier FROM people WHERE source_id = ?",
                (source_type_id,)
            )
        else:
            cursor = conn.execute("SELECT DISTINCT source_identifier FROM people")

        return {row[0] for row in cursor}

    def iter_records(self, source: Optional[str] = None) -> Iterator[PersonRecord]:
        """Iterate over all person records, optionally filtered by source."""
        conn = self._connect()

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
            yield self._row_to_person_record(row)

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
        cursor = conn.execute("""
            SELECT p.id, p.name, p.name_normalized, s.name as source, p.source_identifier as source_id,
                   p.known_for_org_id, p.from_date, p.to_date
            FROM people p
            JOIN source_types s ON p.source_id = s.id
        """)

        people: list[dict] = []
        for row in cursor:
            people.append({
                "id": row["id"],
                "name": row["name"],
                "name_normalized": row["name_normalized"],
                "source": row["source"],
                "source_id": row["source_id"],
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

            # Group by organization (using known_for_org_id)
            org_groups: dict[str, list[dict]] = {}
            for p in same_name:
                org_key = (
                    f"org:{p['known_for_org_id']}" if p["known_for_org_id"]
                    else ""
                )
                if org_key:  # Only group if they have an org/location
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

    def insert_batch(self, records: list[RoleRecord], batch_size: int = 5000) -> int:
        """
        Batch insert role records from Wikidata dump import.

        Uses INSERT OR IGNORE to skip duplicates (unique on name_normalized + source_id).

        Args:
            records: List of RoleRecord objects to insert
            batch_size: Commit interval

        Returns:
            Number of records inserted
        """
        conn = self._connect()
        count = 0

        for record in records:
            name_normalized = record.name.lower().strip()
            source_type_id = SOURCE_NAME_TO_ID.get(record.source, 4)
            record_json = json.dumps(record.record) if record.record else "{}"

            cursor = conn.execute(
                """INSERT OR IGNORE INTO roles
                (name, name_normalized, source_id, source_identifier, qid, record)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (record.name, name_normalized, source_type_id, record.source_id,
                 record.qid, record_json),
            )
            if cursor.rowcount > 0:
                count += 1

            if count % batch_size == 0 and count > 0:
                conn.commit()
                logger.info(f"Inserted {count} role records...")

        conn.commit()
        return count

    def get_all_source_ids(self, source: str = "wikidata") -> set[str]:
        """Get all source_identifiers for a given source."""
        conn = self._connect()
        source_id = SOURCE_NAME_TO_ID.get(source, 4)
        cursor = conn.execute(
            "SELECT source_identifier FROM roles WHERE source_id = ? AND source_identifier IS NOT NULL",
            (source_id,),
        )
        return {row[0] for row in cursor}

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

        Uses pycountry to normalize country names/codes, then looks up existing
        locations by normalized name. Does NOT create new entries — locations
        must already exist (imported from wikidata dump or other sources).

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

        # Try pycountry to normalize the input to a canonical country name
        alpha_2 = self._resolve_via_pycountry(text)
        if alpha_2:
            alpha_2_lower = alpha_2.lower()
            if alpha_2_lower in self._location_cache:
                location_id = self._location_cache[alpha_2_lower]
                self._location_cache[text_lower] = location_id
                return location_id

            # Try the canonical pycountry name in case it matches a wikidata location
            try:
                country = pycountry.countries.get(alpha_2=alpha_2)
                if country:
                    name_lower = country.name.lower()
                    if name_lower in self._location_cache:
                        location_id = self._location_cache[name_lower]
                        self._location_cache[text_lower] = location_id
                        self._location_cache[alpha_2_lower] = location_id
                        return location_id
            except Exception:
                pass

        # Try database lookup by normalized name directly
        conn = self._connect()
        cursor = conn.execute(
            "SELECT id FROM locations WHERE name_normalized = ?", (text_lower,)
        )
        row = cursor.fetchone()
        if row:
            location_id = row["id"]
            self._location_cache[text_lower] = location_id
            return location_id

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

    def resolve_parent_ids(self, qid_fk_data: dict[int, dict]) -> int:
        """
        Pass 2: Resolve parent_ids for locations inserted during pass 1.

        Resolves parent QIDs and country QIDs to location IDs and updates
        the parent_ids JSON column.

        Args:
            qid_fk_data: Mapping of QID int → {"parent_qids": ["Q30"], "country_qids": ["Q30"]}

        Returns:
            Number of records updated
        """
        if not qid_fk_data:
            return 0

        conn = self._connect()

        # Preload QID → location_id lookup
        cursor = conn.execute("SELECT qid, id FROM locations WHERE qid IS NOT NULL")
        qid_to_location_id: dict[int, int] = {row[0]: row[1] for row in cursor}

        updated = 0
        batch_count = 0
        for loc_qid, fk_data in qid_fk_data.items():
            parent_qids = fk_data.get("parent_qids", [])
            country_qids = fk_data.get("country_qids", [])

            # Combine parent + country QIDs, dedup
            all_parent_qids = list(dict.fromkeys(parent_qids + country_qids))
            if not all_parent_qids:
                continue

            # Resolve to location IDs
            parent_ids: list[int] = []
            for pqid in all_parent_qids:
                if pqid.startswith("Q") and pqid[1:].isdigit():
                    lid = qid_to_location_id.get(int(pqid[1:]))
                    if lid is not None:
                        parent_ids.append(lid)

            if not parent_ids:
                continue

            parent_ids_json = json.dumps(parent_ids)
            result = conn.execute(
                "UPDATE locations SET parent_ids = ? WHERE qid = ? AND (parent_ids IS NULL OR parent_ids = '[]')",
                (parent_ids_json, loc_qid),
            )
            updated += result.rowcount
            batch_count += 1

            if batch_count % 10000 == 0:
                conn.commit()
                logger.info(f"Resolved {updated:,} location parent_ids...")

        conn.commit()
        logger.info(f"Resolved {updated:,} location parent_ids total")
        return updated

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
