"""
Seed data for enum lookup tables in the v2 normalized schema.

This module contains all enum values that are seeded into lookup tables
when creating a fresh database or migrating from v1.
"""

from typing import Any

# =============================================================================
# SOURCE TYPES
# =============================================================================

SOURCE_TYPES: list[tuple[int, str]] = [
    (1, "gleif"),
    (2, "sec_edgar"),
    (3, "companies_house"),
    (4, "wikidata"),
]

# Mapping from old v1 source names to v2 source IDs
SOURCE_NAME_TO_ID: dict[str, int] = {
    "gleif": 1,
    "sec_edgar": 2,
    "companies_house": 3,
    "wikidata": 4,
    # Legacy name mapping (v1 used "wikipedia" for Wikidata sources)
    "wikipedia": 4,
}

SOURCE_ID_TO_NAME: dict[int, str] = {
    1: "gleif",
    2: "sec_edgar",
    3: "companies_house",
    4: "wikidata",
}

# =============================================================================
# PEOPLE TYPES
# =============================================================================

PEOPLE_TYPES: list[tuple[int, str]] = [
    (1, "executive"),
    (2, "politician"),
    (3, "government"),
    (4, "military"),
    (5, "legal"),
    (6, "professional"),
    (7, "academic"),
    (8, "artist"),
    (9, "media"),
    (10, "athlete"),
    (11, "entrepreneur"),
    (12, "journalist"),
    (13, "activist"),
    (14, "scientist"),
    (15, "unknown"),
]

PEOPLE_TYPE_NAME_TO_ID: dict[str, int] = {name: id_ for id_, name in PEOPLE_TYPES}
PEOPLE_TYPE_ID_TO_NAME: dict[int, str] = {id_: name for id_, name in PEOPLE_TYPES}

# =============================================================================
# ORGANIZATION TYPES
# =============================================================================

ORGANIZATION_TYPES: list[tuple[int, str]] = [
    (1, "business"),
    (2, "fund"),
    (3, "branch"),
    (4, "nonprofit"),
    (5, "ngo"),
    (6, "foundation"),
    (7, "trade_union"),
    (8, "government"),
    (9, "international_org"),
    (10, "political_party"),
    (11, "educational"),
    (12, "research"),
    (13, "religious"),
    (14, "sports"),
    (15, "media"),
    (16, "healthcare"),
    (17, "unknown"),
]

ORG_TYPE_NAME_TO_ID: dict[str, int] = {name: id_ for id_, name in ORGANIZATION_TYPES}
ORG_TYPE_ID_TO_NAME: dict[int, str] = {id_: name for id_, name in ORGANIZATION_TYPES}

# =============================================================================
# SIMPLIFIED LOCATION TYPES
# =============================================================================

SIMPLIFIED_LOCATION_TYPES: list[tuple[int, str]] = [
    (1, "continent"),
    (2, "country"),
    (3, "subdivision"),  # States, provinces, regions, counties, departments
    (4, "city"),         # Cities, towns, municipalities, communes
    (5, "district"),     # Districts, boroughs, neighborhoods
    (6, "historic"),     # Former countries, historic territories
    (7, "other"),        # Unclassified locations
]

SIMPLIFIED_LOCATION_TYPE_NAME_TO_ID: dict[str, int] = {
    name: id_ for id_, name in SIMPLIFIED_LOCATION_TYPES
}
SIMPLIFIED_LOCATION_TYPE_ID_TO_NAME: dict[int, str] = {
    id_: name for id_, name in SIMPLIFIED_LOCATION_TYPES
}

# =============================================================================
# DETAILED LOCATION TYPES WITH WIKIDATA QID MAPPINGS
# =============================================================================

# Format: (id, name, qid, simplified_id)
# qid is the Wikidata Q code as integer (e.g., Q515 -> 515)
LOCATION_TYPES: list[tuple[int, str, int | None, int]] = [
    # Continents (simplified_id=1)
    (1, "continent", 5107, 1),

    # Countries/Sovereigns (simplified_id=2)
    (2, "country", 6256, 2),
    (3, "sovereign_state", 3624078, 2),
    (4, "dependent_territory", 161243, 2),

    # Subdivisions - US specific (simplified_id=3)
    (5, "us_state", 35657, 3),
    (6, "us_county", 47168, 3),

    # Subdivisions - Other countries (simplified_id=3)
    (7, "state_of_australia", 5852411, 3),
    (8, "state_of_germany", 1221156, 3),
    (9, "state_of_india", 131541, 3),
    (10, "province", 34876, 3),
    (11, "region", 82794, 3),
    (12, "county", 28575, 3),
    (13, "department_france", 6465, 3),
    (14, "prefecture_japan", 50337, 3),
    (15, "canton_switzerland", 23058, 3),
    (16, "autonomous_community_spain", 10742, 3),
    (17, "voivodeship_poland", 150093, 3),
    (18, "oblast_russia", 835714, 3),

    # Cities/Towns (simplified_id=4)
    (19, "city", 515, 4),
    (20, "big_city", 1549591, 4),
    (21, "capital", 5119, 4),
    (22, "town", 3957, 4),
    (23, "municipality", 15284, 4),
    (24, "commune_france", 484170, 4),
    (25, "municipality_germany", 262166, 4),
    (26, "municipality_japan", 1054813, 4),
    (27, "village", 532, 4),
    (28, "hamlet", 5084, 4),

    # Districts (simplified_id=5)
    (29, "district", 149621, 5),
    (30, "borough", 5765681, 5),
    (31, "neighborhood", 123705, 5),
    (32, "ward", 12813115, 5),

    # Historic (simplified_id=6)
    (33, "former_country", 3024240, 6),
    (34, "ancient_civilization", 28171280, 6),
    (35, "historic_territory", 1620908, 6),

    # Other/Unknown (simplified_id=7)
    (36, "other", None, 7),
]

LOCATION_TYPE_NAME_TO_ID: dict[str, int] = {
    name: id_ for id_, name, _, _ in LOCATION_TYPES
}
LOCATION_TYPE_ID_TO_NAME: dict[int, str] = {
    id_: name for id_, name, _, _ in LOCATION_TYPES
}

# Mapping from Wikidata QID (P31 value) to location_type_id
LOCATION_TYPE_QID_TO_ID: dict[int, int] = {
    qid: id_ for id_, name, qid, _ in LOCATION_TYPES if qid is not None
}

# Mapping from location_type_id to simplified_id
LOCATION_TYPE_TO_SIMPLIFIED: dict[int, int] = {
    id_: simplified_id for id_, _, _, simplified_id in LOCATION_TYPES
}


# =============================================================================
# PYCOUNTRY INTEGRATION
# =============================================================================

def get_pycountry_countries() -> list[dict[str, Any]]:
    """
    Get all countries from pycountry for seeding the locations table.

    Returns:
        List of dicts with keys: name, alpha_2, alpha_3, numeric
    """
    import pycountry

    countries = []
    for country in pycountry.countries:
        countries.append({
            "name": country.name,
            "alpha_2": country.alpha_2,
            "alpha_3": getattr(country, "alpha_3", None),
            "numeric": getattr(country, "numeric", None),
        })
    return countries


# =============================================================================
# SEED FUNCTIONS
# =============================================================================

def seed_source_types(conn) -> int:
    """
    Seed source_types table.

    Args:
        conn: SQLite connection

    Returns:
        Number of rows inserted
    """
    conn.executemany(
        "INSERT OR IGNORE INTO source_types (id, name) VALUES (?, ?)",
        SOURCE_TYPES
    )
    conn.commit()
    return len(SOURCE_TYPES)


def seed_people_types(conn) -> int:
    """
    Seed people_types table.

    Args:
        conn: SQLite connection

    Returns:
        Number of rows inserted
    """
    conn.executemany(
        "INSERT OR IGNORE INTO people_types (id, name) VALUES (?, ?)",
        PEOPLE_TYPES
    )
    conn.commit()
    return len(PEOPLE_TYPES)


def seed_organization_types(conn) -> int:
    """
    Seed organization_types table.

    Args:
        conn: SQLite connection

    Returns:
        Number of rows inserted
    """
    conn.executemany(
        "INSERT OR IGNORE INTO organization_types (id, name) VALUES (?, ?)",
        ORGANIZATION_TYPES
    )
    conn.commit()
    return len(ORGANIZATION_TYPES)


def seed_simplified_location_types(conn) -> int:
    """
    Seed simplified_location_types table.

    Args:
        conn: SQLite connection

    Returns:
        Number of rows inserted
    """
    conn.executemany(
        "INSERT OR IGNORE INTO simplified_location_types (id, name) VALUES (?, ?)",
        SIMPLIFIED_LOCATION_TYPES
    )
    conn.commit()
    return len(SIMPLIFIED_LOCATION_TYPES)


def seed_location_types(conn) -> int:
    """
    Seed location_types table with Wikidata QID mappings.

    Args:
        conn: SQLite connection

    Returns:
        Number of rows inserted
    """
    conn.executemany(
        "INSERT OR IGNORE INTO location_types (id, name, qid, simplified_id) VALUES (?, ?, ?, ?)",
        LOCATION_TYPES
    )
    conn.commit()
    return len(LOCATION_TYPES)


def seed_all_enums(conn) -> dict[str, int]:
    """
    Seed all enum tables.

    Args:
        conn: SQLite connection

    Returns:
        Dict mapping table name to number of rows inserted
    """
    return {
        "source_types": seed_source_types(conn),
        "people_types": seed_people_types(conn),
        "organization_types": seed_organization_types(conn),
        "simplified_location_types": seed_simplified_location_types(conn),
        "location_types": seed_location_types(conn),
    }


def seed_pycountry_locations(conn, source_id: int = 4) -> int:
    """
    Seed locations table with pycountry countries.

    Args:
        conn: SQLite connection
        source_id: Source ID to use (default: 4 = wikidata, used for pycountry data)

    Returns:
        Number of locations inserted
    """
    import pycountry

    # Get country location_type_id
    country_type_id = LOCATION_TYPE_NAME_TO_ID["country"]
    count = 0

    for country in pycountry.countries:
        name = country.name
        alpha_2 = country.alpha_2
        name_normalized = name.lower()

        conn.execute(
            """
            INSERT OR IGNORE INTO locations
            (name, name_normalized, source_id, source_identifier, location_type_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, name_normalized, source_id, alpha_2, country_type_id)
        )
        count += 1

    conn.commit()
    return count
