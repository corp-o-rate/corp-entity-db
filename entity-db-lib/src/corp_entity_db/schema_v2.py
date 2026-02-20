"""
Database schema v2 with normalized foreign key references.

This module contains DDL statements for the normalized entity database schema
that replaces TEXT-based enum storage with INTEGER FK references to lookup tables.

Changes from v1:
- Enum tables: source_types, people_types, organization_types, location_types
- New tables: roles, locations, simplified_location_types
- organizations_v2: source_id FK, entity_type_id FK, region_id FK (to locations)
- people_v2: source_id FK, person_type_id FK, country_id FK, known_for_role_id FK
- Human-readable views with JOINs
"""

# =============================================================================
# ENUM LOOKUP TABLES
# =============================================================================

CREATE_SOURCE_TYPES = """
CREATE TABLE IF NOT EXISTS source_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
"""

CREATE_PEOPLE_TYPES = """
CREATE TABLE IF NOT EXISTS people_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
"""

CREATE_ORGANIZATION_TYPES = """
CREATE TABLE IF NOT EXISTS organization_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
"""

CREATE_SIMPLIFIED_LOCATION_TYPES = """
CREATE TABLE IF NOT EXISTS simplified_location_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
"""

CREATE_LOCATION_TYPES = """
CREATE TABLE IF NOT EXISTS location_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    qid INTEGER,
    simplified_id INTEGER NOT NULL,
    FOREIGN KEY (simplified_id) REFERENCES simplified_location_types(id)
);
"""

# =============================================================================
# ROLES TABLE
# =============================================================================

CREATE_ROLES = """
CREATE TABLE IF NOT EXISTS roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qid INTEGER,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    source_identifier TEXT,
    record TEXT NOT NULL DEFAULT '{}',
    canon_id INTEGER DEFAULT NULL,
    canon_size INTEGER DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES source_types(id),
    UNIQUE(name_normalized, source_id)
);
"""

CREATE_ROLES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);
CREATE INDEX IF NOT EXISTS idx_roles_name_normalized ON roles(name_normalized);
CREATE INDEX IF NOT EXISTS idx_roles_qid ON roles(qid);
CREATE INDEX IF NOT EXISTS idx_roles_source_id ON roles(source_id);
CREATE INDEX IF NOT EXISTS idx_roles_canon_id ON roles(canon_id);
"""

# =============================================================================
# LOCATIONS TABLE
# =============================================================================

CREATE_LOCATIONS = """
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qid INTEGER,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    source_identifier TEXT,
    parent_ids TEXT,
    location_type_id INTEGER NOT NULL,
    record TEXT NOT NULL DEFAULT '{}',
    from_date TEXT DEFAULT NULL,
    to_date TEXT DEFAULT NULL,
    canon_id INTEGER DEFAULT NULL,
    canon_size INTEGER DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES source_types(id),
    FOREIGN KEY (location_type_id) REFERENCES location_types(id),
    UNIQUE(source_identifier, source_id)
);
"""

CREATE_LOCATIONS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_locations_name ON locations(name);
CREATE INDEX IF NOT EXISTS idx_locations_name_normalized ON locations(name_normalized);
CREATE INDEX IF NOT EXISTS idx_locations_qid ON locations(qid);
CREATE INDEX IF NOT EXISTS idx_locations_source_id ON locations(source_id);
CREATE INDEX IF NOT EXISTS idx_locations_location_type_id ON locations(location_type_id);
CREATE INDEX IF NOT EXISTS idx_locations_canon_id ON locations(canon_id);
"""

# =============================================================================
# ORGANIZATIONS V2 TABLE
# =============================================================================

CREATE_ORGANIZATIONS_V2 = """
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qid INTEGER,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    source_identifier TEXT NOT NULL,
    region_id INTEGER,
    entity_type_id INTEGER NOT NULL DEFAULT 17,
    from_date TEXT DEFAULT NULL,
    to_date TEXT DEFAULT NULL,
    record TEXT NOT NULL DEFAULT '{}',
    embedding BLOB DEFAULT NULL,
    canon_id INTEGER DEFAULT NULL,
    canon_size INTEGER DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES source_types(id),
    FOREIGN KEY (region_id) REFERENCES locations(id),
    FOREIGN KEY (entity_type_id) REFERENCES organization_types(id),
    UNIQUE(source_identifier, source_id)
);
"""

CREATE_ORGANIZATIONS_V2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_orgs_name ON organizations(name);
CREATE INDEX IF NOT EXISTS idx_orgs_name_normalized ON organizations(name_normalized);
CREATE INDEX IF NOT EXISTS idx_orgs_qid ON organizations(qid);
CREATE INDEX IF NOT EXISTS idx_orgs_source_id ON organizations(source_id);
CREATE INDEX IF NOT EXISTS idx_orgs_source_identifier ON organizations(source_identifier);
CREATE INDEX IF NOT EXISTS idx_orgs_region_id ON organizations(region_id);
CREATE INDEX IF NOT EXISTS idx_orgs_entity_type_id ON organizations(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_orgs_canon_id ON organizations(canon_id);
"""

# =============================================================================
# PEOPLE V2 TABLE
# =============================================================================

CREATE_PEOPLE_V2 = """
CREATE TABLE IF NOT EXISTS people (
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
    embedding BLOB DEFAULT NULL,
    canon_id INTEGER DEFAULT NULL,
    canon_size INTEGER DEFAULT 1,
    FOREIGN KEY (source_id) REFERENCES source_types(id),
    FOREIGN KEY (country_id) REFERENCES locations(id),
    FOREIGN KEY (person_type_id) REFERENCES people_types(id),
    FOREIGN KEY (known_for_role_id) REFERENCES roles(id),
    FOREIGN KEY (known_for_org_id) REFERENCES organizations(id),
    UNIQUE(source_identifier, source_id, known_for_role_id, known_for_org_id, from_date)
);
"""

CREATE_PEOPLE_V2_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);
CREATE INDEX IF NOT EXISTS idx_people_name_normalized ON people(name_normalized);
CREATE INDEX IF NOT EXISTS idx_people_qid ON people(qid);
CREATE INDEX IF NOT EXISTS idx_people_source_id ON people(source_id);
CREATE INDEX IF NOT EXISTS idx_people_source_identifier ON people(source_identifier);
CREATE INDEX IF NOT EXISTS idx_people_country_id ON people(country_id);
CREATE INDEX IF NOT EXISTS idx_people_person_type_id ON people(person_type_id);
CREATE INDEX IF NOT EXISTS idx_people_known_for_role_id ON people(known_for_role_id);
CREATE INDEX IF NOT EXISTS idx_people_known_for_org_id ON people(known_for_org_id);
CREATE INDEX IF NOT EXISTS idx_people_canon_id ON people(canon_id);
"""

# =============================================================================
# METADATA TABLE
# =============================================================================

CREATE_DB_INFO = """
CREATE TABLE IF NOT EXISTS db_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# =============================================================================
# HUMAN-READABLE VIEWS
# =============================================================================

CREATE_ORGANIZATIONS_VIEW = """
CREATE VIEW IF NOT EXISTS organizations_view AS
SELECT
    o.id,
    o.qid,
    o.name,
    o.name_normalized,
    s.name as source,
    o.source_identifier,
    l.name as region,
    slt.name as region_type,
    ot.name as entity_type,
    o.from_date,
    o.to_date,
    o.canon_id,
    o.canon_size
FROM organizations o
JOIN source_types s ON o.source_id = s.id
LEFT JOIN locations l ON o.region_id = l.id
LEFT JOIN location_types lt ON l.location_type_id = lt.id
LEFT JOIN simplified_location_types slt ON lt.simplified_id = slt.id
JOIN organization_types ot ON o.entity_type_id = ot.id;
"""

CREATE_PEOPLE_VIEW = """
CREATE VIEW IF NOT EXISTS people_view AS
SELECT
    p.id,
    p.qid,
    p.name,
    p.name_normalized,
    s.name as source,
    p.source_identifier,
    l.name as country,
    pt.name as person_type,
    r.name as known_for_role,
    kfo.name as known_for_org,
    p.known_for_org_id,
    p.from_date,
    p.to_date,
    p.birth_date,
    p.death_date,
    p.canon_id,
    p.canon_size
FROM people p
JOIN source_types s ON p.source_id = s.id
LEFT JOIN locations l ON p.country_id = l.id
JOIN people_types pt ON p.person_type_id = pt.id
LEFT JOIN roles r ON p.known_for_role_id = r.id
LEFT JOIN organizations kfo ON p.known_for_org_id = kfo.id;
"""

CREATE_ROLES_VIEW = """
CREATE VIEW IF NOT EXISTS roles_view AS
SELECT
    r.id,
    r.qid,
    r.name,
    r.name_normalized,
    s.name as source,
    r.source_identifier,
    r.canon_id,
    r.canon_size
FROM roles r
JOIN source_types s ON r.source_id = s.id;
"""

CREATE_LOCATIONS_VIEW = """
CREATE VIEW IF NOT EXISTS locations_view AS
SELECT
    l.id,
    l.qid,
    l.name,
    l.name_normalized,
    s.name as source,
    l.source_identifier,
    l.parent_ids,
    lt.name as location_type,
    slt.name as simplified_type,
    l.from_date,
    l.to_date,
    l.canon_id,
    l.canon_size
FROM locations l
JOIN source_types s ON l.source_id = s.id
JOIN location_types lt ON l.location_type_id = lt.id
JOIN simplified_location_types slt ON lt.simplified_id = slt.id;
"""

# =============================================================================
# ALL DDL STATEMENTS IN ORDER
# =============================================================================

ALL_DDL_STATEMENTS = [
    # Enum tables first (no dependencies)
    CREATE_SOURCE_TYPES,
    CREATE_PEOPLE_TYPES,
    CREATE_ORGANIZATION_TYPES,
    CREATE_SIMPLIFIED_LOCATION_TYPES,
    CREATE_LOCATION_TYPES,
    # New entity tables
    CREATE_ROLES,
    CREATE_ROLES_INDEXES,
    CREATE_LOCATIONS,
    CREATE_LOCATIONS_INDEXES,
    # Main entity tables
    CREATE_ORGANIZATIONS_V2,
    CREATE_ORGANIZATIONS_V2_INDEXES,
    CREATE_PEOPLE_V2,
    CREATE_PEOPLE_V2_INDEXES,
    # Metadata
    CREATE_DB_INFO,
]

VIEW_DDL_STATEMENTS = [
    CREATE_ORGANIZATIONS_VIEW,
    CREATE_PEOPLE_VIEW,
    CREATE_ROLES_VIEW,
    CREATE_LOCATIONS_VIEW,
]


def create_all_tables(conn, embedding_dim: int = 768) -> None:
    """
    Create all v2 schema tables.

    Embeddings are stored as BLOB columns directly in the organizations and people tables.

    Args:
        conn: SQLite connection
        embedding_dim: Dimension for embedding vectors (unused, kept for API compat)
    """
    for ddl in ALL_DDL_STATEMENTS:
        for statement in ddl.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)

    # Create views
    for ddl in VIEW_DDL_STATEMENTS:
        conn.execute(ddl)

    # Set schema version
    conn.execute("INSERT OR REPLACE INTO db_info (key, value) VALUES ('schema_version', '3')")

    conn.commit()
