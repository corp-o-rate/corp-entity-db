"""
Entity database for organizations, people, roles, and locations.

Provides embedding-based search using USearch HNSW indexes,
with data sourced from GLEIF, SEC EDGAR, Companies House, and Wikidata.
"""

__version__ = "0.1.0"

from corp_entity_db.store import (
    OrganizationDatabase,
    PersonDatabase,
    RolesDatabase,
    LocationsDatabase,
    get_database,
    get_person_database,
)

# Backwards compatibility alias
CompanyDatabase = OrganizationDatabase

from corp_entity_db.models import (
    CompanyRecord,
    CompanyMatch,
    PersonRecord,
    PersonMatch,
    DatabaseStats,
    RoleRecord,
    LocationRecord,
    EntityType,
    PersonType,
    SourceTypeEnum,
    SimplifiedLocationType,
    ResolvedOrganization,
)

from corp_entity_db.embeddings import CompanyEmbedder, get_embedder
from corp_entity_db.hub import (
    download_database,
    get_database_path,
    upload_database,
    upload_database_with_variants,
)
from corp_entity_db.resolver import OrganizationResolver, get_organization_resolver

__all__ = [
    # Organization models
    "CompanyRecord",
    "CompanyMatch",
    "DatabaseStats",
    "OrganizationDatabase",
    "CompanyDatabase",
    "get_database",
    # Person models
    "PersonRecord",
    "PersonMatch",
    "PersonType",
    "PersonDatabase",
    "get_person_database",
    # Other models
    "RoleRecord",
    "LocationRecord",
    "EntityType",
    "SourceTypeEnum",
    "SimplifiedLocationType",
    "ResolvedOrganization",
    # Databases
    "RolesDatabase",
    "LocationsDatabase",
    # Embedding
    "CompanyEmbedder",
    "get_embedder",
    # Hub
    "download_database",
    "get_database_path",
    "upload_database",
    "upload_database_with_variants",
    # Resolver
    "OrganizationResolver",
    "get_organization_resolver",
]
