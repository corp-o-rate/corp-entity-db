"""
Pydantic models for organization/entity database records.

v2 Schema Changes:
- Added SimplifiedLocationType enum for location categorization
- Added SourceTypeEnum for normalized source references
- Added RoleRecord and LocationRecord models for new tables
- Models support both TEXT-based v1 and FK-based v2 schemas
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Legacy source types for backward compatibility with v1 schema
SourceType = Literal["gleif", "sec_edgar", "companies_house", "wikipedia", "wikidata"]


class SourceTypeEnum(str, Enum):
    """
    Data source enumeration for v2 normalized schema.

    Used as foreign key reference to source_types table.
    """
    GLEIF = "gleif"               # id=1: GLEIF LEI data
    SEC_EDGAR = "sec_edgar"       # id=2: SEC EDGAR filings
    COMPANIES_HOUSE = "companies_house"  # id=3: UK Companies House
    WIKIDATA = "wikidata"         # id=4: Wikidata/Wikipedia


class SimplifiedLocationType(str, Enum):
    """
    Simplified location type categories for querying.

    Maps detailed Wikidata location types to canonical categories.
    Used for filtering searches (e.g., "find all cities").
    """
    CONTINENT = "continent"       # id=1: Continents (Q5107)
    COUNTRY = "country"           # id=2: Countries and sovereign states
    SUBDIVISION = "subdivision"   # id=3: States, provinces, regions, counties
    CITY = "city"                 # id=4: Cities, towns, municipalities, communes
    DISTRICT = "district"         # id=5: Districts, boroughs, neighborhoods
    HISTORIC = "historic"         # id=6: Former countries, historic territories
    OTHER = "other"               # id=7: Unclassified locations


class EntityType(str, Enum):
    """
    Classification of organization type.

    Used to distinguish between businesses, non-profits, government agencies, etc.
    """
    # Business entities
    BUSINESS = "business"  # General business/company
    FUND = "fund"  # Investment funds, ETFs, mutual funds
    BRANCH = "branch"  # Branch offices of companies

    # Non-profit/civil society
    NONPROFIT = "nonprofit"  # Non-profit organizations
    NGO = "ngo"  # Non-governmental organizations
    FOUNDATION = "foundation"  # Charitable foundations
    TRADE_UNION = "trade_union"  # Labor unions

    # Government/public sector
    GOVERNMENT = "government"  # Government agencies
    INTERNATIONAL_ORG = "international_org"  # UN, WHO, IMF, etc.
    POLITICAL_PARTY = "political_party"  # Political parties

    # Education/research
    EDUCATIONAL = "educational"  # Schools, universities
    RESEARCH = "research"  # Research institutes

    # Other organization types
    RELIGIOUS = "religious"  # Religious organizations
    SPORTS = "sports"  # Sports clubs/teams
    MEDIA = "media"  # Media companies, studios
    HEALTHCARE = "healthcare"  # Hospitals, healthcare orgs

    # Unknown/unclassified
    UNKNOWN = "unknown"  # Type not determined


class PersonType(str, Enum):
    """
    Classification of notable person type.

    Used for categorizing people in the person database.
    """
    EXECUTIVE = "executive"  # CEOs, board members, C-suite, founders
    POLITICIAN = "politician"  # Elected officials (presidents, MPs, mayors)
    GOVERNMENT = "government"  # Civil servants, diplomats, appointed officials
    MILITARY = "military"  # Military officers, armed forces personnel
    LEGAL = "legal"  # Judges, lawyers, legal professionals
    PROFESSIONAL = "professional"  # Known for their profession (doctors, engineers, architects)
    ACADEMIC = "academic"  # Professors, researchers, scientists
    ARTIST = "artist"  # Traditional creatives (musicians, actors, painters, writers)
    MEDIA = "media"  # Internet/social media personalities (YouTubers, influencers, podcasters)
    ATHLETE = "athlete"  # Sports figures
    JOURNALIST = "journalist"  # Reporters, news presenters, columnists
    ACTIVIST = "activist"  # Advocates, campaigners
    UNKNOWN = "unknown"  # Type not determined


class CompanyRecord(BaseModel):
    """
    An organization record for the embedding database.

    Used for storing and searching organizations by embedding similarity.
    Note: Class name kept as CompanyRecord for API compatibility.
    """
    name: str = Field(..., description="Organization name (used for embedding and display)")
    source: SourceType = Field(..., description="Data source")
    source_id: str = Field(..., description="Unique identifier from source (LEI, CIK, CH number)")
    region: str = Field(default="", description="Geographic region/country (e.g., 'UK', 'US', 'DE')")
    entity_type: EntityType = Field(default=EntityType.UNKNOWN, description="Organization type classification")
    from_date: Optional[str] = Field(default=None, description="Start date (ISO format YYYY-MM-DD)")
    to_date: Optional[str] = Field(default=None, description="End date (ISO format YYYY-MM-DD)")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")
    alias_source: Optional[str] = Field(default=None, description="Alias source type name (e.g., 'wikidata_alias', 'sec_ticker'), NULL for primary records")
    alias_source_identifier: Optional[str] = Field(default=None, description="Identifier from alias dataset, NULL for primary records")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        return f"{self.source}:{self.source_id}"

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        return {
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "region": self.region,
            "entity_type": self.entity_type.value,
            "from_date": self.from_date or "",
            "to_date": self.to_date or "",
            "record": self.record,
        }


# Person sources (same as org sources but without GLEIF)
PersonSourceType = Literal["wikidata", "sec_edgar", "companies_house"]


# =============================================================================
# ROLE RECORD MODEL (v2)
# =============================================================================


class RoleRecord(BaseModel):
    """
    A role/job title record for the roles table.

    Used for normalizing job titles across sources and enabling role-based search.
    Supports canonicalization to group equivalent roles (e.g., CEO, Chief Executive).
    """
    name: str = Field(..., description="Role/title name (e.g., 'Chief Executive Officer')")
    source: SourceType = Field(default="wikidata", description="Data source")
    source_id: Optional[str] = Field(default=None, description="Source identifier (e.g., Q484876 for CEO)")
    qid: Optional[int] = Field(default=None, description="Wikidata QID as integer (e.g., 484876)")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        if self.source_id:
            return f"{self.source}:{self.source_id}"
        return f"{self.source}:{self.name}"

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        return {
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id or "",
            "qid": self.qid,
            "record": self.record,
        }


# =============================================================================
# LOCATION RECORD MODEL (v2)
# =============================================================================


class LocationRecord(BaseModel):
    """
    A location/place record for the locations table.

    Used for storing geopolitical entities (countries, states, cities) with
    hierarchical relationships and type classification.
    """
    name: str = Field(..., description="Location name (e.g., 'United States', 'California')")
    source: SourceType = Field(default="wikidata", description="Data source")
    source_id: Optional[str] = Field(default=None, description="Source identifier (e.g., 'US', 'Q30')")
    qid: Optional[int] = Field(default=None, description="Wikidata QID as integer (e.g., 30 for USA)")
    location_type: str = Field(default="country", description="Detailed location type (e.g., 'us_state', 'city')")
    simplified_type: SimplifiedLocationType = Field(
        default=SimplifiedLocationType.COUNTRY,
        description="Simplified type for filtering"
    )
    parent_ids: list[int] = Field(
        default_factory=list,
        description="Parent location IDs in hierarchy (e.g., [country_id, state_id])"
    )
    from_date: Optional[str] = Field(default=None, description="Start date (ISO format YYYY-MM-DD)")
    to_date: Optional[str] = Field(default=None, description="End date (ISO format YYYY-MM-DD)")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        if self.source_id:
            return f"{self.source}:{self.source_id}"
        return f"{self.source}:{self.name}"

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        import json
        return {
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id or "",
            "qid": self.qid,
            "location_type": self.location_type,
            "parent_ids": json.dumps(self.parent_ids),
            "from_date": self.from_date or "",
            "to_date": self.to_date or "",
            "record": self.record,
        }


class PersonRecord(BaseModel):
    """
    A person record for the embedding database.

    Used for storing and searching notable people by embedding similarity.
    Supports people from Wikipedia/Wikidata with role/org context.
    """
    name: str = Field(..., description="Display name (used for embedding and display)")
    source: PersonSourceType = Field(default="wikidata", description="Data source")
    source_id: str = Field(..., description="Unique identifier from source (Wikidata QID)")
    country: str = Field(default="", description="Country code or name (e.g., 'US', 'Germany')")
    person_type: PersonType = Field(default=PersonType.UNKNOWN, description="Person type classification")
    known_for_role: str = Field(default="", description="Primary role (e.g., 'CEO', 'President')")
    known_for_org_id: Optional[int] = Field(default=None, description="Foreign key to organizations table")
    known_for_org_name: str = Field(default="", description="Transient display name from view COALESCE or importer, not persisted to DB")
    from_date: Optional[str] = Field(default=None, description="Start date of role (ISO format YYYY-MM-DD)")
    to_date: Optional[str] = Field(default=None, description="End date of role (ISO format YYYY-MM-DD)")
    birth_date: Optional[str] = Field(default=None, description="Date of birth (ISO format YYYY-MM-DD)")
    death_date: Optional[str] = Field(default=None, description="Date of death (ISO format YYYY-MM-DD) - if set, person is historic")
    record: dict[str, Any] = Field(default_factory=dict, description="Original record from source")
    canon_size: int = Field(default=1, description="Number of records in canonical group (popularity proxy)")
    matched_record: Optional["PersonRecord"] = Field(default=None, exclude=True, description="The actual indexed record that matched the query, attached when it differs from this canonical record")
    lookup_method: str = Field(default="", exclude=True, description="How this result was found: composite, name, identity")

    @property
    def canonical_id(self) -> str:
        """Generate canonical ID in format source:source_id."""
        return f"{self.source}:{self.source_id}"

    @property
    def is_historic(self) -> bool:
        """Return True if the person is deceased (has a death date)."""
        return self.death_date is not None and self.death_date != ""

    def model_dump_for_db(self) -> dict[str, Any]:
        """Convert to dict suitable for database storage."""
        return {
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "country": self.country,
            "person_type": self.person_type.value,
            "known_for_role": self.known_for_role,
            "known_for_org_id": self.known_for_org_id,
            "from_date": self.from_date or "",
            "to_date": self.to_date or "",
            "birth_date": self.birth_date or "",
            "death_date": self.death_date or "",
            "record": self.record,
        }

    # Person types whose identity is defined by their role at an organization
    _ORG_DEFINED_TYPES = {
        PersonType.EXECUTIVE, PersonType.POLITICIAN, PersonType.GOVERNMENT,
        PersonType.MILITARY, PersonType.LEGAL, PersonType.JOURNALIST,
        PersonType.PROFESSIONAL, PersonType.ACADEMIC,
    }

    # Preposition used to connect the role to the org for each org-defined type
    _TYPE_PREPOSITIONS: dict[PersonType, str] = {
        PersonType.EXECUTIVE: "of",
        PersonType.POLITICIAN: "of",
        PersonType.GOVERNMENT: "at",
        PersonType.MILITARY: "of",
        PersonType.LEGAL: "at",
        PersonType.JOURNALIST: "at",
        PersonType.PROFESSIONAL: "at",
        PersonType.ACADEMIC: "at",
    }

    # Display labels for identity-defined types
    _IDENTITY_LABELS: dict[PersonType, str] = {
        PersonType.ARTIST: "artist",
        PersonType.MEDIA: "media personality",
        PersonType.ATHLETE: "athlete",
        PersonType.ACTIVIST: "activist",
    }

    @staticmethod
    def _a_or_an(word: str) -> str:
        """Return 'an' if word starts with a vowel, else 'a'."""
        return "an" if word and word[0].lower() in "aeiou" else "a"

    def get_embedding_text(self) -> str:
        """Build natural language text for embedding.

        Org-defined types: "{name}, a {role} {prep} {org}"
        Identity-defined types: "{name}, a {type_label}"
        Degraded forms when fields are missing.
        """
        if self.person_type in self._ORG_DEFINED_TYPES:
            prep = self._TYPE_PREPOSITIONS[self.person_type]
            if self.known_for_role and self.known_for_org_name:
                article = self._a_or_an(self.known_for_role)
                return f"{self.name}, {article} {self.known_for_role} {prep} {self.known_for_org_name}"
            elif self.known_for_role:
                article = self._a_or_an(self.known_for_role)
                return f"{self.name}, {article} {self.known_for_role}"
            elif self.known_for_org_name:
                return f"{self.name} {prep} {self.known_for_org_name}"
            else:
                return self.name

        if self.person_type in self._IDENTITY_LABELS:
            label = self._IDENTITY_LABELS[self.person_type]
            article = self._a_or_an(label)
            return f"{self.name}, {article} {label}"

        # Unknown type — use role+org if available
        if self.known_for_role and self.known_for_org_name:
            article = self._a_or_an(self.known_for_role)
            return f"{self.name}, {article} {self.known_for_role} at {self.known_for_org_name}"
        elif self.known_for_role:
            article = self._a_or_an(self.known_for_role)
            return f"{self.name}, {article} {self.known_for_role}"
        elif self.known_for_org_name:
            return f"{self.name} at {self.known_for_org_name}"
        return self.name


class PersonMatch(BaseModel):
    """
    A person match result from embedding search.

    Returned by the person qualifier when finding potential matches.
    """
    query_name: str = Field(..., description="Name extracted from text (the search query)")
    record: PersonRecord = Field(..., description="The matched person record")
    source: PersonSourceType = Field(..., description="Data source of match")
    source_id: str = Field(..., description="Source identifier of match")
    canonical_id: str = Field(..., description="Canonical ID in format source:source_id")
    similarity_score: float = Field(..., description="Embedding similarity score (0-1)")
    llm_confirmed: bool = Field(default=False, description="Whether LLM confirmed this match")

    @property
    def name(self) -> str:
        """Get the matched person name."""
        return self.record.name

    @classmethod
    def from_record(
        cls,
        query_name: str,
        record: PersonRecord,
        similarity_score: float,
        llm_confirmed: bool = False,
    ) -> "PersonMatch":
        """Create a PersonMatch from a person record."""
        return cls(
            query_name=query_name,
            record=record,
            source=record.source,
            source_id=record.source_id,
            canonical_id=record.canonical_id,
            similarity_score=similarity_score,
            llm_confirmed=llm_confirmed,
        )


class CompanyMatch(BaseModel):
    """
    An organization match result from embedding search.

    Returned by the organization qualifier when finding potential matches.
    Note: Class name kept as CompanyMatch for API compatibility.
    """
    query_name: str = Field(..., description="Name extracted from text (the search query)")
    record: CompanyRecord = Field(..., description="The matched organization record")
    source: SourceType = Field(..., description="Data source of match")
    source_id: str = Field(..., description="Source identifier of match")
    canonical_id: str = Field(..., description="Canonical ID in format source:source_id")
    similarity_score: float = Field(..., description="Embedding similarity score (0-1)")
    llm_confirmed: bool = Field(default=False, description="Whether LLM confirmed this match")

    @property
    def name(self) -> str:
        """Get the matched organization name."""
        return self.record.name

    @classmethod
    def from_record(
        cls,
        query_name: str,
        record: CompanyRecord,
        similarity_score: float,
        llm_confirmed: bool = False,
    ) -> "CompanyMatch":
        """Create a CompanyMatch from an organization record."""
        return cls(
            query_name=query_name,
            record=record,
            source=record.source,
            source_id=record.source_id,
            canonical_id=record.canonical_id,
            similarity_score=similarity_score,
            llm_confirmed=llm_confirmed,
        )


class DatabaseStats(BaseModel):
    """Statistics about the organization database."""
    total_records: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    embedding_dimension: int = 0
    database_size_bytes: int = 0


class ResolvedOrganization(BaseModel):
    """
    Resolved/canonical organization information.

    Populated when resolving an organization mentioned in context
    against the organization database (GLEIF, SEC, Companies House, Wikidata).
    """
    canonical_name: str = Field(..., description="Canonical organization name")
    canonical_id: str = Field(..., description="Full canonical ID (e.g., 'LEI:549300XYZ', 'SEC-CIK:1234567')")
    source: str = Field(..., description="Source of resolution (e.g., 'gleif', 'sec_edgar', 'wikidata')")
    source_id: str = Field(..., description="ID in the source")
    region: Optional[str] = Field(None, description="Organization's region/jurisdiction")
    match_confidence: float = Field(default=1.0, description="Confidence in the match (0-1)")
    match_details: Optional[dict[str, Any]] = Field(None, description="Additional match details")
