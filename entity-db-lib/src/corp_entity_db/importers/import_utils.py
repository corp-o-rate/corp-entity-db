"""
Shared utilities for v2 database importers.

Provides helper functions for resolving locations, roles, and QIDs
to their normalized FK references in the v2 schema.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..store import LocationsDatabase, RolesDatabase

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


def format_qid(qid_int: Optional[int]) -> Optional[str]:
    """
    Format an integer QID back to string format.

    Args:
        qid_int: Integer QID (e.g., 12345)

    Returns:
        String QID like "Q12345" or None
    """
    if qid_int is None:
        return None
    return f"Q{qid_int}"


def normalize_name(name: str) -> str:
    """
    Normalize a name for database lookup.

    Args:
        name: Name to normalize

    Returns:
        Lowercase, stripped name
    """
    if not name:
        return ""
    return name.lower().strip()


def get_or_create_location(
    locations_db: "LocationsDatabase",
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
        locations_db: LocationsDatabase instance
        name: Location name
        location_type_id: FK to location_types table
        source_id: FK to source_types table
        qid: Optional Wikidata QID as integer
        source_identifier: Optional source-specific identifier
        parent_ids: Optional list of parent location IDs

    Returns:
        Location ID
    """
    return locations_db.get_or_create(
        name=name,
        location_type_id=location_type_id,
        source_id=source_id,
        qid=qid,
        source_identifier=source_identifier,
        parent_ids=parent_ids,
    )


def get_or_create_role(
    roles_db: "RolesDatabase",
    name: str,
    source_id: int = 4,  # wikidata
    qid: Optional[int] = None,
    source_identifier: Optional[str] = None,
) -> int:
    """
    Get or create a role record.

    Args:
        roles_db: RolesDatabase instance
        name: Role/title name
        source_id: FK to source_types table
        qid: Optional Wikidata QID as integer
        source_identifier: Optional source-specific identifier

    Returns:
        Role ID
    """
    return roles_db.get_or_create(
        name=name,
        source_id=source_id,
        qid=qid,
        source_identifier=source_identifier,
    )


def resolve_country_to_location_id(
    locations_db: "LocationsDatabase",
    country_text: str,
) -> Optional[int]:
    """
    Resolve a country name/code to a location ID.

    Args:
        locations_db: LocationsDatabase instance
        country_text: Country code (e.g., "US") or name (e.g., "United States")

    Returns:
        Location ID or None if not found
    """
    if not country_text:
        return None

    return locations_db.resolve_region_text(country_text)


def get_source_id(source_name: str) -> int:
    """
    Get source_id for a source name.

    Args:
        source_name: Source name (e.g., "gleif", "sec_edgar")

    Returns:
        Source ID (1-4)
    """
    from ..seed_data import SOURCE_NAME_TO_ID
    return SOURCE_NAME_TO_ID.get(source_name, 4)  # default to wikidata


def get_source_name(source_id: int) -> str:
    """
    Get source name for a source_id.

    Args:
        source_id: Source ID (1-4)

    Returns:
        Source name
    """
    from ..seed_data import SOURCE_ID_TO_NAME
    return SOURCE_ID_TO_NAME.get(source_id, "wikidata")


def get_entity_type_id(entity_type_name: str) -> int:
    """
    Get entity_type_id for an entity type name.

    Args:
        entity_type_name: Entity type name (e.g., "business", "fund")

    Returns:
        Entity type ID (1-17)
    """
    from ..seed_data import ORG_TYPE_NAME_TO_ID
    return ORG_TYPE_NAME_TO_ID.get(entity_type_name, 17)  # default to unknown


def get_entity_type_name(entity_type_id: int) -> str:
    """
    Get entity type name for an entity_type_id.

    Args:
        entity_type_id: Entity type ID (1-17)

    Returns:
        Entity type name
    """
    from ..seed_data import ORG_TYPE_ID_TO_NAME
    return ORG_TYPE_ID_TO_NAME.get(entity_type_id, "unknown")


def get_person_type_id(person_type_name: str) -> int:
    """
    Get person_type_id for a person type name.

    Args:
        person_type_name: Person type name (e.g., "executive", "politician")

    Returns:
        Person type ID (1-15)
    """
    from ..seed_data import PEOPLE_TYPE_NAME_TO_ID
    return PEOPLE_TYPE_NAME_TO_ID.get(person_type_name, 15)  # default to unknown


def get_person_type_name(person_type_id: int) -> str:
    """
    Get person type name for a person_type_id.

    Args:
        person_type_id: Person type ID (1-15)

    Returns:
        Person type name
    """
    from ..seed_data import PEOPLE_TYPE_ID_TO_NAME
    return PEOPLE_TYPE_ID_TO_NAME.get(person_type_id, "unknown")


def get_location_type_id(location_type_name: str) -> int:
    """
    Get location_type_id for a location type name.

    Args:
        location_type_name: Location type name (e.g., "country", "city")

    Returns:
        Location type ID
    """
    from ..seed_data import LOCATION_TYPE_NAME_TO_ID
    return LOCATION_TYPE_NAME_TO_ID.get(location_type_name, 36)  # default to other


def get_location_type_id_from_qid(wikidata_qid: int) -> int:
    """
    Get location_type_id from a Wikidata P31 QID.

    Args:
        wikidata_qid: Wikidata instance-of QID (e.g., 515 for city)

    Returns:
        Location type ID (defaults to 36 = other)
    """
    from ..seed_data import LOCATION_TYPE_QID_TO_ID
    return LOCATION_TYPE_QID_TO_ID.get(wikidata_qid, 36)  # default to other
