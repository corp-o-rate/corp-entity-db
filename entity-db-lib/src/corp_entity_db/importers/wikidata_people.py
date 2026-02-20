"""
Wikidata importer for the person database.

Imports notable people data from Wikidata using SPARQL queries
into the embedding database for person name matching.

Uses a two-phase approach for reliability:
1. Bulk fetch: Simple queries to get QID + name + country (fast, no timeouts)
2. Enrich: Targeted per-person queries for role/org/dates (resumable)

Notable people are those with English Wikipedia articles, ensuring
a basic level of notability.

Query categories (organized by PersonType):
- executives: Business executives (CEOs, CFOs, etc.)
- politicians: Politicians and diplomats
- athletes: Sports figures
- artists: Actors, musicians, directors
- academics: Professors and researchers
- scientists: Scientists and inventors
- journalists: Media personalities
- entrepreneurs: Founders and business owners

Uses the public Wikidata Query Service endpoint.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType, PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# =============================================================================
# BULK QUERIES - Simple, fast queries for initial import (no role/org/dates)
# Uses rdfs:label instead of SERVICE wikibase:label for better performance
# Each query targets a single role/occupation for speed
# =============================================================================

# Template for position-held queries (P39) - for executives, politicians
# Matches people who held a position that IS the role, or is an INSTANCE OF the role
# {role_qid} = single role QID, {seed} = unique seed, {limit} = batch limit
POSITION_QUERY_TEMPLATE = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P39 ?position .
  {{ ?position wdt:P31 wd:{role_qid} . }} UNION {{ VALUES ?position {{ wd:{role_qid} }} }}
  ?person rdfs:label ?personLabel FILTER(LANG(?personLabel) = "en") .
  OPTIONAL {{ ?person wdt:P27 ?country . ?country rdfs:label ?countryLabel FILTER(LANG(?countryLabel) = "en") . }}
  OPTIONAL {{ ?person schema:description ?description FILTER(LANG(?description) = "en") }}
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
}}
ORDER BY MD5(CONCAT(STR(?person), "{seed}"))
LIMIT {limit}
"""

# Template for occupation queries (P106) - for athletes, artists, etc.
# {occupation_qid} = single occupation QID, {seed} = unique seed, {limit} = batch limit
OCCUPATION_QUERY_TEMPLATE = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P106 wd:{occupation_qid} .
  ?person rdfs:label ?personLabel FILTER(LANG(?personLabel) = "en") .
  OPTIONAL {{ ?person wdt:P27 ?country . ?country rdfs:label ?countryLabel FILTER(LANG(?countryLabel) = "en") . }}
  OPTIONAL {{ ?person schema:description ?description FILTER(LANG(?description) = "en") }}
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
}}
ORDER BY MD5(CONCAT(STR(?person), "{seed}"))
LIMIT {limit}
"""

# Template for founder queries (P112) - for entrepreneurs
# {seed} = unique seed, {limit} = batch limit
FOUNDER_QUERY_TEMPLATE = """
SELECT DISTINCT ?person ?personLabel ?countryLabel ?description WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?org wdt:P112 ?person .
  ?person rdfs:label ?personLabel FILTER(LANG(?personLabel) = "en") .
  OPTIONAL {{ ?person wdt:P27 ?country . ?country rdfs:label ?countryLabel FILTER(LANG(?countryLabel) = "en") . }}
  OPTIONAL {{ ?person schema:description ?description FILTER(LANG(?description) = "en") }}
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> .
}}
ORDER BY MD5(CONCAT(STR(?person), "{seed}"))
LIMIT {limit}
"""

# Role QIDs for executives (position held - P39)
EXECUTIVE_ROLES = [
    "Q484876",    # CEO
    "Q623279",    # CFO
    "Q1502675",   # COO
    "Q935019",    # CTO
    "Q1057716",   # CIO
    "Q2140589",   # CMO
    "Q1115042",   # chairperson
    "Q4720025",   # board of directors member
    "Q60432825",  # chief human resources officer
    "Q15967139",  # chief compliance officer
    "Q15729310",  # chief risk officer
    "Q47523568",  # chief legal officer
    "Q258557",    # board chair
    "Q114863313", # chief sustainability officer
    "Q726114",    # company president
    "Q1372944",   # managing director
    "Q18918145",  # chief commercial officer
    "Q1057569",   # chief strategy officer
    "Q24058752",  # chief product officer
    "Q3578048",   # vice president
    "Q476675",    # business executive (generic)
    "Q5441744",   # finance director
    "Q4188234",   # general manager
    "Q38844673",  # chief data officer
    "Q97273203",  # chief digital officer
    "Q60715311",  # chief growth officer
    "Q3563879",   # treasurer
    "Q3505845",   # corporate secretary
]

# Role QIDs for politicians (position held - P39)
POLITICIAN_ROLES = [
    "Q30461",     # president
    "Q14212",     # prime minister
    "Q83307",     # minister
    "Q2285706",   # head of government
    "Q4175034",   # legislator
    "Q486839",    # member of parliament
    "Q193391",    # member of national legislature
    "Q212071",    # mayor
    "Q382617",    # governor
    "Q116",       # monarch
    "Q484529",    # member of congress
]

# Note: Politicians with generic position types (like "public office") may not be found
# because querying all public office holders times out. This includes some mayors
# whose positions are typed as "public office" rather than "mayor".

# Occupation QIDs for athletes (P106)
ATHLETE_OCCUPATIONS = [
    "Q2066131",   # athlete
    "Q937857",    # football player
    "Q3665646",   # basketball player
    "Q10871364",  # baseball player
    "Q19204627",  # ice hockey player
    "Q10843402",  # tennis player
    "Q13381376",  # golfer
    "Q11338576",  # boxer
    "Q10873124",  # swimmer
]

# Occupation QIDs for artists (P106)
ARTIST_OCCUPATIONS = [
    "Q33999",     # actor
    "Q177220",    # singer
    "Q639669",    # musician
    "Q2526255",   # film director
    "Q36180",     # writer
    "Q483501",    # artist
    "Q488205",    # singer-songwriter
    "Q753110",    # songwriter
    "Q2405480",   # voice actor
    "Q10800557",  # film actor
]

# Occupation QIDs for academics (P106)
ACADEMIC_OCCUPATIONS = [
    "Q121594",    # professor
    "Q3400985",   # academic
    "Q1622272",   # university professor
]

# Occupation QIDs for scientists (P106)
SCIENTIST_OCCUPATIONS = [
    "Q901",       # scientist
    "Q1650915",   # researcher
    "Q169470",    # physicist
    "Q593644",    # chemist
    "Q864503",    # biologist
    "Q11063",     # astronomer
]

# Occupation QIDs for journalists (P106)
JOURNALIST_OCCUPATIONS = [
    "Q1930187",   # journalist
    "Q13590141",  # news presenter
    "Q947873",    # television presenter
    "Q4263842",   # columnist
]

# Occupation QIDs for activists (P106)
ACTIVIST_OCCUPATIONS = [
    "Q15253558",  # activist
    "Q11631410",  # human rights activist
    "Q18939491",  # environmental activist
]

# Mapping query type to role/occupation lists and query template type
# Each entry can have multiple query groups to combine different approaches
QUERY_TYPE_CONFIG: dict[str, list[dict]] = {
    "executive": [
        {"template": "position", "items": EXECUTIVE_ROLES},
    ],
    "politician": [
        {"template": "position", "items": POLITICIAN_ROLES},
    ],
    "athlete": [
        {"template": "occupation", "items": ATHLETE_OCCUPATIONS},
    ],
    "artist": [
        {"template": "occupation", "items": ARTIST_OCCUPATIONS},
    ],
    "academic": [
        {"template": "occupation", "items": ACADEMIC_OCCUPATIONS},
    ],
    "scientist": [
        {"template": "occupation", "items": SCIENTIST_OCCUPATIONS},
    ],
    "journalist": [
        {"template": "occupation", "items": JOURNALIST_OCCUPATIONS},
    ],
    "activist": [
        {"template": "occupation", "items": ACTIVIST_OCCUPATIONS},
    ],
    "entrepreneur": [
        {"template": "founder", "items": []},  # No items, uses special template
    ],
}

# Mapping query type to PersonType
QUERY_TYPE_TO_PERSON_TYPE: dict[str, PersonType] = {
    "executive": PersonType.EXECUTIVE,
    "politician": PersonType.POLITICIAN,
    "athlete": PersonType.ATHLETE,
    "artist": PersonType.ARTIST,
    "academic": PersonType.ACADEMIC,
    "scientist": PersonType.SCIENTIST,
    "journalist": PersonType.JOURNALIST,
    "entrepreneur": PersonType.ENTREPRENEUR,
    "activist": PersonType.ACTIVIST,
}


class WikidataPeopleImporter:
    """
    Importer for Wikidata person data.

    Uses SPARQL queries against the public Wikidata Query Service
    to fetch notable people including executives, politicians, athletes, etc.

    Query types:
    - executive: Business executives (CEOs, CFOs, etc.)
    - politician: Politicians and diplomats
    - athlete: Sports figures
    - artist: Actors, musicians, directors, writers
    - academic: Professors and researchers
    - scientist: Scientists and inventors
    - journalist: Media personalities
    - entrepreneur: Company founders
    - activist: Activists and advocates
    """

    def __init__(
        self,
        batch_size: int = 5000,
        delay_seconds: float = 2.0,
        timeout: int = 120,
        max_retries: int = 3,
        min_batch_size: int = 50,
    ):
        """
        Initialize the Wikidata people importer.

        Args:
            batch_size: Number of records to fetch per SPARQL query (default 5000)
            delay_seconds: Delay between requests to be polite to the endpoint
            timeout: HTTP timeout in seconds (default 120)
            max_retries: Maximum retries per batch on timeout (default 3)
            min_batch_size: Minimum batch size before giving up (default 50)
        """
        self._batch_size = batch_size
        self._delay = delay_seconds
        self._timeout = timeout
        self._max_retries = max_retries
        self._min_batch_size = min_batch_size
        # Track discovered organizations: org_qid -> org_label
        self._discovered_orgs: dict[str, str] = {}

    def import_from_sparql(
        self,
        limit: Optional[int] = None,
        query_type: str = "executive",
        import_all: bool = False,
        convergence_threshold: int = 5,
    ) -> Iterator[PersonRecord]:
        """
        Import person records from Wikidata via SPARQL (bulk fetch phase).

        This performs the fast bulk import with minimal data (QID, name, country).
        Use enrich_people_batch() afterwards to add role/org/dates.

        Iterates through each role/occupation individually for faster queries,
        using random sampling with convergence detection per role.

        Args:
            limit: Optional limit on total records
            query_type: Which query to use (executive, politician, athlete, etc.)
            import_all: If True, run all query types sequentially
            convergence_threshold: Stop after this many consecutive batches with no new records per role

        Yields:
            PersonRecord for each person (without role/org - use enrich to add)
        """
        if import_all:
            yield from self._import_all_types(limit)
            return

        if query_type not in QUERY_TYPE_CONFIG:
            raise ValueError(f"Unknown query type: {query_type}. Use one of: {list(QUERY_TYPE_CONFIG.keys())}")

        config_groups = QUERY_TYPE_CONFIG[query_type]
        person_type = QUERY_TYPE_TO_PERSON_TYPE.get(query_type, PersonType.UNKNOWN)

        logger.info(f"Starting Wikidata bulk import (query_type={query_type}, person_type={person_type.value})...")

        total_count = 0
        # Track seen QIDs to deduplicate across all roles
        seen_qids: set[str] = set()

        # Iterate through each config group (e.g., position queries + occupation queries)
        for config in config_groups:
            if limit and total_count >= limit:
                break

            template_type = config["template"]
            items = config["items"]

            # For founder template, run a single query
            if template_type == "founder":
                for record in self._import_single_template(
                    template=FOUNDER_QUERY_TEMPLATE,
                    template_params={},
                    person_type=person_type,
                    seen_qids=seen_qids,
                    limit=(limit - total_count) if limit else None,
                    convergence_threshold=convergence_threshold,
                    role_name="founder",
                ):
                    total_count += 1
                    yield record
                continue

            # Select the right template
            if template_type == "position":
                template = POSITION_QUERY_TEMPLATE
                param_name = "role_qid"
            else:  # occupation
                template = OCCUPATION_QUERY_TEMPLATE
                param_name = "occupation_qid"

            # Iterate through each role/occupation in this group
            for item_qid in items:
                if limit and total_count >= limit:
                    break

                remaining = (limit - total_count) if limit else None
                role_count = 0

                for record in self._import_single_template(
                    template=template,
                    template_params={param_name: item_qid},
                    person_type=person_type,
                    seen_qids=seen_qids,
                    limit=remaining,
                    convergence_threshold=convergence_threshold,
                    role_name=item_qid,
                ):
                    role_count += 1
                    total_count += 1
                    yield record

                logger.info(f"Role {item_qid}: {role_count} new (total: {total_count})")

        logger.info(f"Completed Wikidata bulk import: {total_count} records (use enrich to add role/org)")

    def _import_single_template(
        self,
        template: str,
        template_params: dict[str, str],
        person_type: PersonType,
        seen_qids: set[str],
        limit: Optional[int],
        convergence_threshold: int,
        role_name: str,
    ) -> Iterator[PersonRecord]:
        """
        Import from a single role/occupation using random sampling with convergence.

        Args:
            template: SPARQL query template
            template_params: Parameters to format into template (role_qid or occupation_qid)
            person_type: PersonType to assign to records
            seen_qids: Set of already-seen QIDs (shared across roles)
            limit: Optional limit on records from this role
            convergence_threshold: Stop after this many consecutive empty batches
            role_name: Name for logging

        Yields:
            PersonRecord for each new person found
        """
        batch_num = 0
        total_count = 0
        current_batch_size = self._batch_size
        consecutive_empty_batches = 0

        logger.info(f"Querying role {role_name}...")

        while True:
            if limit and total_count >= limit:
                break

            batch_num += 1
            batch_limit = min(current_batch_size, (limit - total_count) if limit else current_batch_size)

            # Generate unique seed for this batch
            batch_seed = f"{role_name}_{batch_num}_{int(time.time() * 1000)}"

            # Build query
            query = template.format(
                **template_params,
                seed=batch_seed,
                limit=batch_limit,
            )

            # Execute with retries
            results = None
            retries = 0
            retry_batch_size = batch_limit

            while retries <= self._max_retries:
                try:
                    # Rebuild query with potentially smaller batch size
                    if retry_batch_size != batch_limit:
                        query = template.format(
                            **template_params,
                            seed=batch_seed,
                            limit=retry_batch_size,
                        )
                    results = self._execute_sparql(query)
                    if retry_batch_size < current_batch_size:
                        current_batch_size = retry_batch_size
                    break
                except Exception as e:
                    is_timeout = "timeout" in str(e).lower() or "504" in str(e) or "503" in str(e)
                    if is_timeout and retry_batch_size > self._min_batch_size:
                        retries += 1
                        retry_batch_size = max(retry_batch_size // 2, self._min_batch_size)
                        wait_time = self._delay * (2 ** retries)
                        logger.warning(
                            f"Timeout on {role_name} batch #{batch_num}, retry {retries}/{self._max_retries} "
                            f"with batch_size={retry_batch_size} after {wait_time:.1f}s wait"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"SPARQL query failed on {role_name} batch #{batch_num}: {e}")
                        break

            if results is None:
                logger.warning(f"Giving up on {role_name} after {retries} retries")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= convergence_threshold:
                    logger.debug(f"Role {role_name}: convergence after {batch_num} batches")
                    break
                continue

            batch_count = 0
            for binding in bindings:
                if limit and total_count >= limit:
                    break

                record, skip_reason = self._parse_bulk_binding(binding, person_type=person_type)
                if record is None:
                    continue

                # Deduplicate
                if record.source_id in seen_qids:
                    continue

                seen_qids.add(record.source_id)
                total_count += 1
                batch_count += 1
                yield record

            # Check convergence
            if batch_count == 0:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= convergence_threshold:
                    logger.debug(f"Role {role_name}: convergence after {batch_num} batches")
                    break
            else:
                consecutive_empty_batches = 0

            # Rate limit
            if self._delay > 0:
                time.sleep(self._delay)

    def _import_all_types(self, limit: Optional[int]) -> Iterator[PersonRecord]:
        """Import from all query types sequentially, deduplicating across types."""
        # Track seen QIDs across all types
        seen_qids: set[str] = set()
        total_count = 0

        # Calculate per-type limits if a total limit is set
        num_types = len(QUERY_TYPE_CONFIG)
        per_type_limit = limit // num_types if limit else None

        for query_type in QUERY_TYPE_CONFIG:
            logger.info(f"=== Importing people: {query_type} ===")
            type_count = 0
            skipped_count = 0

            for record in self.import_from_sparql(limit=per_type_limit, query_type=query_type):
                if record.source_id in seen_qids:
                    skipped_count += 1
                    continue

                seen_qids.add(record.source_id)
                total_count += 1
                type_count += 1
                yield record

                if limit and total_count >= limit:
                    logger.info(f"Reached total limit of {limit} records")
                    return

            logger.info(
                f"Got {type_count} new from {query_type}, skipped {skipped_count} (total: {total_count})"
            )

        logger.info(f"Completed all query types: {total_count} total people records")

    @staticmethod
    def _parse_wikidata_date(date_str: str) -> Optional[str]:
        """
        Parse a Wikidata date string into ISO format (YYYY-MM-DD).

        Wikidata returns dates like "2020-01-15T00:00:00Z" or just "2020".
        Returns None if the date cannot be parsed.
        """
        if not date_str:
            return None
        # Handle ISO datetime format (e.g., "2020-01-15T00:00:00Z")
        if "T" in date_str:
            return date_str.split("T")[0]
        # Handle year-only format (e.g., "2020")
        if len(date_str) == 4 and date_str.isdigit():
            return f"{date_str}-01-01"
        # Return as-is if it looks like a date
        if len(date_str) >= 4:
            return date_str[:10]  # Take first 10 chars (YYYY-MM-DD)
        return None

    def _execute_sparql(self, query: str) -> dict[str, Any]:
        """Execute a SPARQL query against Wikidata."""
        params = urllib.parse.urlencode({
            "query": query,
            "format": "json",
        })

        url = f"{WIKIDATA_SPARQL_URL}?{params}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "corp-extractor/1.0 (person database builder)",
            }
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_bulk_binding(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> tuple[Optional[PersonRecord], Optional[str]]:
        """
        Parse a bulk SPARQL result binding into a PersonRecord.

        Bulk bindings only have: person, personLabel, countryLabel, description.
        Role/org/dates are NOT included - use enrich methods to add them later.

        Returns:
            Tuple of (PersonRecord or None, skip_reason or None)
        """
        try:
            # Get Wikidata entity ID
            person_uri = binding.get("person", {}).get("value", "")
            if not person_uri:
                return None, "missing person URI"

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = person_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None, f"invalid Wikidata ID format: {wikidata_id}"

            # Get label
            label = binding.get("personLabel", {}).get("value", "")
            if not label:
                return None, f"{wikidata_id}: no label"
            if label == wikidata_id:
                return None, f"{wikidata_id}: no English label (label equals QID)"

            # Get optional fields from bulk query
            country = binding.get("countryLabel", {}).get("value", "")
            description = binding.get("description", {}).get("value", "")

            # Build minimal record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if country:
                record_data["country"] = country
            if description:
                record_data["description"] = description

            return PersonRecord(
                name=label.strip(),
                source="wikidata",
                source_id=wikidata_id,
                country=country or "",
                person_type=person_type,
                known_for_role="",  # To be enriched later
                known_for_org_name="",   # To be enriched later
                from_date=None,     # To be enriched later
                to_date=None,       # To be enriched later
                record=record_data,
            ), None

        except Exception as e:
            return None, f"parse error: {e}"

    def _parse_binding_with_reason(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> tuple[Optional[PersonRecord], Optional[str]]:
        """
        Parse a SPARQL result binding into a PersonRecord.

        Returns:
            Tuple of (PersonRecord or None, skip_reason or None)
        """
        try:
            # Get Wikidata entity ID
            person_uri = binding.get("person", {}).get("value", "")
            if not person_uri:
                return None, "missing person URI"

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = person_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None, f"invalid Wikidata ID format: {wikidata_id}"

            # Get label
            label = binding.get("personLabel", {}).get("value", "")
            if not label:
                return None, f"{wikidata_id}: no label"
            if label == wikidata_id:
                return None, f"{wikidata_id}: no English label (label equals QID)"

            # Get optional fields
            country = binding.get("countryLabel", {}).get("value", "")
            role = binding.get("roleLabel", {}).get("value", "")
            org_label = binding.get("orgLabel", {}).get("value", "")
            org_uri = binding.get("org", {}).get("value", "")
            description = binding.get("description", {}).get("value", "")

            # Extract org QID from URI (e.g., "http://www.wikidata.org/entity/Q715583" -> "Q715583")
            org_qid = ""
            if org_uri:
                org_qid = org_uri.split("/")[-1]
                if not org_qid.startswith("Q"):
                    org_qid = ""

            # Get dates (Wikidata returns ISO datetime, extract just the date part)
            start_date_raw = binding.get("startDate", {}).get("value", "")
            end_date_raw = binding.get("endDate", {}).get("value", "")
            from_date = WikidataPeopleImporter._parse_wikidata_date(start_date_raw)
            to_date = WikidataPeopleImporter._parse_wikidata_date(end_date_raw)

            # Clean up role and org label (remove QID if it's the same as the label)
            if role and role.startswith("Q"):
                role = ""
            if org_label and org_label.startswith("Q"):
                org_label = ""

            # Track discovered organization if we have both QID and label
            if org_qid and org_label:
                self._discovered_orgs[org_qid] = org_label

            # Build record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if country:
                record_data["country"] = country
            if role:
                record_data["role"] = role
            if org_label:
                record_data["org"] = org_label
            if org_qid:
                record_data["org_qid"] = org_qid
            if description:
                record_data["description"] = description
            if from_date:
                record_data["from_date"] = from_date
            if to_date:
                record_data["to_date"] = to_date

            return PersonRecord(
                name=label.strip(),
                source="wikidata",
                source_id=wikidata_id,
                country=country or "",
                person_type=person_type,
                known_for_role=role or "",
                known_for_org_name=org_label or "",
                from_date=from_date,
                to_date=to_date,
                record=record_data,
            ), None

        except Exception as e:
            return None, f"parse error: {e}"

    def _parse_binding(
        self,
        binding: dict[str, Any],
        person_type: PersonType = PersonType.UNKNOWN,
    ) -> Optional[PersonRecord]:
        """Parse a SPARQL result binding into a PersonRecord (legacy wrapper)."""
        record, _ = self._parse_binding_with_reason(binding, person_type)
        return record

    def search_person(self, name: str, limit: int = 10) -> list[PersonRecord]:
        """
        Search for a specific person by name.

        Args:
            name: Person name to search for
            limit: Maximum results to return

        Returns:
            List of matching PersonRecords
        """
        # Use Wikidata search API for better name matching
        search_url = "https://www.wikidata.org/w/api.php"
        params = urllib.parse.urlencode({
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": "item",
            "limit": limit,
            "format": "json",
        })

        req = urllib.request.Request(
            f"{search_url}?{params}",
            headers={"User-Agent": "corp-extractor/1.0"}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        for item in data.get("search", []):
            qid = item.get("id")
            label = item.get("label", "")
            description = item.get("description", "")

            # Check if it looks like a person
            person_keywords = [
                "politician", "actor", "actress", "singer", "musician",
                "businessman", "businesswoman", "ceo", "executive", "director",
                "president", "founder", "professor", "scientist", "author",
                "writer", "journalist", "athlete", "player", "coach",
            ]
            description_lower = description.lower()
            is_person = any(kw in description_lower for kw in person_keywords)
            if not is_person:
                continue

            # Try to infer person type from description
            person_type = PersonType.UNKNOWN
            if any(kw in description_lower for kw in ["ceo", "executive", "businessman", "businesswoman"]):
                person_type = PersonType.EXECUTIVE
            elif any(kw in description_lower for kw in ["politician", "president", "senator", "minister"]):
                person_type = PersonType.POLITICIAN
            elif any(kw in description_lower for kw in ["athlete", "player", "coach"]):
                person_type = PersonType.ATHLETE
            elif any(kw in description_lower for kw in ["actor", "actress", "singer", "musician", "director"]):
                person_type = PersonType.ARTIST
            elif any(kw in description_lower for kw in ["professor", "academic"]):
                person_type = PersonType.ACADEMIC
            elif any(kw in description_lower for kw in ["scientist", "researcher"]):
                person_type = PersonType.SCIENTIST
            elif any(kw in description_lower for kw in ["journalist", "reporter"]):
                person_type = PersonType.JOURNALIST
            elif any(kw in description_lower for kw in ["founder", "entrepreneur"]):
                person_type = PersonType.ENTREPRENEUR

            record = PersonRecord(
                name=label,
                source="wikidata",
                source_id=qid,
                country="",  # Not available from search API
                person_type=person_type,
                known_for_role="",
                known_for_org_name="",
                record={
                    "wikidata_id": qid,
                    "label": label,
                    "description": description,
                },
            )
            results.append(record)

        return results

    def get_discovered_organizations(self) -> list[CompanyRecord]:
        """
        Get organizations discovered during the people import.

        These are organizations associated with people (employers, positions, etc.)
        that can be inserted into the organizations database if not already present.

        Returns:
            List of CompanyRecord objects for discovered organizations
        """
        records = []
        for org_qid, org_label in self._discovered_orgs.items():
            record = CompanyRecord(
                name=org_label,
                source="wikipedia",  # Use "wikipedia" as source per wikidata.py convention
                source_id=org_qid,
                region="",  # Not available from this context
                entity_type=EntityType.BUSINESS,  # Default to business for orgs linked to people
                record={
                    "wikidata_id": org_qid,
                    "label": org_label,
                    "discovered_from": "people_import",
                },
            )
            records.append(record)
        logger.info(f"Discovered {len(records)} organizations from people import")
        return records

    def clear_discovered_organizations(self) -> None:
        """Clear the discovered organizations cache."""
        self._discovered_orgs.clear()

    def enrich_person_dates(self, person_qid: str, role: str = "", org: str = "") -> tuple[Optional[str], Optional[str]]:
        """
        Query Wikidata to get start/end dates for a person's position.

        Args:
            person_qid: Wikidata QID of the person (e.g., 'Q123')
            role: Optional role label to match (e.g., 'chief executive officer')
            org: Optional org label to match (e.g., 'Apple Inc')

        Returns:
            Tuple of (from_date, to_date) in ISO format, or (None, None) if not found
        """
        # Query for position dates for this specific person
        # Uses rdfs:label instead of SERVICE wikibase:label for better performance
        query = """
        SELECT ?roleLabel ?orgLabel ?startDate ?endDate WHERE {
          wd:%s p:P39 ?positionStatement .
          ?positionStatement ps:P39 ?role .
          ?role rdfs:label ?roleLabel FILTER(LANG(?roleLabel) = "en") .
          OPTIONAL { ?positionStatement pq:P642 ?org . ?org rdfs:label ?orgLabel FILTER(LANG(?orgLabel) = "en") . }
          OPTIONAL { ?positionStatement pq:P580 ?startDate }
          OPTIONAL { ?positionStatement pq:P582 ?endDate }
        }
        LIMIT 50
        """ % person_qid

        try:
            url = f"{WIKIDATA_SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "corp-extractor/1.0"})

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Find the best matching position
            best_start = None
            best_end = None

            for binding in data.get("results", {}).get("bindings", []):
                role_label = binding.get("roleLabel", {}).get("value", "")
                org_label = binding.get("orgLabel", {}).get("value", "")
                start_raw = binding.get("startDate", {}).get("value", "")
                end_raw = binding.get("endDate", {}).get("value", "")

                # If role/org specified, try to match
                if role and role.lower() not in role_label.lower():
                    continue
                if org and org.lower() not in org_label.lower():
                    continue

                # Parse dates
                start_date = self._parse_wikidata_date(start_raw)
                end_date = self._parse_wikidata_date(end_raw)

                # Prefer entries with dates
                if start_date or end_date:
                    best_start = start_date
                    best_end = end_date
                    break  # Found a match with dates

            return best_start, best_end

        except Exception as e:
            logger.debug(f"Failed to enrich dates for {person_qid}: {e}")
            return None, None

    def enrich_people_batch(
        self,
        people: list[PersonRecord],
        delay_seconds: float = 0.5,
    ) -> int:
        """
        Enrich a batch of people with start/end dates.

        Args:
            people: List of PersonRecord objects to enrich
            delay_seconds: Delay between requests

        Returns:
            Number of people enriched with dates
        """
        enriched_count = 0

        for person in people:
            if person.from_date or person.to_date:
                continue  # Already has dates

            qid = person.source_id
            role = person.known_for_role
            org = person.known_for_org_name

            from_date, to_date = self.enrich_person_dates(qid, role, org)

            if from_date or to_date:
                person.from_date = from_date
                person.to_date = to_date
                enriched_count += 1
                logger.debug(f"Enriched {person.name}: {from_date} - {to_date}")

            time.sleep(delay_seconds)

        logger.info(f"Enriched {enriched_count}/{len(people)} people with dates")
        return enriched_count

    def enrich_person_role_org(
        self, person_qid: str
    ) -> tuple[str, str, str, Optional[str], Optional[str]]:
        """
        Query Wikidata to get role, org, and dates for a person.

        Args:
            person_qid: Wikidata QID of the person (e.g., 'Q123')

        Returns:
            Tuple of (role_label, org_label, org_qid, from_date, to_date)
            Empty strings/None if not found
        """
        # Query for position held (P39) with org qualifier and dates
        # Uses rdfs:label instead of SERVICE wikibase:label for better performance
        query = """
        SELECT ?roleLabel ?org ?orgLabel ?startDate ?endDate WHERE {
          wd:%s p:P39 ?stmt .
          ?stmt ps:P39 ?role .
          ?role rdfs:label ?roleLabel FILTER(LANG(?roleLabel) = "en") .
          OPTIONAL { ?stmt pq:P642 ?org . ?org rdfs:label ?orgLabel FILTER(LANG(?orgLabel) = "en") . }
          OPTIONAL { ?stmt pq:P580 ?startDate . }
          OPTIONAL { ?stmt pq:P582 ?endDate . }
        }
        LIMIT 5
        """ % person_qid

        try:
            url = f"{WIKIDATA_SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "corp-extractor/1.0"})

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            bindings = data.get("results", {}).get("bindings", [])

            # Find the best result (prefer one with org and dates)
            best_result = None
            for binding in bindings:
                role_label = binding.get("roleLabel", {}).get("value", "")
                org_label = binding.get("orgLabel", {}).get("value", "")
                org_uri = binding.get("org", {}).get("value", "")
                start_raw = binding.get("startDate", {}).get("value", "")
                end_raw = binding.get("endDate", {}).get("value", "")

                # Skip if role is just a QID (no label resolved)
                if role_label and role_label.startswith("Q"):
                    continue
                if org_label and org_label.startswith("Q"):
                    org_label = ""

                # Extract QID from URI
                org_qid = ""
                if org_uri:
                    org_qid = org_uri.split("/")[-1]
                    if not org_qid.startswith("Q"):
                        org_qid = ""

                from_date = self._parse_wikidata_date(start_raw)
                to_date = self._parse_wikidata_date(end_raw)

                result = (role_label, org_label, org_qid, from_date, to_date)

                # Prefer results with org and dates
                if org_label and (from_date or to_date):
                    return result
                elif org_label and best_result is None:
                    best_result = result
                elif role_label and best_result is None:
                    best_result = result

            if best_result:
                # If we have a role but no org, try P108 (employer) as fallback
                role_label, org_label, org_qid, from_date, to_date = best_result
                if role_label and not org_label:
                    fallback_org, fallback_org_qid = self._get_employer(person_qid)
                    if fallback_org:
                        return role_label, fallback_org, fallback_org_qid, from_date, to_date
                return best_result

            return "", "", "", None, None

        except Exception as e:
            logger.debug(f"Failed to enrich role/org for {person_qid}: {e}")
            return "", "", "", None, None

    def _get_employer(self, person_qid: str) -> tuple[str, str]:
        """
        Query P108 (employer) as fallback for org.

        Args:
            person_qid: Wikidata QID of the person

        Returns:
            Tuple of (org_label, org_qid) or ("", "") if not found
        """
        query = """
        SELECT ?org ?orgLabel WHERE {
          wd:%s wdt:P108 ?org .
          ?org rdfs:label ?orgLabel FILTER(LANG(?orgLabel) = "en") .
        }
        LIMIT 1
        """ % person_qid

        try:
            url = f"{WIKIDATA_SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "corp-extractor/1.0"})

            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                org_label = bindings[0].get("orgLabel", {}).get("value", "")
                org_uri = bindings[0].get("org", {}).get("value", "")
                org_qid = org_uri.split("/")[-1] if org_uri else ""
                if org_label and not org_label.startswith("Q"):
                    return org_label, org_qid

        except Exception as e:
            logger.debug(f"Failed to get employer for {person_qid}: {e}")

        return "", ""

    def enrich_people_role_org_batch(
        self,
        people: list[PersonRecord],
        delay_seconds: float = 0.1,
        max_workers: int = 5,
    ) -> int:
        """
        Enrich a batch of people with role/org/dates data using parallel queries.

        Args:
            people: List of PersonRecord objects to enrich
            delay_seconds: Delay between requests (per worker)
            max_workers: Number of parallel workers (default 5 for Wikidata rate limits)

        Returns:
            Number of people enriched with role/org
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Filter to people that need enrichment
        to_enrich = [p for p in people if not p.known_for_role and not p.known_for_org_name]

        if not to_enrich:
            logger.info("No people need enrichment")
            return 0

        enriched_count = 0
        total = len(to_enrich)

        def enrich_one(person: PersonRecord) -> tuple[PersonRecord, bool]:
            """Enrich a single person, returns (person, success)."""
            try:
                role, org, org_qid, from_date, to_date = self.enrich_person_role_org(person.source_id)

                if role or org:
                    person.known_for_role = role
                    person.known_for_org_name = org
                    if org_qid:
                        person.record["org_qid"] = org_qid
                    if from_date:
                        person.from_date = from_date
                    if to_date:
                        person.to_date = to_date
                    return person, True

                return person, False
            except Exception as e:
                logger.debug(f"Failed to enrich {person.source_id}: {e}")
                return person, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(enrich_one, person): person for person in to_enrich}

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                person, success = future.result()
                if success:
                    enriched_count += 1
                    logger.debug(f"Enriched {person.name}: {person.known_for_role} at {person.known_for_org_name}")

                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Enriched {completed}/{total} people ({enriched_count} with data)...")

                # Small delay to avoid rate limiting
                time.sleep(delay_seconds)

        logger.info(f"Enriched {enriched_count}/{total} people with role/org/dates")
        return enriched_count
