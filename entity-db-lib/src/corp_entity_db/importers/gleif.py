"""
GLEIF data importer for the company database.

Imports Legal Entity Identifier (LEI) data from GLEIF files
into the embedding database for company name matching.

Supports:
- JSON files (API responses, concatenated JSON)
- XML files (official GLEIF LEI-CDF v3.1 format)
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType

logger = logging.getLogger(__name__)

# XML namespaces for GLEIF LEI-CDF format
LEI_NAMESPACES = {
    'lei': 'http://www.gleif.org/data/schema/leidata/2016',
}

# Mapping from GLEIF EntityCategory to our EntityType
# See: https://www.gleif.org/en/about-lei/common-data-file-format
GLEIF_CATEGORY_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "GENERAL": EntityType.BUSINESS,  # Regular legal entities (companies)
    "FUND": EntityType.FUND,  # Investment funds, ETFs, mutual funds
    "BRANCH": EntityType.BRANCH,  # Branch offices of companies
    "SOLE_PROPRIETOR": EntityType.BUSINESS,  # Sole proprietorships (still a business)
    "INTERNATIONAL_ORGANIZATION": EntityType.INTERNATIONAL_ORG,  # UN, WHO, IMF, etc.
    "": EntityType.UNKNOWN,  # Empty/unset
}


class GleifImporter:
    """
    Importer for GLEIF LEI data.

    Supports:
    - JSON concatenated files (level1-concatenated.json)
    - Individual JSON records
    - Streaming import for large files

    Maps GLEIF EntityCategory to EntityType:
    - GENERAL -> BUSINESS
    - FUND -> FUND
    - BRANCH -> BRANCH
    - SOLE_PROPRIETOR -> BUSINESS
    - INTERNATIONAL_ORGANIZATION -> INTERNATIONAL_ORG
    """

    def __init__(self, active_only: bool = True):
        """
        Initialize the GLEIF importer.

        Args:
            active_only: Only import ACTIVE entities (default True)
        """
        self._active_only = active_only

    def import_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import records from a GLEIF file.

        Supports:
        - XML files (official GLEIF LEI-CDF v3.1 format)
        - JSON array files
        - Concatenated JSON files (one object per line)

        Args:
            file_path: Path to GLEIF file (XML or JSON)
            limit: Optional limit on number of records

        Yields:
            CompanyRecord for each valid entity
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"GLEIF file not found: {file_path}")

        logger.info(f"Importing GLEIF data from {file_path}")

        # Detect file format by extension or content
        if file_path.suffix.lower() == ".xml":
            yield from self._import_xml_streaming(file_path, limit)
        else:
            # Try to detect JSON format
            with open(file_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)

            if first_char == "<":
                # XML content
                yield from self._import_xml_streaming(file_path, limit)
            elif first_char == "[":
                # JSON array format
                yield from self._import_json_array(file_path, limit)
            else:
                # Concatenated JSON format (one object per line)
                yield from self._import_concatenated_json(file_path, limit)

    def _import_json_array(
        self,
        file_path: Path,
        limit: Optional[int],
    ) -> Iterator[CompanyRecord]:
        """Import from JSON array format."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else data.get("data", [])
        count = 0

        for raw_record in records:
            if limit and count >= limit:
                break

            record = self._parse_record(raw_record)
            if record:
                count += 1
                yield record

                if count % 10000 == 0:
                    logger.info(f"Imported {count} GLEIF records")

        logger.info(f"Completed GLEIF import: {count} records")

    def _import_concatenated_json(
        self,
        file_path: Path,
        limit: Optional[int],
    ) -> Iterator[CompanyRecord]:
        """Import from concatenated JSON format (one object per line)."""
        count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    raw_record = json.loads(line)
                    record = self._parse_record(raw_record)
                    if record:
                        count += 1
                        yield record

                        if count % 10000 == 0:
                            logger.info(f"Imported {count} GLEIF records")

                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {e}")
                    continue

        logger.info(f"Completed GLEIF import: {count} records")

    def _import_xml_streaming(
        self,
        file_path: Path,
        limit: Optional[int],
    ) -> Iterator[CompanyRecord]:
        """
        Import from XML file using streaming parser.

        Uses iterparse for memory-efficient parsing of large XML files.
        """
        logger.info(f"Starting streaming XML parse of {file_path}")
        count = 0

        try:
            context = ET.iterparse(str(file_path), events=('end',))

            for event, elem in context:
                # Look for LEIRecord elements (handle both namespaced and non-namespaced)
                if elem.tag.endswith('LEIRecord'):
                    if limit and count >= limit:
                        break

                    record = self._parse_xml_record(elem)
                    if record:
                        count += 1
                        yield record

                        if count % 10000 == 0:
                            logger.info(f"Parsed {count} XML records")

                    # Clear element to free memory
                    elem.clear()

            logger.info(f"Completed XML import: {count} records")

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise ValueError(f"Failed to parse XML file {file_path}: {e}")

    def _parse_xml_record(self, lei_record: ET.Element) -> Optional[CompanyRecord]:
        """Parse a single LEI record from XML."""
        try:
            # Helper to find elements with or without namespace
            def find_text(parent: ET.Element, tag: str) -> Optional[str]:
                # Try with namespace first
                elem = parent.find(f'.//lei:{tag}', LEI_NAMESPACES)
                if elem is None:
                    # Try without namespace
                    elem = parent.find(f'.//{tag}')
                return elem.text if elem is not None else None

            def find_elem(parent: ET.Element, tag: str) -> Optional[ET.Element]:
                elem = parent.find(f'.//lei:{tag}', LEI_NAMESPACES)
                if elem is None:
                    elem = parent.find(f'.//{tag}')
                return elem

            # Get LEI
            lei = find_text(lei_record, 'LEI')
            if not lei or len(lei) != 20:
                return None

            # Get Entity element
            entity_elem = find_elem(lei_record, 'Entity')
            if entity_elem is None:
                return None

            # Get legal name
            legal_name = find_text(entity_elem, 'LegalName')
            if not legal_name:
                return None

            # Get status - skip inactive if configured
            status = find_text(lei_record, 'EntityStatus')
            if self._active_only and status and status.upper() != 'ACTIVE':
                return None

            # Get entity category and map to EntityType
            entity_category = find_text(entity_elem, 'EntityCategory') or ""
            entity_type = GLEIF_CATEGORY_TO_ENTITY_TYPE.get(
                entity_category.upper(),
                GLEIF_CATEGORY_TO_ENTITY_TYPE.get(entity_category, EntityType.UNKNOWN)
            )

            # Get jurisdiction
            jurisdiction = find_text(entity_elem, 'LegalJurisdiction')

            # Get address info
            legal_address = find_elem(entity_elem, 'LegalAddress')
            country = ""
            city = ""
            if legal_address is not None:
                country = find_text(legal_address, 'Country') or ""
                city = find_text(legal_address, 'City') or ""

            # Get other names
            other_names = []
            other_names_elem = find_elem(entity_elem, 'OtherEntityNames')
            if other_names_elem is not None:
                for name_elem in other_names_elem:
                    if name_elem.text:
                        other_names.append(name_elem.text)

            # Get registration dates from Registration element
            registration_elem = find_elem(lei_record, 'Registration')
            initial_reg_date = None
            if registration_elem is not None:
                initial_reg_date = find_text(registration_elem, 'InitialRegistrationDate')
                # Extract just the date part (YYYY-MM-DD) from ISO datetime
                if initial_reg_date and "T" in initial_reg_date:
                    initial_reg_date = initial_reg_date.split("T")[0]

            # Build record
            name = legal_name.strip()
            record_data = {
                "lei": lei,
                "legal_name": legal_name,
                "status": status,
                "jurisdiction": jurisdiction,
                "country": country,
                "city": city,
                "entity_category": entity_category,
                "other_names": other_names,
            }
            if initial_reg_date:
                record_data["initial_registration_date"] = initial_reg_date

            return CompanyRecord(
                name=name,
                source="gleif",
                source_id=lei,
                region=country,
                entity_type=entity_type,
                from_date=initial_reg_date,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse XML record: {e}")
            return None

    def _parse_record(self, raw: dict[str, Any]) -> Optional[CompanyRecord]:
        """
        Parse a raw GLEIF record into a CompanyRecord.

        Handles both API response format and bulk file format.
        """
        try:
            # Handle nested structure from API or bulk files
            attrs = raw.get("attributes", raw)
            entity = attrs.get("entity", attrs)

            # Get status - skip inactive if configured
            registration = attrs.get("registration", {})
            status = registration.get("status") or entity.get("status") or raw.get("status")
            if self._active_only and status and status.upper() != "ACTIVE":
                return None

            # Get entity category and map to EntityType
            entity_category = entity.get("category") or entity.get("EntityCategory") or ""
            entity_type = GLEIF_CATEGORY_TO_ENTITY_TYPE.get(
                entity_category.upper(),
                GLEIF_CATEGORY_TO_ENTITY_TYPE.get(entity_category, EntityType.UNKNOWN)
            )

            # Get LEI
            lei = raw.get("id") or attrs.get("lei") or raw.get("LEI")
            if not lei:
                return None

            # Get legal name - handle GLEIF JSON format with nested "$" key
            legal_name_obj = entity.get("legalName", {})
            if isinstance(legal_name_obj, dict):
                legal_name = legal_name_obj.get("name") or legal_name_obj.get("$", "")
            else:
                legal_name = legal_name_obj or ""

            if not legal_name:
                # Try alternative locations
                legal_name = entity.get("LegalName") or raw.get("legal_name") or ""

            if not legal_name:
                return None

            # Get other names for better matching
            other_names = []
            other_names_list = entity.get("otherNames", []) or entity.get("OtherEntityNames", [])
            for other in other_names_list:
                if isinstance(other, dict):
                    name = other.get("name") or other.get("$", "")
                else:
                    name = str(other)
                if name:
                    other_names.append(name)

            # Use legal name as primary, but store others in record
            name = legal_name.strip()

            # Get jurisdiction and address info
            jurisdiction = entity.get("jurisdiction") or entity.get("LegalJurisdiction")
            legal_address = entity.get("legalAddress", {})
            if isinstance(legal_address, dict):
                country = legal_address.get("country") or legal_address.get("Country", "")
                city = legal_address.get("city") or legal_address.get("City", "")
            else:
                country = ""
                city = ""

            # Get registration dates
            initial_reg_date = registration.get("initialRegistrationDate")
            if not initial_reg_date:
                initial_reg_date = registration.get("InitialRegistrationDate")
            # Extract just the date part (YYYY-MM-DD) from ISO datetime
            if initial_reg_date and "T" in initial_reg_date:
                initial_reg_date = initial_reg_date.split("T")[0]

            # Build record with relevant data
            record_data = {
                "lei": lei,
                "legal_name": legal_name,
                "status": status,
                "jurisdiction": jurisdiction,
                "country": country,
                "city": city,
                "entity_category": entity_category,
                "other_names": other_names,
            }
            if initial_reg_date:
                record_data["initial_registration_date"] = initial_reg_date

            return CompanyRecord(
                name=name,
                source="gleif",
                source_id=lei,
                region=country,
                entity_type=entity_type,
                from_date=initial_reg_date,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse GLEIF record: {e}")
            return None

    def get_latest_file_info(self) -> dict[str, Any]:
        """
        Get information about the latest GLEIF LEI file.

        Returns:
            Dict with file metadata including 'id', 'publish_date', 'record_count'
        """
        import urllib.request

        # GLEIF API to list available concatenated files
        api_url = "https://leidata.gleif.org/api/v1/concatenated-files/lei2"

        logger.info("Checking for latest GLEIF data file...")

        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/json"}
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

        # The API returns files sorted by date, most recent first
        files = data.get("data", [])
        if not files:
            raise RuntimeError("No GLEIF files available from API")

        latest = files[0]
        file_id = latest.get("id")
        # Fields are at top level, not nested under "attributes"
        record_count = latest.get("record_count")
        content_date = latest.get("content_date")

        info = {
            "id": file_id,
            "publish_date": content_date,
            "record_count": record_count,
            "cdf_version": latest.get("cdf_version"),
        }

        record_str = f"{record_count:,}" if record_count else "unknown"
        logger.info(
            f"Latest GLEIF file: ID={file_id}, "
            f"date={content_date}, "
            f"records={record_str}"
        )

        return info

    def download_latest(
        self,
        output_path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """
        Download the latest GLEIF data file.

        Automatically fetches the most recent file from GLEIF's API.
        Caches downloads and skips re-downloading if the same file ID exists.

        Args:
            output_path: Where to save the file (default: temp directory)
            force: Force re-download even if cached

        Returns:
            Path to downloaded file
        """
        import shutil
        import tempfile
        import urllib.request
        import zipfile

        # Get latest file info from API
        file_info = self.get_latest_file_info()
        file_id = file_info["id"]

        # Set up output directory and paths
        if output_path is None:
            output_dir = Path(tempfile.gettempdir()) / "gleif"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "lei-records.xml"
        else:
            output_dir = output_path.parent

        # Check for cached version using metadata file
        metadata_path = output_dir / "gleif_metadata.json"
        if not force and output_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    cached_metadata = json.load(f)
                if cached_metadata.get("file_id") == file_id:
                    logger.info(
                        f"Using cached GLEIF data (file ID: {file_id}, "
                        f"date: {cached_metadata.get('publish_date')})"
                    )
                    return output_path
            except (json.JSONDecodeError, IOError):
                pass  # Metadata corrupted, re-download

        # Build download URL for the latest file
        url = f"https://leidata.gleif.org/api/v1/concatenated-files/lei2/get/{file_id}/zip"

        logger.info(f"Downloading GLEIF data (file ID: {file_id}) from {url}")

        # Download ZIP file
        zip_path = output_path.with_suffix(".zip")
        urllib.request.urlretrieve(url, zip_path)

        # Extract data file from ZIP (XML or JSON)
        with zipfile.ZipFile(zip_path, "r") as zf:
            extracted = False
            for name in zf.namelist():
                # Prefer XML (official format), fall back to JSON
                if name.endswith(".xml") or name.endswith(".json"):
                    logger.info(f"Extracting {name}...")
                    # Update output path extension to match extracted file
                    if name.endswith(".xml"):
                        output_path = output_path.with_suffix(".xml")
                    else:
                        output_path = output_path.with_suffix(".json")
                    with zf.open(name) as src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted = True
                    break

            if not extracted:
                raise RuntimeError(f"No XML or JSON file found in GLEIF ZIP archive")

        # Clean up ZIP
        zip_path.unlink()

        # Save metadata for caching
        metadata = {
            "file_id": file_id,
            "publish_date": file_info.get("publish_date"),
            "record_count": file_info.get("record_count"),
            "downloaded_at": str(Path(output_path).stat().st_mtime),
            "output_path": str(output_path),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        record_count = file_info.get('record_count')
        record_str = f"{record_count:,}" if record_count else "unknown"
        logger.info(
            f"Downloaded GLEIF data to {output_path} "
            f"(published: {file_info['publish_date']}, records: {record_str})"
        )
        return output_path
