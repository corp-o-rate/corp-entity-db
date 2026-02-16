"""
Companies House importer for the company database.

Imports UK company data from Companies House API
into the embedding database for company name matching.

Note: The Companies House API requires a free API key for bulk access.
Register at: https://developer.company-information.service.gov.uk/
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType

logger = logging.getLogger(__name__)

# Companies House API endpoints
CH_API_BASE = "https://api.company-information.service.gov.uk"
CH_SEARCH_URL = f"{CH_API_BASE}/search/companies"
CH_COMPANY_URL = f"{CH_API_BASE}/company"

# Bulk data download URL
CH_BULK_DATA_URL = "https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-{date}.zip"
CH_BULK_DATA_PAGE = "https://download.companieshouse.gov.uk/en_output.html"

# Company status prefixes to include (active companies)
ACTIVE_STATUS_PREFIXES = ("active", "open", "live")

# Company type prefixes that are operating companies (matched case-insensitively via startswith)
# Values from bulk CSV data - using prefix matching to handle truncated values
COMPANY_TYPE_PREFIXES = (
    # Limited companies (most common - ~4.8M records)
    "private limited company",
    "public limited company",
    "private unlimited company",
    # Limited by guarantee (values get truncated in CSV due to commas)
    "pri/ltd by guar",
    "pri/lbg/nsc",
    # LLPs (~46K)
    "limited liability partnership",
    # Community interest companies (~38K)
    "community interest company",
    # Charitable incorporated organisations (~45K combined)
    "charitable incorporated organisation",
    "scottish charitable incorporated organisation",
    # Registered societies (~10K)
    "registered society",
    # Overseas entities (~18K)
    "overseas entity",
    # Other
    "other company type",
    "royal charter",
    "old public company",
    # Note: Excluded - "Limited Partnership" (often used for funds)
)

# Mapping from company_type prefixes to EntityType
# Uses prefix matching since CSV values can be truncated
COMPANY_TYPE_TO_ENTITY_TYPE: list[tuple[str, EntityType]] = [
    # Charitable/non-profit (check these first - more specific)
    ("charitable incorporated organisation", EntityType.NONPROFIT),
    ("scottish charitable incorporated organisation", EntityType.NONPROFIT),
    ("community interest company", EntityType.NONPROFIT),
    ("pri/ltd by guar", EntityType.NONPROFIT),  # Limited by guarantee - often charities
    ("pri/lbg/nsc", EntityType.NONPROFIT),  # Limited by guarantee no share capital
    ("registered society", EntityType.NONPROFIT),  # Co-ops, friendly societies

    # Business entities (default for most)
    ("private limited company", EntityType.BUSINESS),
    ("public limited company", EntityType.BUSINESS),
    ("private unlimited company", EntityType.BUSINESS),
    ("limited liability partnership", EntityType.BUSINESS),
    ("overseas entity", EntityType.BUSINESS),
    ("old public company", EntityType.BUSINESS),
    ("royal charter", EntityType.BUSINESS),  # Could be various, default to business
    ("other company type", EntityType.UNKNOWN),
]


def _get_entity_type_from_company_type(company_type: str) -> EntityType:
    """Determine EntityType from Companies House company_type."""
    company_type_lower = company_type.lower().strip()
    for prefix, entity_type in COMPANY_TYPE_TO_ENTITY_TYPE:
        if company_type_lower.startswith(prefix):
            return entity_type
    return EntityType.BUSINESS  # Default to business for unmatched types


class CompaniesHouseImporter:
    """
    Importer for UK Companies House data.

    Uses the Companies House API to fetch company records.
    Requires an API key for bulk access.

    Get a free API key at:
    https://developer.company-information.service.gov.uk/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        active_only: bool = True,
        delay_seconds: float = 0.6,  # API rate limit is ~600/min
    ):
        """
        Initialize the Companies House importer.

        Args:
            api_key: Companies House API key (or set COMPANIES_HOUSE_API_KEY env var)
            active_only: Only import active companies (default True)
            delay_seconds: Delay between requests to respect rate limits
        """
        self._api_key = api_key or os.environ.get("COMPANIES_HOUSE_API_KEY")
        self._active_only = active_only
        self._delay = delay_seconds

        if not self._api_key:
            logger.warning(
                "No Companies House API key provided. "
                "Set COMPANIES_HOUSE_API_KEY env var or pass api_key parameter. "
                "Get a free key at: https://developer.company-information.service.gov.uk/"
            )

    def import_from_search(
        self,
        search_terms: list[str],
        limit_per_term: int = 100,
        total_limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import companies by searching for specific terms.

        This is useful for targeted imports since the API doesn't support
        bulk enumeration without search terms.

        Args:
            search_terms: List of search terms (e.g., ["bank", "insurance", "energy"])
            limit_per_term: Max results per search term
            total_limit: Optional total limit across all terms

        Yields:
            CompanyRecord for each company
        """
        if not self._api_key:
            raise ValueError(
                "Companies House API key required. "
                "Set COMPANIES_HOUSE_API_KEY env var."
            )

        logger.info(f"Starting Companies House import for {len(search_terms)} search terms...")

        total_count = 0
        seen_ids = set()

        for term in search_terms:
            if total_limit and total_count >= total_limit:
                break

            logger.info(f"Searching Companies House for '{term}'...")

            try:
                for record in self._search_companies(term, limit_per_term):
                    if total_limit and total_count >= total_limit:
                        break

                    if record.source_id not in seen_ids:
                        seen_ids.add(record.source_id)
                        total_count += 1
                        yield record

                        if total_count % 100 == 0:
                            logger.info(f"Imported {total_count} Companies House records")

            except Exception as e:
                logger.error(f"Failed to search for '{term}': {e}")
                continue

            # Rate limiting
            time.sleep(self._delay)

        logger.info(f"Completed Companies House import: {total_count} records")

    def import_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import from a local Companies House data file.

        Companies House provides bulk data products (paid) in CSV format.
        This method can parse those files.

        Args:
            file_path: Path to Companies House CSV/JSON file
            limit: Optional limit on records

        Yields:
            CompanyRecord for each company
        """
        import csv

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Companies House file not found: {file_path}")

        logger.info(f"Importing Companies House data from {file_path}")

        count = 0

        if file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if limit and count >= limit:
                        break

                    record = self._parse_csv_row(row)
                    if record:
                        count += 1
                        yield record

                        if count % 10000 == 0:
                            logger.info(f"Imported {count} Companies House records")

        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("items", [])
            for item in items:
                if limit and count >= limit:
                    break

                record = self._parse_api_response(item)
                if record:
                    count += 1
                    yield record

        logger.info(f"Completed Companies House import: {count} records")

    def _search_companies(
        self,
        query: str,
        limit: int = 100,
    ) -> Iterator[CompanyRecord]:
        """Search for companies via the API."""
        items_per_page = min(100, limit)  # API max is 100
        start_index = 0
        fetched = 0

        while fetched < limit:
            params = urllib.parse.urlencode({
                "q": query,
                "items_per_page": items_per_page,
                "start_index": start_index,
            })

            url = f"{CH_SEARCH_URL}?{params}"
            data = self._api_request(url)

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                if fetched >= limit:
                    break

                record = self._parse_api_response(item)
                if record:
                    fetched += 1
                    yield record

            # Check if more results available
            total_results = data.get("total_results", 0)
            start_index += items_per_page

            if start_index >= total_results or start_index >= 400:  # API limit
                break

            time.sleep(self._delay)

    def _api_request(self, url: str) -> dict[str, Any]:
        """Make an authenticated API request."""
        # Companies House uses HTTP Basic Auth with API key as username
        auth_string = base64.b64encode(f"{self._api_key}:".encode()).decode()

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Basic {auth_string}",
                "Accept": "application/json",
            }
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_api_response(self, item: dict[str, Any]) -> Optional[CompanyRecord]:
        """Parse an API response item into a CompanyRecord."""
        try:
            company_number = item.get("company_number")
            title = item.get("title") or item.get("company_name", "")
            company_status = item.get("company_status", "").lower()
            company_type = item.get("company_type", "").lower()

            if not company_number or not title:
                return None

            # Filter by status if configured
            if self._active_only and not company_status.startswith(ACTIVE_STATUS_PREFIXES):
                return None

            # Filter to only include actual companies (not sole traders, individuals, etc.)
            if company_type and not company_type.startswith(COMPANY_TYPE_PREFIXES):
                return None

            # Get address info
            address = item.get("registered_office_address") or item.get("address", {})
            if isinstance(address, dict):
                locality = address.get("locality", "")
                region = address.get("region", "")
                country = address.get("country", "United Kingdom")
            else:
                locality = ""
                region = ""
                country = "United Kingdom"

            # Determine entity type from company_type
            raw_company_type = item.get("company_type", "")
            entity_type = _get_entity_type_from_company_type(raw_company_type)

            # Get dates
            date_of_creation = item.get("date_of_creation")
            date_of_cessation = item.get("date_of_cessation")  # For dissolved companies

            # Build record
            record_data = {
                "company_number": company_number,
                "title": title,
                "company_status": company_status,
                "company_type": raw_company_type,
                "date_of_creation": date_of_creation,
                "date_of_cessation": date_of_cessation,
                "locality": locality,
                "region": region,
                "country": country,
            }

            return CompanyRecord(
                name=title.strip(),
                source="companies_house",
                source_id=company_number,
                region=country,
                entity_type=entity_type,
                from_date=date_of_creation,
                to_date=date_of_cessation,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse Companies House item: {e}")
            return None

    def _parse_csv_row(self, row: dict[str, Any]) -> Optional[CompanyRecord]:
        """Parse a CSV row from bulk data file."""
        try:
            # Normalize column names (strip whitespace from keys)
            row = {k.strip(): v for k, v in row.items()}

            # Companies House CSV column names
            company_number = row.get("CompanyNumber") or row.get("company_number", "")
            company_name = row.get("CompanyName") or row.get("company_name", "")
            company_status = (row.get("CompanyStatus") or row.get("company_status", "")).lower().strip()
            company_type = (row.get("CompanyCategory") or row.get("company_type", "")).lower().strip()

            if not company_number or not company_name:
                return None

            # Strip whitespace from values too
            company_number = company_number.strip()
            company_name = company_name.strip()

            if self._active_only and not company_status.startswith(ACTIVE_STATUS_PREFIXES):
                return None

            # Filter to only include actual companies (not sole traders, individuals, etc.)
            if company_type and not company_type.startswith(COMPANY_TYPE_PREFIXES):
                return None

            # Determine entity type from company_type
            raw_company_type = row.get("CompanyCategory", "").strip()
            entity_type = _get_entity_type_from_company_type(raw_company_type)

            # Get dates from CSV
            date_of_creation = row.get("IncorporationDate", "").strip() or None
            date_of_cessation = row.get("DissolutionDate", "").strip() or None

            record_data = {
                "company_number": company_number,
                "title": company_name,
                "company_status": company_status,
                "company_type": raw_company_type,
                "date_of_creation": date_of_creation,
                "date_of_cessation": date_of_cessation,
                "country": row.get("CountryOfOrigin", "United Kingdom").strip(),
                "sic_code": row.get("SICCode.SicText_1", "").strip(),
            }

            # Use CountryOfOrigin for region
            region = row.get("CountryOfOrigin", "United Kingdom").strip()

            return CompanyRecord(
                name=company_name,
                source="companies_house",
                source_id=company_number,
                region=region,
                entity_type=entity_type,
                from_date=date_of_creation,
                to_date=date_of_cessation,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse CSV row: {e}")
            return None

    def get_company(self, company_number: str) -> Optional[CompanyRecord]:
        """
        Fetch a specific company by number.

        Args:
            company_number: UK company registration number

        Returns:
            CompanyRecord or None if not found
        """
        if not self._api_key:
            raise ValueError("Companies House API key required")

        try:
            url = f"{CH_COMPANY_URL}/{company_number}"
            data = self._api_request(url)
            return self._parse_api_response(data)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def download_bulk_data(
        self,
        output_path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """
        Download the bulk company data file from Companies House.

        This is a free download containing all active UK companies.
        No API key required.

        Args:
            output_path: Where to save the CSV file (default: temp directory)
            force: Force re-download even if cached

        Returns:
            Path to the extracted CSV file
        """
        import re
        import shutil
        import tempfile
        import zipfile

        # Find the latest file date from the download page
        logger.info("Checking for latest Companies House bulk data...")

        req = urllib.request.Request(
            CH_BULK_DATA_PAGE,
            headers={"User-Agent": "corp-extractor/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")

        # Find the filename pattern: BasicCompanyDataAsOneFile-YYYY-MM-DD.zip
        match = re.search(r'BasicCompanyDataAsOneFile-(\d{4}-\d{2}-\d{2})\.zip', html)
        if not match:
            raise RuntimeError("Could not find bulk data file on Companies House page")

        file_date = match.group(1)
        download_url = CH_BULK_DATA_URL.format(date=file_date)

        # Set up output directory
        if output_path is None:
            output_dir = Path(tempfile.gettempdir()) / "companies_house"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "BasicCompanyData.csv"
        else:
            output_dir = output_path.parent

        # Check for cached version
        metadata_path = output_dir / "ch_metadata.json"
        if not force and output_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    cached_metadata = json.load(f)
                if cached_metadata.get("file_date") == file_date:
                    logger.info(f"Using cached Companies House data (date: {file_date})")
                    return output_path
            except (json.JSONDecodeError, IOError):
                pass

        # Download the ZIP file
        logger.info(f"Downloading Companies House bulk data ({file_date})...")
        zip_path = output_path.with_suffix(".zip")

        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "corp-extractor/1.0"}
        )
        with urllib.request.urlopen(req) as response:
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(response, f)

        # Extract CSV from ZIP
        logger.info("Extracting CSV file...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                raise RuntimeError("No CSV file found in ZIP archive")

            # Extract the first (usually only) CSV file
            with zf.open(csv_files[0]) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # Clean up ZIP
        zip_path.unlink()

        # Save metadata
        metadata = {
            "file_date": file_date,
            "downloaded_at": str(output_path.stat().st_mtime),
            "output_path": str(output_path),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        logger.info(f"Downloaded Companies House data to {output_path}")
        return output_path
