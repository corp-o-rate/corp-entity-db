"""
SEC Edgar data importer for the company database.

Imports company data from SEC's bulk submissions.zip file
into the embedding database for company name matching.

The submissions.zip contains JSON files for ALL SEC filers (~100K+),
not just companies with ticker symbols (~10K).
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType

logger = logging.getLogger(__name__)

# SEC Edgar bulk data URLs
SEC_SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# User agent for SEC requests (required)
SEC_USER_AGENT = "corp-extractor/1.0 (contact@corp-o-rate.com)"

# Entity types that are operating companies (exclude individuals and funds)
ORG_ENTITY_TYPES = {
    "operating",  # Operating companies
    "foreign-private-issuer",  # Foreign companies
}

# SIC codes for funds/financial instruments (map to FUND entity type)
FUND_SIC_CODES = {
    "6722",  # Management Investment Offices, Open-End
    "6726",  # Other Investment Offices
    "6732",  # Educational, Religious, and Charitable Trusts
    "6733",  # Trusts, Except Educational, Religious, and Charitable
    "6792",  # Oil Royalty Traders
    "6794",  # Patent Owners and Lessors
    "6795",  # Mineral Royalty Traders
    "6798",  # Real Estate Investment Trusts
    "6799",  # Investors, NEC
}

# Mapping from SEC entity types to our EntityType
SEC_ENTITY_TYPE_MAP: dict[str, EntityType] = {
    "operating": EntityType.BUSINESS,
    "foreign-private-issuer": EntityType.BUSINESS,
    "": EntityType.UNKNOWN,
}


def _get_entity_type_from_sec(sec_entity_type: str, sic: str) -> EntityType:
    """Determine EntityType from SEC entity type and SIC code."""
    # Check SIC codes first - they're more specific
    if sic in FUND_SIC_CODES:
        return EntityType.FUND

    # Map SEC entity type
    return SEC_ENTITY_TYPE_MAP.get(sec_entity_type.lower(), EntityType.BUSINESS)


class SecEdgarImporter:
    """
    Importer for SEC Edgar company data.

    Uses the bulk submissions.zip file which contains all SEC filers,
    not just companies with ticker symbols.
    """

    def __init__(self):
        """Initialize the SEC Edgar importer."""
        self._ticker_lookup: Optional[dict[str, str]] = None

    def import_from_url(
        self,
        limit: Optional[int] = None,
        download_dir: Optional[Path] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import records by downloading SEC bulk submissions.zip.

        Args:
            limit: Optional limit on number of records
            download_dir: Directory to download zip file to

        Yields:
            CompanyRecord for each company
        """
        # Download submissions.zip
        zip_path = self.download_submissions_zip(download_dir)
        yield from self.import_from_zip(zip_path, limit)

    def import_from_zip(
        self,
        zip_path: str | Path,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import records from a local submissions.zip file.

        Args:
            zip_path: Path to submissions.zip
            limit: Optional limit on number of records

        Yields:
            CompanyRecord for each company
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"SEC submissions.zip not found: {zip_path}")

        logger.info(f"Importing SEC Edgar data from {zip_path}")

        # Load ticker lookup for enrichment
        self._load_ticker_lookup()

        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get list of JSON files (CIK*.json)
            json_files = [n for n in zf.namelist() if n.startswith("CIK") and n.endswith(".json")]
            logger.info(f"Found {len(json_files)} submission files in archive")

            for filename in json_files:
                if limit and count >= limit:
                    break

                try:
                    with zf.open(filename) as f:
                        data = json.load(f)
                        record = self._parse_submission(data)
                        if record:
                            count += 1
                            yield record

                            if count % 10000 == 0:
                                logger.info(f"Imported {count} SEC Edgar records")
                except Exception as e:
                    logger.debug(f"Failed to parse {filename}: {e}")

        logger.info(f"Completed SEC Edgar import: {count} records")

    def import_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Import records from a local file (zip or legacy tickers JSON).

        Args:
            file_path: Path to submissions.zip or company_tickers.json
            limit: Optional limit on number of records

        Yields:
            CompanyRecord for each company
        """
        file_path = Path(file_path)

        if file_path.suffix == ".zip":
            yield from self.import_from_zip(file_path, limit)
        elif file_path.suffix == ".json":
            # Legacy support for company_tickers.json
            yield from self._import_from_tickers_json(file_path, limit)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _import_from_tickers_json(
        self,
        file_path: Path,
        limit: Optional[int],
    ) -> Iterator[CompanyRecord]:
        """Legacy import from company_tickers.json."""
        logger.info(f"Importing from legacy tickers file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for entry in data.values():
            if limit and count >= limit:
                break

            cik = entry.get("cik_str")
            ticker = entry.get("ticker", "")
            title = entry.get("title", "")

            if not cik or not title:
                continue

            cik_str = str(cik).zfill(10)
            record_data = {"cik": cik_str, "ticker": ticker, "title": title}

            yield CompanyRecord(
                name=title.strip(),
                source="sec_edgar",
                source_id=cik_str,
                region="US",  # Tickers file is US-only
                entity_type=EntityType.BUSINESS,  # Ticker file = publicly traded businesses
                record=record_data,
            )
            count += 1

        logger.info(f"Completed legacy SEC import: {count} records")

    def _load_ticker_lookup(self) -> None:
        """Load ticker symbols for CIK enrichment."""
        if self._ticker_lookup is not None:
            return

        self._ticker_lookup = {}
        try:
            import urllib.request

            req = urllib.request.Request(
                SEC_TICKERS_URL,
                headers={"User-Agent": SEC_USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            for entry in data.values():
                cik = str(entry.get("cik_str", "")).zfill(10)
                ticker = entry.get("ticker", "")
                if cik and ticker:
                    self._ticker_lookup[cik] = ticker

            logger.info(f"Loaded {len(self._ticker_lookup)} ticker symbols")
        except Exception as e:
            logger.warning(f"Failed to load ticker lookup: {e}")

    def _parse_submission(self, data: dict[str, Any]) -> Optional[CompanyRecord]:
        """Parse a submission JSON file into a CompanyRecord."""
        try:
            cik = str(data.get("cik", "")).zfill(10)
            name = data.get("name", "").strip()
            entity_type = data.get("entityType", "").lower()

            if not cik or not name:
                return None

            # Filter to only include organizations (exclude individuals without business indicators)
            ticker = self._ticker_lookup.get(cik, "") if self._ticker_lookup else ""
            sic = data.get("sic", "")

            # Include if: has a known org entity type, OR has a ticker (publicly traded), OR has SIC code
            is_organization = (
                entity_type in ORG_ENTITY_TYPES
                or ticker  # Has a ticker symbol = publicly traded company
                or sic  # Has SIC code = classified business
            )

            if not is_organization:
                return None

            # Determine entity type (business, fund, etc.)
            record_entity_type = _get_entity_type_from_sec(entity_type, sic)

            # Get additional fields
            sic_description = data.get("sicDescription", "")
            state = data.get("stateOfIncorporation", "")
            fiscal_year_end = data.get("fiscalYearEnd", "")

            # Get addresses
            addresses = data.get("addresses", {})
            business_addr = addresses.get("business", {})

            # Get ticker from lookup
            ticker = self._ticker_lookup.get(cik, "") if self._ticker_lookup else ""

            # Get exchange info from filings if available
            exchanges = data.get("exchanges", [])
            exchange = exchanges[0] if exchanges else ""

            # Get dates from filings history
            # Use oldest filing date as from_date (when company started filing with SEC)
            filings = data.get("filings", {})
            recent_filings = filings.get("recent", {})
            filing_dates = recent_filings.get("filingDate", [])

            # Get the oldest filing date (last in the list, as they're typically newest-first)
            from_date = None
            if filing_dates:
                # Filing dates are in YYYY-MM-DD format
                oldest_date = filing_dates[-1] if filing_dates else None
                if oldest_date and len(oldest_date) >= 10:
                    from_date = oldest_date[:10]

            # Build record
            record_data = {
                "cik": cik,
                "name": name,
                "sic": sic,
                "sic_description": sic_description,
                "entity_type": entity_type,
                "state_of_incorporation": state,
                "fiscal_year_end": fiscal_year_end,
                "ticker": ticker,
                "exchange": exchange,
                "business_address": {
                    "street": business_addr.get("street1", ""),
                    "city": business_addr.get("city", ""),
                    "state": business_addr.get("stateOrCountry", ""),
                    "zip": business_addr.get("zipCode", ""),
                },
            }
            if from_date:
                record_data["first_filing_date"] = from_date

            # Use stateOrCountry for region (2-letter US state or country code)
            region = business_addr.get("stateOrCountry", "US")

            return CompanyRecord(
                name=name,
                source="sec_edgar",
                source_id=cik,
                region=region,
                entity_type=record_entity_type,
                from_date=from_date,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse submission: {e}")
            return None

    def download_submissions_zip(self, output_dir: Optional[Path] = None) -> Path:
        """
        Download the SEC bulk submissions.zip file.

        Args:
            output_dir: Directory to save the file

        Returns:
            Path to downloaded file
        """
        import tempfile
        import urllib.request

        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "sec_edgar"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "submissions.zip"

        logger.info(f"Downloading SEC submissions.zip (~500MB)...")
        logger.info(f"URL: {SEC_SUBMISSIONS_URL}")

        req = urllib.request.Request(
            SEC_SUBMISSIONS_URL,
            headers={"User-Agent": SEC_USER_AGENT},
        )

        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({pct:.1f}%)")

        logger.info(f"Downloaded SEC submissions.zip to {output_path}")
        return output_path

    def download_latest(self, output_path: Optional[Path] = None) -> Path:
        """
        Download the latest SEC bulk data.

        Args:
            output_path: Where to save the file (directory or file path)

        Returns:
            Path to downloaded file
        """
        if output_path is None:
            return self.download_submissions_zip()

        output_path = Path(output_path)
        if output_path.is_dir():
            return self.download_submissions_zip(output_path)
        else:
            return self.download_submissions_zip(output_path.parent)
