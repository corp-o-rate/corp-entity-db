"""
SEC Form 4 importer for the people database.

Imports insider ownership data from SEC Form 4 filings to identify
officers and directors of public companies.

Form 4 Structure (XML):
- issuer: CIK, name, ticker of the company
- reportingOwner: CIK, name, relationship (isDirector, isOfficer, officerTitle)
- transactions: Stock transactions (not used for people import)

Data Source:
- Quarterly index files at: /Archives/edgar/full-index/{year}/QTR{q}/form.idx
- Individual filings at: /Archives/edgar/data/{cik}/{accession}.txt

Resume Support:
- Progress tracked by (year, quarter, filing_index)
- Progress saved to JSON file for resume on interruption
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from ..models import PersonRecord, PersonType

logger = logging.getLogger(__name__)

# SEC Edgar URLs
SEC_BASE_URL = "https://www.sec.gov"
SEC_FULL_INDEX_URL = f"{SEC_BASE_URL}/Archives/edgar/full-index"

# User agent for SEC requests (required)
SEC_USER_AGENT = "corp-extractor/1.0 (contact@corp-o-rate.com)"

# Rate limiting: SEC allows 10 requests/second, we use 5 to be safe
SEC_REQUEST_DELAY = 0.2  # 200ms between requests

# Default progress file path
DEFAULT_PROGRESS_PATH = Path.home() / ".cache" / "corp-extractor" / "sec-form4-progress.json"


def _normalize_name(name: str) -> str:
    """Normalize a person name for consistent storage."""
    # Remove extra whitespace
    name = " ".join(name.split())
    # Title case
    name = name.title()
    return name


def _map_to_person_type(
    is_director: bool, is_officer: bool, is_ten_percent_owner: bool, officer_title: str
) -> PersonType:
    """Map Form 4 relationship to PersonType."""
    if is_officer:
        return PersonType.EXECUTIVE
    if is_director:
        return PersonType.EXECUTIVE  # Directors are also executives
    if is_ten_percent_owner:
        return PersonType.ENTREPRENEUR  # Significant investors
    return PersonType.UNKNOWN


def _extract_officer_role(
    is_director: bool, is_officer: bool, is_ten_percent_owner: bool, officer_title: str
) -> str:
    """Extract the role description from Form 4 data."""
    roles = []
    if is_director:
        roles.append("Director")
    if is_officer and officer_title:
        roles.append(officer_title)
    elif is_officer:
        roles.append("Officer")
    if is_ten_percent_owner and not is_director and not is_officer:
        roles.append("Investor")
    return ", ".join(roles) if roles else "Insider"


@dataclass
class Form4Progress:
    """
    Tracks progress through SEC Form 4 import for resume support.

    Progress is tracked by:
    - year: Current year being processed
    - quarter: Current quarter (1-4)
    - filing_index: Index within current quarter's Form 4 filings
    - total_imported: Total records imported so far
    """
    year: int = 0
    quarter: int = 0
    filing_index: int = 0
    total_imported: int = 0
    last_accession: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: Path = DEFAULT_PROGRESS_PATH) -> None:
        """Save progress to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump({
                "year": self.year,
                "quarter": self.quarter,
                "filing_index": self.filing_index,
                "total_imported": self.total_imported,
                "last_accession": self.last_accession,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
            }, f, indent=2)
        logger.debug(f"Saved progress: year={self.year}, Q{self.quarter}, index={self.filing_index}")

    @classmethod
    def load(cls, path: Path = DEFAULT_PROGRESS_PATH) -> Optional["Form4Progress"]:
        """Load progress from JSON file, returns None if not found."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                year=data.get("year", 0),
                quarter=data.get("quarter", 0),
                filing_index=data.get("filing_index", 0),
                total_imported=data.get("total_imported", 0),
                last_accession=data.get("last_accession", ""),
                started_at=data.get("started_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
            )
        except Exception as e:
            logger.warning(f"Failed to load progress from {path}: {e}")
            return None

    @staticmethod
    def clear(path: Path = DEFAULT_PROGRESS_PATH) -> None:
        """Delete the progress file."""
        if path.exists():
            path.unlink()
            logger.info(f"Cleared progress file: {path}")


@dataclass
class Form4Filing:
    """Represents a Form 4 filing from the index."""
    form_type: str
    company_name: str
    cik: str
    date_filed: str
    file_path: str

    @property
    def accession_number(self) -> str:
        """Extract accession number from file path."""
        # Path like: edgar/data/1084869/0001437749-25-030850.txt
        match = re.search(r"/(\d+-\d+-\d+)\.txt$", self.file_path)
        return match.group(1) if match else ""

    @property
    def xml_url(self) -> str:
        """Get URL to the filing document."""
        return f"{SEC_BASE_URL}/Archives/{self.file_path}"


class SecForm4Importer:
    """
    Importer for SEC Form 4 insider ownership filings.

    Imports officers and directors from Form 4 filings into the people database.
    """

    def __init__(self):
        """Initialize the SEC Form 4 importer."""
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_REQUEST_DELAY:
            time.sleep(SEC_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_url(self, url: str) -> str:
        """Fetch URL content with proper headers and rate limiting."""
        self._rate_limit()
        req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def _fetch_index(self, year: int, quarter: int) -> list[Form4Filing]:
        """
        Fetch and parse quarterly form index for Form 4 filings.

        Args:
            year: Year (e.g., 2025)
            quarter: Quarter (1-4)

        Returns:
            List of Form4Filing objects
        """
        url = f"{SEC_FULL_INDEX_URL}/{year}/QTR{quarter}/form.idx"
        logger.info(f"Fetching index: {url}")

        try:
            content = self._fetch_url(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Index not found: {year} Q{quarter}")
                return []
            raise

        filings = []
        for line in content.split("\n"):
            # Form 4 lines start with "4 " followed by spaces and company name
            # Format: Form Type   Company Name   CIK   Date Filed   File Name
            if not line.startswith("4 "):
                continue

            # Parse fixed-width format
            # Columns are roughly: 0-12 (form), 13-75 (company), 76-87 (cik), 88-99 (date), 100+ (file)
            parts = line.split()
            if len(parts) < 5:
                continue

            # Extract fields - the format is space-padded fixed width
            form_type = parts[0]
            # Company name is everything between form type and CIK (which is numeric)
            # Find CIK by looking for numeric field
            cik_idx = -1
            for i, part in enumerate(parts[1:], 1):
                if part.isdigit() and len(part) >= 6:
                    cik_idx = i
                    break

            if cik_idx == -1:
                continue

            company_name = " ".join(parts[1:cik_idx])
            cik = parts[cik_idx]
            date_filed = parts[cik_idx + 1] if cik_idx + 1 < len(parts) else ""
            file_path = parts[cik_idx + 2] if cik_idx + 2 < len(parts) else ""

            if not file_path:
                continue

            filings.append(Form4Filing(
                form_type=form_type,
                company_name=company_name,
                cik=cik.zfill(10),
                date_filed=date_filed,
                file_path=file_path,
            ))

        logger.info(f"Found {len(filings)} Form 4 filings for {year} Q{quarter}")
        return filings

    def _parse_form4_xml(self, content: str) -> Iterator[PersonRecord]:
        """
        Parse Form 4 XML content and yield PersonRecord objects.

        A single Form 4 can have multiple reporting owners, each yielding a record.
        """
        # Extract XML from the SEC filing wrapper
        xml_match = re.search(r"<\?xml.*?</ownershipDocument>", content, re.DOTALL)
        if not xml_match:
            return

        xml_content = xml_match.group(0)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.debug(f"Failed to parse Form 4 XML: {e}")
            return

        # Extract issuer info
        issuer = root.find("issuer")
        if issuer is None:
            return

        issuer_cik = issuer.findtext("issuerCik", "").lstrip("0")
        issuer_name = issuer.findtext("issuerName", "")
        issuer_ticker = issuer.findtext("issuerTradingSymbol", "")

        if not issuer_cik or not issuer_name:
            return

        # Extract period of report (filing date)
        period_of_report = root.findtext("periodOfReport", "")

        # Process each reporting owner
        for owner in root.findall("reportingOwner"):
            owner_id = owner.find("reportingOwnerId")
            if owner_id is None:
                continue

            owner_cik = owner_id.findtext("rptOwnerCik", "").lstrip("0")
            owner_name = owner_id.findtext("rptOwnerName", "")

            if not owner_cik or not owner_name:
                continue

            # Get relationship info
            relationship = owner.find("reportingOwnerRelationship")
            is_director = False
            is_officer = False
            officer_title = ""
            is_ten_percent_owner = False

            if relationship is not None:
                is_director = relationship.findtext("isDirector", "0") == "1"
                is_officer = relationship.findtext("isOfficer", "0") == "1"
                officer_title = relationship.findtext("officerTitle", "") or ""
                is_ten_percent_owner = relationship.findtext("isTenPercentOwner", "0") == "1"

            # Skip if no relationship at all
            if not is_director and not is_officer and not is_ten_percent_owner:
                continue

            # Map to PersonType and role
            person_type = _map_to_person_type(is_director, is_officer, is_ten_percent_owner, officer_title)
            role = _extract_officer_role(is_director, is_officer, is_ten_percent_owner, officer_title)

            # Create unique source_id from owner CIK + issuer CIK
            # This allows same person to have multiple records for different companies
            source_id = f"{owner_cik}_{issuer_cik}"

            # Build record data
            record_data = {
                "owner_cik": owner_cik,
                "issuer_cik": issuer_cik,
                "issuer_name": issuer_name,
                "issuer_ticker": issuer_ticker,
                "is_director": is_director,
                "is_officer": is_officer,
                "officer_title": officer_title,
                "period_of_report": period_of_report,
            }

            yield PersonRecord(
                name=_normalize_name(owner_name),
                source="sec_edgar",
                source_id=source_id,
                country="US",
                person_type=person_type,
                known_for_role=role,
                known_for_org=issuer_name,
                # Note: known_for_org_id will be set during import if org exists in DB
                from_date=period_of_report,
                record=record_data,
            )

    def _fetch_and_parse_filing(self, filing: Form4Filing) -> Iterator[PersonRecord]:
        """Fetch a Form 4 filing and parse it for person records."""
        try:
            content = self._fetch_url(filing.xml_url)
            yield from self._parse_form4_xml(content)
        except Exception as e:
            logger.debug(f"Failed to fetch/parse {filing.accession_number}: {e}")

    def import_quarter(
        self,
        year: int,
        quarter: int,
        start_index: int = 0,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> Iterator[PersonRecord]:
        """
        Import Form 4 filings for a specific quarter.

        Args:
            year: Year (e.g., 2025)
            quarter: Quarter (1-4)
            start_index: Index to start from (for resume)
            limit: Optional limit on number of records
            progress_callback: Optional callback(filing_index, accession, records_yielded)

        Yields:
            PersonRecord for each officer/director
        """
        filings = self._fetch_index(year, quarter)

        if not filings:
            return

        count = 0
        for i, filing in enumerate(filings):
            if i < start_index:
                continue

            if limit and count >= limit:
                break

            for record in self._fetch_and_parse_filing(filing):
                yield record
                count += 1

                if limit and count >= limit:
                    break

                if count % 1000 == 0:
                    logger.info(f"Imported {count} records from {year} Q{quarter}")

            if progress_callback:
                progress_callback(i, filing.accession_number, count)

    def import_range(
        self,
        start_year: int = 2020,
        end_year: Optional[int] = None,
        limit: Optional[int] = None,
        resume: bool = False,
        progress_callback: Optional[Callable[[int, int, int, str, int], None]] = None,
    ) -> Iterator[PersonRecord]:
        """
        Import Form 4 filings for a range of years.

        Args:
            start_year: First year to import
            end_year: Last year to import (defaults to current year)
            limit: Optional total limit on records
            resume: If True, resume from saved progress
            progress_callback: Optional callback(year, quarter, filing_index, accession, total)

        Yields:
            PersonRecord for each officer/director
        """
        if end_year is None:
            end_year = datetime.now().year

        # Load or initialize progress
        progress = None
        if resume:
            progress = Form4Progress.load()
            if progress:
                logger.info(f"Resuming from {progress.year} Q{progress.quarter} index {progress.filing_index}")
                logger.info(f"Previously imported: {progress.total_imported} records")

        if progress is None:
            progress = Form4Progress(year=start_year, quarter=1)

        count = progress.total_imported

        for year in range(progress.year or start_year, end_year + 1):
            start_q = progress.quarter if year == progress.year else 1

            for quarter in range(start_q, 5):
                start_idx = progress.filing_index if (year == progress.year and quarter == progress.quarter) else 0

                logger.info(f"Processing {year} Q{quarter} (starting at index {start_idx})")

                def track_progress(filing_idx: int, accession: str, quarter_count: int) -> None:
                    progress.year = year
                    progress.quarter = quarter
                    progress.filing_index = filing_idx
                    progress.total_imported = count + quarter_count
                    progress.last_accession = accession
                    # Save progress periodically
                    if filing_idx % 100 == 0:
                        progress.save()
                    if progress_callback:
                        progress_callback(year, quarter, filing_idx, accession, progress.total_imported)

                quarter_limit = limit - count if limit else None

                for record in self.import_quarter(year, quarter, start_idx, quarter_limit, track_progress):
                    yield record
                    count += 1

                    if limit and count >= limit:
                        progress.total_imported = count
                        progress.save()
                        return

                # Reset filing index for next quarter
                progress.filing_index = 0

        # Clear progress on successful completion
        Form4Progress.clear()
        logger.info(f"Completed Form 4 import: {count} total records")

    def get_available_quarters(self, start_year: int = 2020) -> list[tuple[int, int]]:
        """
        Get list of available (year, quarter) pairs.

        Args:
            start_year: First year to check

        Returns:
            List of (year, quarter) tuples
        """
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1

        quarters = []
        for year in range(start_year, current_year + 1):
            max_q = current_quarter if year == current_year else 4
            for quarter in range(1, max_q + 1):
                quarters.append((year, quarter))

        return quarters
