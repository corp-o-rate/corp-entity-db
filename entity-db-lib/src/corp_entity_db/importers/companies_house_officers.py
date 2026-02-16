"""
Companies House officers bulk data importer.

Imports officer/director data from Companies House bulk data files (Prod195).
The data is in fixed-width format with `<` as field separators.

Data format:
- Header: DDDDSNAP{product_id}{date}
- Company record (type 1): company_number + 1 + company_name
- Officer record (type 2/3): company_number + type + officer_details

Officer record structure (after company_number):
- Position 8: Record type (2=address only, 3=full appointment)
- Position 9-10: Sub-type (30=current, 01=resigned, etc.)
- Position 11-23: Person ID
- Position 24-31: Appointment date (YYYYMMDD)
- Position 32-39: (varies)
- Position 40-47: Postcode
- Position 48-53: Birth date (YYYYMM)
- Then `<`-separated fields: Title, Forenames, Surname, ..., Occupation, Nationality, Country

Resume support:
- Progress tracked by file index and line number
- Progress saved to JSON file for resume on interruption
"""

import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from ..models import PersonRecord, PersonType

logger = logging.getLogger(__name__)

# Default progress file path
DEFAULT_PROGRESS_PATH = Path.home() / ".cache" / "corp-extractor" / "ch-officers-progress.json"


def _normalize_name(name: str) -> str:
    """Normalize a person name for consistent storage."""
    if not name:
        return ""
    # Remove extra whitespace and title case
    name = " ".join(name.split())
    return name.title()


def _parse_date(date_str: str) -> Optional[str]:
    """Parse date from YYYYMMDD or YYYYMM format to ISO format."""
    date_str = date_str.strip()
    if not date_str or not date_str.isdigit():
        return None

    if len(date_str) == 8:  # YYYYMMDD
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            return None
    elif len(date_str) == 6:  # YYYYMM
        try:
            return f"{date_str[:4]}-{date_str[4:6]}"
        except Exception:
            return None
    return None


def _map_role_to_person_type(role: str) -> PersonType:
    """Map officer role to PersonType."""
    role_lower = role.lower()

    # Executive roles
    if any(x in role_lower for x in ["director", "ceo", "cfo", "cto", "coo", "chief", "managing", "president", "chairman"]):
        return PersonType.EXECUTIVE

    # Legal roles
    if any(x in role_lower for x in ["secretary", "solicitor", "lawyer", "legal"]):
        return PersonType.LEGAL

    # Professional roles
    if any(x in role_lower for x in ["accountant", "engineer", "architect", "doctor", "consultant"]):
        return PersonType.PROFESSIONAL

    return PersonType.EXECUTIVE  # Default to executive for company officers


@dataclass
class CHProgress:
    """Tracks progress through Companies House bulk import."""
    file_index: int = 0
    line_number: int = 0
    total_imported: int = 0
    last_company: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: Path = DEFAULT_PROGRESS_PATH) -> None:
        """Save progress to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump({
                "file_index": self.file_index,
                "line_number": self.line_number,
                "total_imported": self.total_imported,
                "last_company": self.last_company,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
            }, f, indent=2)
        logger.debug(f"Saved progress: file={self.file_index}, line={self.line_number}")

    @classmethod
    def load(cls, path: Path = DEFAULT_PROGRESS_PATH) -> Optional["CHProgress"]:
        """Load progress from JSON file."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
            return None

    @staticmethod
    def clear(path: Path = DEFAULT_PROGRESS_PATH) -> None:
        """Delete progress file."""
        if path.exists():
            path.unlink()
            logger.info(f"Cleared progress file: {path}")


class CompaniesHouseOfficersImporter:
    """
    Importer for Companies House bulk officers data.

    Parses the Prod195 fixed-width format files and extracts officer records.
    """

    def __init__(self):
        """Initialize the importer."""
        self._current_company_number: str = ""
        self._current_company_name: str = ""

    def _parse_officer_line(self, line: str) -> Optional[PersonRecord]:
        """
        Parse a single officer record line per CH specification.

        Returns PersonRecord or None if line is not a valid officer record.

        Fixed-width field positions (1-indexed in spec, 0-indexed here):
        - 0-7: Company Number (8)
        - 8: Record Type (1) - '1'=company, '2'=person
        - 9: App Date Origin (1)
        - 10-11: Appointment Type (2) - 00-19
        - 12-23: Person Number (12)
        - 24: Corporate Indicator (1) - 'Y'=corporate, ' '=individual
        - 25-31: Filler (7)
        - 32-39: Appointment Date (8) - CCYYMMDD
        - 40-47: Resignation Date (8) - CCYYMMDD
        - 48-55: Person Postcode (8)
        - 56-63: Partial DOB (8) - CCYYMM + 2 spaces
        - 64-71: Full DOB (8) - CCYYMMDD
        - 72-75: Variable Data Length (4)
        - 76+: Variable Data (<-delimited, 14 fields)

        Variable data fields (14 total):
        0:TITLE, 1:FORENAMES, 2:SURNAME, 3:HONOURS, 4:CARE_OF, 5:PO_BOX,
        6:ADDRESS1, 7:ADDRESS2, 8:POST_TOWN, 9:COUNTY, 10:COUNTRY,
        11:OCCUPATION, 12:NATIONALITY, 13:USUAL_RESIDENTIAL_COUNTRY
        """
        if len(line) < 76:
            return None

        # Extract fixed-width fields
        company_number = line[0:8].strip()
        record_type = line[8:9]

        # Type 1 is company record - extract company name
        if record_type == "1":
            # Company record: positions 32-35 = officer count, 36-39 = name length, 40+ = name
            name_part = line[40:].split("<")[0].strip()
            self._current_company_number = company_number
            self._current_company_name = name_part
            return None

        # Only process officer records (type 2)
        if record_type != "2":
            return None

        # Position 24: Corporate Indicator - 'Y' = corporate officer, space = individual
        corporate_indicator = line[24:25]
        if corporate_indicator == "Y":
            # Skip corporate officers (companies acting as secretary)
            return None

        # Use current company info
        if company_number != self._current_company_number:
            self._current_company_number = company_number
            self._current_company_name = ""

        # Person ID: positions 12-23 (12 chars)
        person_id = line[12:24].strip()

        # Appointment date: positions 32-39 (CCYYMMDD)
        appt_date_raw = line[32:40].strip()
        appointment_date = _parse_date(appt_date_raw)

        # Resignation date: positions 40-47 (CCYYMMDD)
        res_date_raw = line[40:48].strip()
        resignation_date = _parse_date(res_date_raw) if res_date_raw else None

        # Postcode: positions 48-55
        postcode = line[48:56].strip()

        # Partial DOB: positions 56-63 (CCYYMM + 2 spaces)
        partial_dob_raw = line[56:62].strip()
        birth_date = _parse_date(partial_dob_raw)

        # Full DOB: positions 64-71 (CCYYMMDD) - prefer this if available
        full_dob_raw = line[64:72].strip()
        if full_dob_raw:
            full_dob = _parse_date(full_dob_raw)
            if full_dob:
                birth_date = full_dob

        # Variable data starts at position 76
        # First we need to skip the variable data length field
        var_start = 76
        if len(line) <= var_start:
            return None

        # Parse `<`-separated fields (14 defined fields)
        var_section = line[var_start:]
        fields = var_section.split("<")

        # Field indices (0-based): 0=TITLE, 1=FORENAMES, 2=SURNAME, ..., 11=OCCUPATION, 12=NATIONALITY, 13=COUNTRY
        forenames = fields[1].strip() if len(fields) > 1 else ""
        surname = fields[2].strip() if len(fields) > 2 else ""

        # Build full name (without title for cleaner data)
        name_parts = []
        if forenames:
            name_parts.append(forenames)
        if surname:
            name_parts.append(surname)

        full_name = _normalize_name(" ".join(name_parts))
        if not full_name or not surname:
            return None

        # Get occupation (field 11), nationality (field 12)
        occupation = fields[11].strip() if len(fields) > 11 else ""
        nationality = fields[12].strip() if len(fields) > 12 else ""

        # Determine role from occupation
        role = occupation if occupation else "Director"
        person_type = _map_role_to_person_type(role)

        # Create unique source_id from person_id + company
        source_id = f"{person_id}_{company_number}"

        # Determine if current (no resignation date)
        is_current = resignation_date is None

        # Build record data
        record_data = {
            "person_id": person_id,
            "company_number": company_number,
            "company_name": self._current_company_name,
            "appointment_date": appointment_date,
            "resignation_date": resignation_date,
            "postcode": postcode,
            "occupation": occupation,
            "nationality": nationality,
            "is_current": is_current,
        }

        return PersonRecord(
            name=full_name,
            source="companies_house",
            source_id=source_id,
            country="GB",
            person_type=person_type,
            known_for_role=role,
            known_for_org=self._current_company_name,
            from_date=appointment_date,
            to_date=resignation_date,
            birth_date=birth_date,
            record=record_data,
        )

    def import_from_zip(
        self,
        zip_path: str | Path,
        limit: Optional[int] = None,
        resume: bool = False,
        current_only: bool = True,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Iterator[PersonRecord]:
        """
        Import officer records from Companies House bulk zip file.

        Args:
            zip_path: Path to the Prod195 zip file
            limit: Optional limit on number of records
            resume: If True, resume from saved progress
            current_only: If True, only import current officers (sub_type=30)
            progress_callback: Optional callback(file_idx, line_num, total)

        Yields:
            PersonRecord for each officer
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        # Load or initialize progress
        progress = CHProgress.load() if resume else None
        if progress:
            logger.info(f"Resuming from file {progress.file_index}, line {progress.line_number}")
            logger.info(f"Previously imported: {progress.total_imported}")
        else:
            progress = CHProgress()

        count = progress.total_imported

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get list of .dat files, sorted
            dat_files = sorted([n for n in zf.namelist() if n.endswith(".dat")])
            logger.info(f"Found {len(dat_files)} data files in archive")

            for file_idx, filename in enumerate(dat_files):
                # Skip files before resume point
                if file_idx < progress.file_index:
                    continue

                logger.info(f"Processing file {file_idx + 1}/{len(dat_files)}: {filename}")

                start_line = progress.line_number if file_idx == progress.file_index else 0

                with zf.open(filename) as f:
                    for line_num, line_bytes in enumerate(f):
                        # Skip lines before resume point
                        if line_num < start_line:
                            continue

                        if limit and count >= limit:
                            progress.file_index = file_idx
                            progress.line_number = line_num
                            progress.total_imported = count
                            progress.save()
                            return

                        try:
                            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
                        except Exception:
                            continue

                        # Skip header
                        if line.startswith("DDDD"):
                            continue

                        # Parse officer record
                        record = self._parse_officer_line(line)
                        if record:
                            # Skip resigned officers if current_only
                            if current_only and record.to_date:
                                continue

                            yield record
                            count += 1

                            if count % 10000 == 0:
                                logger.info(f"Imported {count} officers...")
                                progress.file_index = file_idx
                                progress.line_number = line_num
                                progress.total_imported = count
                                progress.last_company = self._current_company_number
                                progress.save()

                        if progress_callback and line_num % 10000 == 0:
                            progress_callback(file_idx, line_num, count)

                # Reset line counter for next file
                progress.line_number = 0

        # Clear progress on successful completion
        CHProgress.clear()
        logger.info(f"Completed CH officers import: {count} total records")

    def import_from_file(
        self,
        file_path: str | Path,
        limit: Optional[int] = None,
        current_only: bool = True,
    ) -> Iterator[PersonRecord]:
        """
        Import from a single uncompressed .dat file.

        Args:
            file_path: Path to .dat file
            limit: Optional limit
            current_only: Only current officers

        Yields:
            PersonRecord for each officer
        """
        file_path = Path(file_path)
        count = 0

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if limit and count >= limit:
                    break

                line = line.rstrip("\n\r")
                if line.startswith("DDDD"):
                    continue

                record = self._parse_officer_line(line)
                if record:
                    if current_only and record.to_date:
                        continue
                    yield record
                    count += 1

        logger.info(f"Imported {count} officers from {file_path}")
