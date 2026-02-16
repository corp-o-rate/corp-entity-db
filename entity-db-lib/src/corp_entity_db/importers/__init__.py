"""
Data importers for the entity database.

Provides importers for various data sources:
- GLEIF: Legal Entity Identifier data
- SEC Edgar: US SEC company data
- SEC Form 4: US SEC insider ownership data (officers/directors)
- Companies House: UK company data
- Wikidata: Wikipedia/Wikidata organization data (SPARQL-based, may timeout)
- Wikidata People: Notable people from Wikipedia/Wikidata (SPARQL-based, may timeout)
- Wikidata Dump: Bulk import from Wikidata JSON dump (recommended for large imports)
"""

from .gleif import GleifImporter
from .sec_edgar import SecEdgarImporter
from .sec_form4 import SecForm4Importer
from .companies_house import CompaniesHouseImporter
from .companies_house_officers import CompaniesHouseOfficersImporter
from .wikidata import WikidataImporter
from .wikidata_people import WikidataPeopleImporter
from .wikidata_dump import WikidataDumpImporter

__all__ = [
    "GleifImporter",
    "SecEdgarImporter",
    "SecForm4Importer",
    "CompaniesHouseImporter",
    "CompaniesHouseOfficersImporter",
    "WikidataImporter",
    "WikidataPeopleImporter",
    "WikidataDumpImporter",
]
