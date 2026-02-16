"""
Wikidata importer for the company/organization database.

Imports organization data from Wikidata using SPARQL queries
into the embedding database for entity name matching.

Supports 35+ entity types across 4 categories:

Organizations (highest priority):
- Organizations, nonprofits, NGOs, foundations
- Government agencies, international organizations
- Political parties, trade unions
- Educational institutions, universities, research institutes
- Hospitals, sports clubs

Companies:
- Companies with LEI codes or stock tickers
- Public companies, business enterprises, corporations
- Subsidiaries, conglomerates

Industry-specific:
- Banks, insurance companies, investment companies
- Airlines, retailers, manufacturers
- Pharma, tech companies, law firms
- Record labels, film studios, video game companies

Property-based (catches untyped entities):
- Entities with CEO, subsidiaries, legal form
- Entities with employee count or revenue data

Uses the public Wikidata Query Service endpoint.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

from ..models import CompanyRecord, EntityType

logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Simpler SPARQL query - directly query for companies with LEI codes (fastest, most reliable)
# Avoids property path wildcards (wdt:P279*) which timeout on Wikidata
LEI_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P1278 ?lei.
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for companies with stock exchange listing (has ticker)
TICKER_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?ticker ?exchange ?exchangeLabel ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P414 ?exchange.
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for direct instances of public company (Q891723) - no subclass traversal
PUBLIC_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q891723.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for direct instances of business enterprise (Q4830453) - no subclass traversal
BUSINESS_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q4830453.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for direct instances of organization (Q43229) - includes NGOs, gov agencies, etc.
ORGANIZATION_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q43229.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for non-profit organizations (Q163740)
NONPROFIT_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q163740.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for government agencies (Q327333)
GOV_AGENCY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q327333.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for enterprises (Q6881511) - broader than business enterprise
ENTERPRISE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q6881511.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for corporations (Q167037)
CORPORATION_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q167037.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for subsidiaries (Q658255)
SUBSIDIARY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q658255.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for banks (Q22687)
BANK_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q22687.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for insurance companies (Q6881511)
INSURANCE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q1145276.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for airlines (Q46970)
AIRLINE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q46970.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for law firms (Q613142)
LAW_FIRM_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q613142.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for educational institutions (Q2385804)
EDUCATIONAL_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q2385804.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for universities (Q3918)
UNIVERSITY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q3918.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for research institutes (Q31855)
RESEARCH_INSTITUTE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q31855.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for political parties (Q7278)
POLITICAL_PARTY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q7278.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for trade unions (Q178790)
TRADE_UNION_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q178790.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for NGOs (Q79913)
NGO_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q79913.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for foundations (Q157031)
FOUNDATION_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q157031.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for international organizations (Q484652)
INTL_ORG_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q484652.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for sports teams/clubs (Q476028)
SPORTS_CLUB_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q476028.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for hospitals (Q16917)
HOSPITAL_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q16917.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for record labels (Q18127)
RECORD_LABEL_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q18127.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for film studios (Q1366047)
FILM_STUDIO_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q1366047.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for video game companies (Q1137109)
VIDEO_GAME_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q1137109.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for pharmaceutical companies (Q507619)
PHARMA_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q507619.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for tech companies (Q2979960)
TECH_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q2979960.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for retailers (Q1631111)
RETAILER_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q1631111.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for manufacturers (Q187652)
MANUFACTURER_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q187652.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for conglomerates (Q206652)
CONGLOMERATE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q206652.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Query for investment companies (Q380649)
INVESTMENT_COMPANY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P31 wd:Q380649.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities with a CEO (P169) - likely companies
HAS_CEO_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P169 ?ceo.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities with subsidiaries (P355) - parent companies
HAS_SUBSIDIARIES_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P355 ?subsidiary.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities owned by another entity (P127) - subsidiaries/companies
OWNED_BY_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P127 ?owner.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities with legal form (P1454) - structured companies
HAS_LEGAL_FORM_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P1454 ?legalForm.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities with employees count (P1128) - organizations
HAS_EMPLOYEES_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P1128 ?employees.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""

# Property-based query: entities with revenue (P2139) - companies
HAS_REVENUE_QUERY = """
SELECT ?company ?companyLabel ?lei ?ticker ?country ?countryLabel ?inception ?dissolution WHERE {
  ?company wdt:P2139 ?revenue.
  OPTIONAL { ?company wdt:P1278 ?lei. }
  OPTIONAL { ?company wdt:P249 ?ticker. }
  OPTIONAL { ?company wdt:P17 ?country. }
  OPTIONAL { ?company wdt:P571 ?inception. }
  OPTIONAL { ?company wdt:P576 ?dissolution. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT %d
OFFSET %d
"""


# Query types available for import - organized by category
# Organization types (highest priority - run first)
ORG_QUERY_TYPES = {
    "organization": ORGANIZATION_QUERY,
    "nonprofit": NONPROFIT_QUERY,
    "ngo": NGO_QUERY,
    "foundation": FOUNDATION_QUERY,
    "government": GOV_AGENCY_QUERY,
    "intl_org": INTL_ORG_QUERY,
    "political_party": POLITICAL_PARTY_QUERY,
    "trade_union": TRADE_UNION_QUERY,
    "educational": EDUCATIONAL_QUERY,
    "university": UNIVERSITY_QUERY,
    "research_institute": RESEARCH_INSTITUTE_QUERY,
    "hospital": HOSPITAL_QUERY,
    "sports_club": SPORTS_CLUB_QUERY,
}

# Company types
COMPANY_QUERY_TYPES = {
    "lei": LEI_COMPANY_QUERY,
    "ticker": TICKER_COMPANY_QUERY,
    "public": PUBLIC_COMPANY_QUERY,
    "business": BUSINESS_QUERY,
    "enterprise": ENTERPRISE_QUERY,
    "corporation": CORPORATION_QUERY,
    "subsidiary": SUBSIDIARY_QUERY,
    "conglomerate": CONGLOMERATE_QUERY,
}

# Industry-specific company types
INDUSTRY_QUERY_TYPES = {
    "bank": BANK_QUERY,
    "insurance": INSURANCE_QUERY,
    "airline": AIRLINE_QUERY,
    "law_firm": LAW_FIRM_QUERY,
    "pharma": PHARMA_QUERY,
    "tech_company": TECH_COMPANY_QUERY,
    "retailer": RETAILER_QUERY,
    "manufacturer": MANUFACTURER_QUERY,
    "investment_company": INVESTMENT_COMPANY_QUERY,
    "record_label": RECORD_LABEL_QUERY,
    "film_studio": FILM_STUDIO_QUERY,
    "video_game_company": VIDEO_GAME_COMPANY_QUERY,
}

# Property-based queries (catches entities not typed correctly)
PROPERTY_QUERY_TYPES = {
    "has_ceo": HAS_CEO_QUERY,
    "has_subsidiaries": HAS_SUBSIDIARIES_QUERY,
    "owned_by": OWNED_BY_QUERY,
    "has_legal_form": HAS_LEGAL_FORM_QUERY,
    "has_employees": HAS_EMPLOYEES_QUERY,
    "has_revenue": HAS_REVENUE_QUERY,
}

# All query types combined
QUERY_TYPES = {
    **ORG_QUERY_TYPES,
    **COMPANY_QUERY_TYPES,
    **INDUSTRY_QUERY_TYPES,
    **PROPERTY_QUERY_TYPES,
}

# Mapping from query type to EntityType
QUERY_TYPE_TO_ENTITY_TYPE: dict[str, EntityType] = {
    # Organizations
    "organization": EntityType.NONPROFIT,  # Generic org, default to nonprofit
    "nonprofit": EntityType.NONPROFIT,
    "ngo": EntityType.NGO,
    "foundation": EntityType.FOUNDATION,
    "government": EntityType.GOVERNMENT,
    "intl_org": EntityType.INTERNATIONAL_ORG,
    "political_party": EntityType.POLITICAL_PARTY,
    "trade_union": EntityType.TRADE_UNION,
    "educational": EntityType.EDUCATIONAL,
    "university": EntityType.EDUCATIONAL,
    "research_institute": EntityType.RESEARCH,
    "hospital": EntityType.HEALTHCARE,
    "sports_club": EntityType.SPORTS,

    # Companies
    "lei": EntityType.BUSINESS,
    "ticker": EntityType.BUSINESS,
    "public": EntityType.BUSINESS,
    "business": EntityType.BUSINESS,
    "enterprise": EntityType.BUSINESS,
    "corporation": EntityType.BUSINESS,
    "subsidiary": EntityType.BUSINESS,
    "conglomerate": EntityType.BUSINESS,

    # Industry-specific (all business)
    "bank": EntityType.BUSINESS,
    "insurance": EntityType.BUSINESS,
    "airline": EntityType.BUSINESS,
    "law_firm": EntityType.BUSINESS,
    "pharma": EntityType.BUSINESS,
    "tech_company": EntityType.BUSINESS,
    "retailer": EntityType.BUSINESS,
    "manufacturer": EntityType.BUSINESS,
    "investment_company": EntityType.FUND,
    "record_label": EntityType.MEDIA,
    "film_studio": EntityType.MEDIA,
    "video_game_company": EntityType.MEDIA,

    # Property-based (assume business as they have CEO/revenue/etc)
    "has_ceo": EntityType.BUSINESS,
    "has_subsidiaries": EntityType.BUSINESS,
    "owned_by": EntityType.BUSINESS,
    "has_legal_form": EntityType.BUSINESS,
    "has_employees": EntityType.UNKNOWN,  # Could be any org type
    "has_revenue": EntityType.BUSINESS,
}


class WikidataImporter:
    """
    Importer for Wikidata organization data.

    Uses SPARQL queries against the public Wikidata Query Service
    to fetch organizations including companies, nonprofits, government agencies, etc.

    Query categories (run in this order with import_all=True):

    Organizations:
    - organization: All organizations (Q43229)
    - nonprofit: Non-profit organizations (Q163740)
    - ngo: NGOs (Q79913)
    - foundation: Foundations (Q157031)
    - government: Government agencies (Q327333)
    - intl_org: International organizations (Q484652)
    - political_party: Political parties (Q7278)
    - trade_union: Trade unions (Q178790)
    - educational: Educational institutions (Q2385804)
    - university: Universities (Q3918)
    - research_institute: Research institutes (Q31855)
    - hospital: Hospitals (Q16917)
    - sports_club: Sports clubs (Q476028)

    Companies:
    - lei: Companies with LEI codes
    - ticker: Companies with stock exchange listings
    - public: Public companies (Q891723)
    - business: Business enterprises (Q4830453)
    - enterprise: Enterprises (Q6881511)
    - corporation: Corporations (Q167037)
    - subsidiary: Subsidiaries (Q658255)
    - conglomerate: Conglomerates (Q206652)

    Industry-specific:
    - bank: Banks (Q22687)
    - insurance: Insurance companies (Q1145276)
    - airline: Airlines (Q46970)
    - law_firm: Law firms (Q613142)
    - pharma: Pharmaceutical companies (Q507619)
    - tech_company: Tech companies (Q2979960)
    - retailer: Retailers (Q1631111)
    - manufacturer: Manufacturers (Q187652)
    - investment_company: Investment companies (Q380649)
    - record_label: Record labels (Q18127)
    - film_studio: Film studios (Q1366047)
    - video_game_company: Video game companies (Q1137109)

    Property-based (catches untyped entities):
    - has_ceo: Entities with CEO (P169)
    - has_subsidiaries: Entities with subsidiaries (P355)
    - owned_by: Entities owned by another (P127)
    - has_legal_form: Entities with legal form (P1454)
    - has_employees: Entities with employee count (P1128)
    - has_revenue: Entities with revenue (P2139)
    """

    def __init__(self, batch_size: int = 1000, delay_seconds: float = 2.0, timeout: int = 120):
        """
        Initialize the Wikidata importer.

        Args:
            batch_size: Number of records to fetch per SPARQL query (default 1000)
            delay_seconds: Delay between requests to be polite to the endpoint
            timeout: HTTP timeout in seconds (default 120)
        """
        self._batch_size = batch_size
        self._delay = delay_seconds
        self._timeout = timeout

    def import_from_sparql(
        self,
        limit: Optional[int] = None,
        query_type: str = "lei",
        import_all: bool = False,
    ) -> Iterator[CompanyRecord]:
        """
        Import organization records from Wikidata via SPARQL.

        Args:
            limit: Optional limit on total records
            query_type: Which query to use (see class docstring for full list).
                Common options:
                - "lei": Companies with LEI codes (default, fastest)
                - "organization": All organizations (Q43229)
                - "nonprofit": Non-profit organizations (Q163740)
                - "government": Government agencies (Q327333)
                - "has_ceo": Entities with CEO property (catches many companies)
            import_all: If True, run all query types sequentially in priority order:
                1. Organization types (nonprofits, gov agencies, NGOs, etc.)
                2. Company types (public companies, business enterprises, etc.)
                3. Industry-specific types (banks, airlines, pharma, etc.)
                4. Property-based queries (catches entities not properly typed)

        Yields:
            CompanyRecord for each organization
        """
        if import_all:
            yield from self._import_all_types(limit)
            return

        if query_type not in QUERY_TYPES:
            raise ValueError(f"Unknown query type: {query_type}. Use one of: {list(QUERY_TYPES.keys())}")

        query_template = QUERY_TYPES[query_type]
        entity_type = QUERY_TYPE_TO_ENTITY_TYPE.get(query_type, EntityType.UNKNOWN)
        logger.info(f"Starting Wikidata company import via SPARQL (query_type={query_type}, entity_type={entity_type.value})...")

        offset = 0
        total_count = 0
        seen_ids = set()  # Track seen Wikidata IDs to avoid duplicates

        while True:
            if limit and total_count >= limit:
                break

            batch_limit = min(self._batch_size, (limit - total_count) if limit else self._batch_size)
            query = query_template % (batch_limit, offset)

            logger.info(f"Fetching Wikidata batch at offset {offset}...")

            try:
                results = self._execute_sparql(query)
            except Exception as e:
                logger.error(f"SPARQL query failed at offset {offset}: {e}")
                break

            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                logger.info("No more results from Wikidata")
                break

            batch_count = 0
            for binding in bindings:
                if limit and total_count >= limit:
                    break

                record = self._parse_binding(binding, entity_type=entity_type)
                if record and record.source_id not in seen_ids:
                    seen_ids.add(record.source_id)
                    total_count += 1
                    batch_count += 1
                    yield record

            logger.info(f"Processed {batch_count} records from batch (total: {total_count})")

            if len(bindings) < batch_limit:
                # Last batch
                break

            offset += self._batch_size

            # Be polite to the endpoint
            if self._delay > 0:
                time.sleep(self._delay)

        logger.info(f"Completed Wikidata import: {total_count} records")

    def _import_all_types(self, limit: Optional[int]) -> Iterator[CompanyRecord]:
        """Import from all query types sequentially, deduplicating across types.

        Query categories are run in priority order:
        1. Organization types (nonprofits, gov agencies, NGOs, etc.)
        2. Company types (public companies, business enterprises, etc.)
        3. Industry-specific types (banks, airlines, pharma, etc.)
        4. Property-based queries (catches entities not properly typed)
        """
        seen_ids: set[str] = set()
        total_count = 0

        # Calculate per-category limits if a total limit is set
        num_categories = 4
        per_category_limit = limit // num_categories if limit else None

        # Run categories in priority order: organizations first
        categories = [
            ("Organizations", ORG_QUERY_TYPES, per_category_limit),
            ("Companies", COMPANY_QUERY_TYPES, per_category_limit),
            ("Industry-specific", INDUSTRY_QUERY_TYPES, per_category_limit),
            ("Property-based", PROPERTY_QUERY_TYPES, per_category_limit),
        ]

        for category_name, query_types, category_limit in categories:
            logger.info(f"=== Starting category: {category_name} ({len(query_types)} query types) ===")
            category_count = 0
            per_type_limit = category_limit // len(query_types) if category_limit else None

            for query_type in query_types:
                logger.info(f"Importing from query type: {query_type}")
                type_count = 0

                for record in self.import_from_sparql(limit=per_type_limit, query_type=query_type):
                    if record.source_id not in seen_ids:
                        seen_ids.add(record.source_id)
                        total_count += 1
                        type_count += 1
                        category_count += 1
                        yield record

                        if limit and total_count >= limit:
                            logger.info(f"Reached total limit of {limit} records")
                            return

                logger.info(f"Got {type_count} new records from {query_type} (total: {total_count})")

            logger.info(f"=== Completed {category_name}: {category_count} new records ===")

        logger.info(f"Completed all query types: {total_count} total records")

    @staticmethod
    def _parse_wikidata_date(date_str: Optional[str]) -> Optional[str]:
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
                "User-Agent": "corp-extractor/1.0 (company database builder)",
            }
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_binding(
        self,
        binding: dict[str, Any],
        entity_type: EntityType = EntityType.UNKNOWN,
    ) -> Optional[CompanyRecord]:
        """Parse a SPARQL result binding into a CompanyRecord."""
        try:
            # Get Wikidata entity ID
            company_uri = binding.get("company", {}).get("value", "")
            if not company_uri:
                return None

            # Extract QID from URI (e.g., "http://www.wikidata.org/entity/Q312" -> "Q312")
            wikidata_id = company_uri.split("/")[-1]
            if not wikidata_id.startswith("Q"):
                return None

            # Get label
            label = binding.get("companyLabel", {}).get("value", "")
            if not label or label == wikidata_id:  # Skip if no English label
                return None

            # Get optional fields
            lei = binding.get("lei", {}).get("value")
            ticker = binding.get("ticker", {}).get("value")
            exchange_label = binding.get("exchangeLabel", {}).get("value")
            country_label = binding.get("countryLabel", {}).get("value")
            inception_raw = binding.get("inception", {}).get("value")
            dissolution_raw = binding.get("dissolution", {}).get("value")

            # Parse dates (Wikidata returns ISO datetime, extract date part)
            from_date = WikidataImporter._parse_wikidata_date(inception_raw)
            to_date = WikidataImporter._parse_wikidata_date(dissolution_raw)

            # Build record data
            record_data: dict[str, Any] = {
                "wikidata_id": wikidata_id,
                "label": label,
            }
            if lei:
                record_data["lei"] = lei
            if ticker:
                record_data["ticker"] = ticker
            if exchange_label:
                record_data["exchange"] = exchange_label
            if country_label:
                record_data["country"] = country_label
            if from_date:
                record_data["inception"] = from_date
            if to_date:
                record_data["dissolution"] = to_date

            return CompanyRecord(
                name=label.strip(),
                source="wikipedia",  # Use "wikipedia" as source per schema
                source_id=wikidata_id,
                region=country_label or "",
                entity_type=entity_type,
                from_date=from_date,
                to_date=to_date,
                record=record_data,
            )

        except Exception as e:
            logger.debug(f"Failed to parse Wikidata binding: {e}")
            return None

    def search_company(self, name: str, limit: int = 10) -> list[CompanyRecord]:
        """
        Search for a specific company by name.

        Args:
            name: Company name to search for
            limit: Maximum results to return

        Returns:
            List of matching CompanyRecords
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

            # Check if it looks like a company
            company_keywords = ["company", "corporation", "inc", "ltd", "enterprise", "business"]
            if not any(kw in description.lower() for kw in company_keywords):
                continue

            record = CompanyRecord(
                name=label,
                source="wikipedia",
                source_id=qid,
                region="",  # Not available from search API
                record={
                    "wikidata_id": qid,
                    "label": label,
                    "description": description,
                },
            )
            results.append(record)

        return results
