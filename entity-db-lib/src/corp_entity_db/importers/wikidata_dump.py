"""
Wikidata dump importer for people and organizations.

Uses the Wikidata JSON dump (~100GB compressed) to import:
1. People: All humans (P31=Q5) with English Wikipedia articles
2. Organizations: All organizations with English Wikipedia articles

This avoids SPARQL query timeouts that occur with large result sets.
The dump is processed line-by-line to minimize memory usage.

Dump format:
- File: `latest-all.json.bz2` (~100GB) or `.gz` (~150GB)
- Format: JSON array where each line is a separate entity (after first `[` line)
- Each line: `{"type":"item","id":"Q123","labels":{...},"claims":{...},"sitelinks":{...}},`
- Streaming: Read line-by-line, strip trailing comma, parse JSON

Resume support:
- Progress is tracked by entity index (count of entities processed)
- Progress can be saved to a JSON file and loaded on resume
- On resume, entities are skipped efficiently until reaching the saved position
"""

import bz2
import gzip
import io
import json
import logging
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field

# Use orjson for ~3x faster JSON parsing if available
try:
    import orjson as _json_mod

    def _json_loads(data: bytes | str) -> dict:
        return _json_mod.loads(data)

    _HAVE_ORJSON = True
except ImportError:
    _json_loads = json.loads  # type: ignore[assignment]
    _HAVE_ORJSON = False

# Use indexed_bzip2 for parallel decompression if available
try:
    import indexed_bzip2 as _ibz2
    _HAVE_IBZ2 = True
except ImportError:
    _ibz2 = None  # type: ignore[assignment]
    _HAVE_IBZ2 = False
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..models import CompanyRecord, EntityType, LocationRecord, PersonRecord, PersonType, RoleRecord, SimplifiedLocationType

# Type alias for records that can be either people or orgs or locations
ImportRecord = PersonRecord | CompanyRecord | LocationRecord | RoleRecord

logger = logging.getLogger(__name__)

# Wikidata dump URLs - mirrors for faster downloads
# Primary is Wikimedia (slow), alternatives may be faster
DUMP_MIRRORS = [
    # Wikimedia Foundation (official, often slow)
    "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2",
    # Academic Torrents mirror (if available) - typically faster
    # Note: Check https://academictorrents.com/browse?search=wikidata for current links
]

# Default URL (can be overridden)
DUMP_URL = DUMP_MIRRORS[0]

# For even faster downloads, users can:
# 1. Use a torrent client with the Academic Torrents magnet link
# 2. Download from a regional Wikimedia mirror
# 3. Use aria2c with multiple connections: aria2c -x 16 -s 16 <url>

# =============================================================================
# POSITION TO PERSON TYPE MAPPING (P39 - position held)
# =============================================================================

# Executive positions (P39 values)
EXECUTIVE_POSITION_QIDS = {
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
}

# Politician positions (P39 values)
# Includes heads of state/government, legislators, and local officials
POLITICIAN_POSITION_QIDS = {
    # Heads of state/government
    "Q30461",     # president
    "Q14212",     # prime minister
    "Q83307",     # minister
    "Q2285706",   # head of government
    "Q48352",     # head of state
    "Q116",       # monarch
    "Q382617",    # governor
    "Q212071",    # mayor
    "Q1553195",   # deputy prime minister
    "Q1670573",   # cabinet minister
    "Q13218630",  # secretary of state
    "Q581682",    # vice president

    # Legislators - national
    "Q4175034",   # legislator
    "Q486839",    # member of parliament
    "Q193391",    # member of national legislature
    "Q484529",    # member of congress
    "Q1711695",   # senator
    "Q18941264",  # member of the House of Representatives (US)
    "Q16707842",  # member of the House of Commons (UK)
    "Q18015642",  # member of the House of Lords (UK)
    "Q17295570",  # member of the Bundestag (Germany)
    "Q27169",     # member of the European Parliament
    "Q64366569",  # member of Dáil Éireann (Ireland)
    "Q19823090",  # member of the Riksdag (Sweden)
    "Q18229048",  # member of Sejm (Poland)
    "Q21032547",  # member of the National Assembly (France)
    "Q64511800",  # member of the Knesset (Israel)
    "Q50393121",  # member of the State Duma (Russia)
    "Q18558055",  # member of the Diet (Japan)
    "Q109862831", # member of Lok Sabha (India)
    "Q63078776",  # member of the Canadian House of Commons
    "Q83767637",  # member of the Australian House of Representatives

    # Legislators - regional/local
    "Q4382506",   # member of state legislature
    "Q17765219",  # member of regional parliament
    "Q1752514",   # councillor (local government)
    "Q18824436",  # city councillor

    # Other political offices
    "Q294414",    # public office (generic)
    "Q889821",    # ambassador
    "Q15966511",  # diplomat
    "Q334344",    # lord lieutenant
    "Q16533",     # judge (some are appointed politicians)
    "Q3099732",   # ombudsman
    "Q1500443",   # prefect
    "Q611644",    # envoy
    "Q2824523",   # political commissar
}

# =============================================================================
# OCCUPATION TO PERSON TYPE MAPPING (P106 - occupation)
# =============================================================================

OCCUPATION_TO_TYPE: dict[str, PersonType] = {
    # Politicians (elected officials)
    "Q82955": PersonType.POLITICIAN,     # politician
    "Q193391": PersonType.POLITICIAN,    # member of parliament
    "Q372436": PersonType.POLITICIAN,    # statesperson

    # Government (civil servants, diplomats, appointed officials)
    "Q212238": PersonType.GOVERNMENT,    # civil servant
    "Q806798": PersonType.GOVERNMENT,    # diplomat
    "Q15627169": PersonType.GOVERNMENT,  # trade unionist (often govt-adjacent)

    # Military
    "Q189290": PersonType.MILITARY,      # military officer
    "Q47064": PersonType.MILITARY,       # military personnel
    "Q4991371": PersonType.MILITARY,     # soldier
    "Q10669499": PersonType.MILITARY,    # naval officer
    "Q11974939": PersonType.MILITARY,    # air force officer
    "Q10974448": PersonType.MILITARY,    # army officer

    # Legal professionals
    "Q16533": PersonType.LEGAL,          # judge
    "Q40348": PersonType.LEGAL,          # lawyer
    "Q185351": PersonType.LEGAL,         # jurist
    "Q3242871": PersonType.LEGAL,        # prosecutor
    "Q1792450": PersonType.LEGAL,        # barrister
    "Q3406182": PersonType.LEGAL,        # solicitor

    # Athletes
    "Q2066131": PersonType.ATHLETE,      # athlete
    "Q937857": PersonType.ATHLETE,       # football player
    "Q3665646": PersonType.ATHLETE,      # basketball player
    "Q10871364": PersonType.ATHLETE,     # baseball player
    "Q19204627": PersonType.ATHLETE,     # ice hockey player
    "Q10843402": PersonType.ATHLETE,     # tennis player
    "Q13381376": PersonType.ATHLETE,     # golfer
    "Q11338576": PersonType.ATHLETE,     # boxer
    "Q10873124": PersonType.ATHLETE,     # swimmer
    "Q11303721": PersonType.ATHLETE,     # racing driver
    "Q10833314": PersonType.ATHLETE,     # cricket player
    "Q13141064": PersonType.ATHLETE,     # rugby player

    # Artists (traditional creative professions)
    "Q33999": PersonType.ARTIST,         # actor
    "Q177220": PersonType.ARTIST,        # singer
    "Q639669": PersonType.ARTIST,        # musician
    "Q2526255": PersonType.ARTIST,       # film director
    "Q36180": PersonType.ARTIST,         # writer
    "Q483501": PersonType.ARTIST,        # artist
    "Q488205": PersonType.ARTIST,        # singer-songwriter
    "Q753110": PersonType.ARTIST,        # songwriter
    "Q2405480": PersonType.ARTIST,       # voice actor
    "Q10800557": PersonType.ARTIST,      # film actor
    "Q3455803": PersonType.ARTIST,       # director
    "Q28389": PersonType.ARTIST,         # screenwriter
    "Q6625963": PersonType.ARTIST,       # comedian
    "Q2259451": PersonType.ARTIST,       # stand-up comedian
    "Q2490358": PersonType.ARTIST,       # choreographer
    "Q2722764": PersonType.ARTIST,       # DJ (disc jockey)
    "Q183945": PersonType.ARTIST,        # record producer
    "Q3282637": PersonType.ARTIST,       # film producer
    "Q49757": PersonType.ARTIST,         # poet
    "Q28640": PersonType.ARTIST,         # illustrator
    "Q1028181": PersonType.ARTIST,       # painter
    "Q1281618": PersonType.ARTIST,       # sculptor
    "Q33231": PersonType.ARTIST,         # photographer
    "Q806349": PersonType.ARTIST,        # band leader
    "Q855091": PersonType.ARTIST,        # rapper
    "Q4351403": PersonType.ARTIST,       # novelist
    "Q158852": PersonType.ARTIST,        # conductor (music)
    "Q486748": PersonType.ARTIST,        # pianist
    "Q1415090": PersonType.ARTIST,       # guitarist

    # Media (internet/social media personalities)
    "Q6168364": PersonType.MEDIA,        # YouTuber
    "Q15077007": PersonType.MEDIA,       # podcaster
    "Q17125263": PersonType.MEDIA,       # social media influencer
    "Q15981151": PersonType.MEDIA,       # internet celebrity
    "Q2059704": PersonType.MEDIA,        # television personality
    "Q4610556": PersonType.MEDIA,        # model
    "Q578109": PersonType.MEDIA,         # television producer
    "Q2516866": PersonType.MEDIA,        # publisher
    "Q93191800": PersonType.MEDIA,       # content creator
    "Q105756498": PersonType.MEDIA,      # streamer (Twitch etc.)

    # Professionals (known for their profession/work)
    "Q39631": PersonType.PROFESSIONAL,   # physician/doctor
    "Q774306": PersonType.PROFESSIONAL,  # surgeon
    "Q1234713": PersonType.PROFESSIONAL, # dentist
    "Q15924224": PersonType.PROFESSIONAL, # psychiatrist
    "Q212980": PersonType.PROFESSIONAL,  # psychologist
    "Q81096": PersonType.PROFESSIONAL,   # engineer
    "Q42603": PersonType.PROFESSIONAL,   # priest/clergy
    "Q432386": PersonType.PROFESSIONAL,  # architect
    "Q3621491": PersonType.PROFESSIONAL, # nurse
    "Q18805": PersonType.PROFESSIONAL,   # pharmacist
    "Q15895020": PersonType.PROFESSIONAL, # veterinarian
    "Q131512": PersonType.PROFESSIONAL,  # chef
    "Q3499072": PersonType.PROFESSIONAL, # pilot
    "Q15895449": PersonType.PROFESSIONAL, # accountant
    "Q806750": PersonType.PROFESSIONAL,  # consultant
    "Q584301": PersonType.PROFESSIONAL,  # economist (often professional)
    "Q1371925": PersonType.PROFESSIONAL, # real estate agent
    "Q266569": PersonType.PROFESSIONAL,  # librarian
    "Q5323050": PersonType.PROFESSIONAL, # electrical engineer
    "Q13582652": PersonType.PROFESSIONAL, # civil engineer
    "Q81965": PersonType.PROFESSIONAL,   # software engineer
    "Q5482740": PersonType.PROFESSIONAL, # data scientist

    # Academics
    "Q121594": PersonType.ACADEMIC,      # professor
    "Q3400985": PersonType.ACADEMIC,     # academic
    "Q1622272": PersonType.ACADEMIC,     # university professor

    # Scientists
    "Q901": PersonType.SCIENTIST,        # scientist
    "Q1650915": PersonType.SCIENTIST,    # researcher
    "Q169470": PersonType.SCIENTIST,     # physicist
    "Q593644": PersonType.SCIENTIST,     # chemist
    "Q864503": PersonType.SCIENTIST,     # biologist
    "Q11063": PersonType.SCIENTIST,      # astronomer

    # Journalists
    "Q1930187": PersonType.JOURNALIST,   # journalist
    "Q13590141": PersonType.JOURNALIST,  # news presenter
    "Q947873": PersonType.JOURNALIST,    # television presenter
    "Q4263842": PersonType.JOURNALIST,   # columnist

    # Activists
    "Q15253558": PersonType.ACTIVIST,    # activist
    "Q11631410": PersonType.ACTIVIST,    # human rights activist
    "Q18939491": PersonType.ACTIVIST,    # environmental activist

    # Entrepreneurs/Executives via occupation
    "Q131524": PersonType.ENTREPRENEUR,  # entrepreneur
    "Q43845": PersonType.ENTREPRENEUR,   # businessperson
}

# =============================================================================
# ROLE/POSITION TYPE MAPPING (P31 - instance of)
# Entities with these P31 values are position/office items referenced by P39.
# =============================================================================

ROLE_TYPE_QIDS: frozenset[str] = frozenset({
    # Core position types
    "Q4164871",    # position
    "Q294414",     # public office
    "Q12737077",   # occupation

    # Political positions
    "Q2285706",    # type of political position
    "Q1553195",    # head of state
    "Q30185",      # head of government
    "Q15618781",   # legislative seat
    "Q740464",     # judicial office
    "Q4185145",    # civil service position
    "Q107711",     # minister
    "Q4175034",    # diplomat rank

    # Military positions
    "Q56019",      # military rank
    "Q4338774",    # military appointment

    # Academic / professional
    "Q15711026",   # academic position
    "Q193391",     # diplomatic rank
    "Q736600",     # religious rank

    # Corporate governance
    "Q1075651",    # type of management position
})

# =============================================================================
# ORGANIZATION TYPE MAPPING (P31 - instance of)
# =============================================================================

ORG_TYPE_TO_ENTITY_TYPE: dict[str, EntityType] = {
    # Business - core types
    "Q4830453": EntityType.BUSINESS,     # business
    "Q6881511": EntityType.BUSINESS,     # enterprise
    "Q783794": EntityType.BUSINESS,      # company
    "Q891723": EntityType.BUSINESS,      # public company
    "Q167037": EntityType.BUSINESS,      # corporation
    "Q658255": EntityType.BUSINESS,      # subsidiary
    "Q206652": EntityType.BUSINESS,      # conglomerate
    "Q22687": EntityType.BUSINESS,       # bank
    "Q1145276": EntityType.BUSINESS,     # insurance company
    "Q46970": EntityType.BUSINESS,       # airline
    "Q613142": EntityType.BUSINESS,      # law firm
    "Q507619": EntityType.BUSINESS,      # pharmaceutical company
    "Q2979960": EntityType.BUSINESS,     # technology company
    "Q1631111": EntityType.BUSINESS,     # retailer
    "Q187652": EntityType.BUSINESS,      # manufacturer
    # Business - additional types
    "Q43229": EntityType.BUSINESS,       # organization (generic)
    "Q4671277": EntityType.BUSINESS,     # academic institution (some are businesses)
    "Q1664720": EntityType.BUSINESS,     # institute
    "Q15911314": EntityType.BUSINESS,    # association
    "Q15925165": EntityType.BUSINESS,    # private company
    "Q5225895": EntityType.BUSINESS,     # credit union
    "Q161726": EntityType.BUSINESS,      # multinational corporation
    "Q134161": EntityType.BUSINESS,      # joint venture
    "Q1589009": EntityType.BUSINESS,     # privately held company
    "Q270791": EntityType.BUSINESS,      # state-owned enterprise
    "Q1762059": EntityType.BUSINESS,     # online service provider
    "Q17127659": EntityType.BUSINESS,    # energy company
    "Q2695280": EntityType.BUSINESS,     # construction company
    "Q1624464": EntityType.BUSINESS,     # telecommunications company
    "Q1668024": EntityType.BUSINESS,     # car manufacturer
    "Q3914": EntityType.BUSINESS,        # school (some are businesses)
    "Q1030034": EntityType.BUSINESS,     # management consulting firm
    "Q1370614": EntityType.BUSINESS,     # investment bank
    "Q1785271": EntityType.BUSINESS,     # advertising agency
    "Q4686042": EntityType.BUSINESS,     # automotive supplier
    "Q431289": EntityType.BUSINESS,      # brand
    "Q622438": EntityType.BUSINESS,      # supermarket chain
    "Q6500733": EntityType.BUSINESS,     # licensed retailer
    "Q2659904": EntityType.BUSINESS,     # government-owned corporation
    "Q1065118": EntityType.BUSINESS,     # bookmaker
    "Q179179": EntityType.BUSINESS,      # startup
    "Q210167": EntityType.BUSINESS,      # video game developer
    "Q18388277": EntityType.BUSINESS,    # video game publisher
    "Q1762913": EntityType.BUSINESS,     # film production company
    "Q18558478": EntityType.BUSINESS,    # money services business
    "Q6463968": EntityType.BUSINESS,     # asset management company
    "Q2864737": EntityType.BUSINESS,     # cooperative bank
    "Q161380": EntityType.BUSINESS,      # cooperative
    "Q15850590": EntityType.BUSINESS,    # real estate company
    "Q1048835": EntityType.BUSINESS,     # political organization
    "Q1254933": EntityType.BUSINESS,     # astronomical observatory (often research orgs)
    "Q294414": EntityType.BUSINESS,      # public office

    # Funds
    "Q45400320": EntityType.FUND,        # investment fund
    "Q476028": EntityType.FUND,          # hedge fund
    "Q380649": EntityType.FUND,          # investment company
    "Q1377053": EntityType.FUND,         # mutual fund
    "Q3312546": EntityType.FUND,         # private equity firm
    "Q751705": EntityType.FUND,          # venture capital firm
    "Q2296920": EntityType.FUND,         # sovereign wealth fund
    "Q2824951": EntityType.FUND,         # exchange-traded fund
    "Q1755098": EntityType.FUND,         # pension fund

    # Nonprofits
    "Q163740": EntityType.NONPROFIT,     # nonprofit organization
    "Q79913": EntityType.NGO,            # non-governmental organization
    "Q157031": EntityType.FOUNDATION,    # foundation
    "Q48204": EntityType.NONPROFIT,      # voluntary association
    "Q988108": EntityType.NONPROFIT,     # club
    "Q476436": EntityType.NONPROFIT,     # charitable organization
    "Q3591957": EntityType.NONPROFIT,    # cultural institution
    "Q162633": EntityType.NONPROFIT,     # academy
    "Q270791": EntityType.NONPROFIT,     # learned society
    "Q484652": EntityType.NONPROFIT,     # international organization

    # Government
    "Q327333": EntityType.GOVERNMENT,    # government agency
    "Q7278": EntityType.POLITICAL_PARTY, # political party
    "Q178790": EntityType.TRADE_UNION,   # trade union
    "Q7188": EntityType.GOVERNMENT,      # government
    "Q2659904": EntityType.GOVERNMENT,   # government-owned corporation
    "Q35798": EntityType.GOVERNMENT,     # executive branch
    "Q35749": EntityType.GOVERNMENT,     # legislature
    "Q12076836": EntityType.GOVERNMENT,  # law enforcement agency
    "Q17362920": EntityType.GOVERNMENT,  # public body
    "Q1063239": EntityType.GOVERNMENT,   # regulatory agency
    "Q3624078": EntityType.GOVERNMENT,   # sovereign state
    "Q133442": EntityType.GOVERNMENT,    # embassy
    "Q174834": EntityType.GOVERNMENT,    # authority (government)

    # International organizations
    "Q484652": EntityType.INTERNATIONAL_ORG,  # international organization
    "Q1335818": EntityType.INTERNATIONAL_ORG, # supranational organisation
    "Q1616075": EntityType.INTERNATIONAL_ORG, # intergovernmental organization

    # Education/Research
    "Q2385804": EntityType.EDUCATIONAL,  # educational institution
    "Q3918": EntityType.EDUCATIONAL,     # university
    "Q31855": EntityType.RESEARCH,       # research institute
    "Q875538": EntityType.EDUCATIONAL,   # public university
    "Q23002039": EntityType.EDUCATIONAL, # private university
    "Q38723": EntityType.EDUCATIONAL,    # higher education institution
    "Q1371037": EntityType.EDUCATIONAL,  # secondary school
    "Q9842": EntityType.EDUCATIONAL,     # primary school
    "Q189004": EntityType.EDUCATIONAL,   # college
    "Q1188663": EntityType.EDUCATIONAL,  # community college
    "Q1321960": EntityType.RESEARCH,     # think tank
    "Q31855": EntityType.RESEARCH,       # research institute
    "Q3354859": EntityType.RESEARCH,     # observatory
    "Q1298668": EntityType.RESEARCH,     # research center

    # Healthcare
    "Q16917": EntityType.HEALTHCARE,     # hospital
    "Q1774898": EntityType.HEALTHCARE,   # health care organization
    "Q180958": EntityType.HEALTHCARE,    # clinic
    "Q4260475": EntityType.HEALTHCARE,   # medical facility
    "Q871964": EntityType.HEALTHCARE,    # biotechnology company
    "Q902104": EntityType.HEALTHCARE,    # health insurance company

    # Sports
    "Q847017": EntityType.SPORTS,        # sports club
    "Q476068": EntityType.SPORTS,        # sports team
    "Q12973014": EntityType.SPORTS,      # sports organization
    "Q14350": EntityType.SPORTS,         # association football club
    "Q20639847": EntityType.SPORTS,      # American football team
    "Q13393265": EntityType.SPORTS,      # basketball team
    "Q13406463": EntityType.SPORTS,      # baseball team
    "Q1410877": EntityType.SPORTS,       # ice hockey team
    "Q18558301": EntityType.SPORTS,      # rugby union club
    "Q2093802": EntityType.SPORTS,       # cricket team
    "Q5137836": EntityType.SPORTS,       # motorsport racing team

    # Media
    "Q18127": EntityType.MEDIA,          # record label
    "Q1366047": EntityType.MEDIA,        # film studio
    "Q1137109": EntityType.MEDIA,        # video game company
    "Q11032": EntityType.MEDIA,          # newspaper
    "Q1002697": EntityType.MEDIA,        # periodical
    "Q5398426": EntityType.MEDIA,        # television series
    "Q1110794": EntityType.MEDIA,        # daily newspaper
    "Q1616075": EntityType.MEDIA,        # news agency
    "Q14350": EntityType.MEDIA,          # magazine
    "Q15265344": EntityType.MEDIA,       # broadcaster
    "Q131436": EntityType.MEDIA,         # radio station
    "Q1616075": EntityType.MEDIA,        # television station
    "Q41298": EntityType.MEDIA,          # magazine
    "Q30022": EntityType.MEDIA,          # television channel
    "Q17232649": EntityType.MEDIA,       # publishing company
    "Q28803812": EntityType.MEDIA,       # streaming service
    "Q159334": EntityType.MEDIA,         # entertainment company

    # Religious
    "Q9174": EntityType.RELIGIOUS,       # religion
    "Q1530022": EntityType.RELIGIOUS,    # religious organization
    "Q2994867": EntityType.RELIGIOUS,    # religious community
    "Q34651": EntityType.RELIGIOUS,      # church (building as org)
    "Q44613": EntityType.RELIGIOUS,      # monastery
}


# =============================================================================
# LOCATION TYPE MAPPING (P31 - instance of)
# Maps P31 QID -> (location_type_name, simplified_type)
# =============================================================================

LOCATION_TYPE_QIDS: dict[str, tuple[str, SimplifiedLocationType]] = {
    # ==========================================================================
    # IMPORTANT: The type names (first element of tuple) MUST match exactly
    # the names in database/seed_data.py LOCATION_TYPES. Any new types need
    # to be added there first, or use existing type names.
    # ==========================================================================

    # Continents (maps to: continent)
    "Q5107": ("continent", SimplifiedLocationType.CONTINENT),

    # Countries / Sovereign states (maps to: country, sovereign_state, dependent_territory)
    "Q6256": ("country", SimplifiedLocationType.COUNTRY),
    "Q3624078": ("sovereign_state", SimplifiedLocationType.COUNTRY),
    "Q161243": ("dependent_territory", SimplifiedLocationType.COUNTRY),
    # Additional country-like types -> map to country
    "Q15634554": ("country", SimplifiedLocationType.COUNTRY),  # state with limited recognition
    "Q1763527": ("country", SimplifiedLocationType.COUNTRY),   # constituent country
    "Q46395": ("dependent_territory", SimplifiedLocationType.COUNTRY),  # british overseas territory

    # Subdivisions (states/provinces) - US
    "Q35657": ("us_state", SimplifiedLocationType.SUBDIVISION),
    "Q47168": ("us_county", SimplifiedLocationType.SUBDIVISION),

    # Subdivisions - Country-specific
    "Q5852411": ("state_of_australia", SimplifiedLocationType.SUBDIVISION),
    "Q1221156": ("state_of_germany", SimplifiedLocationType.SUBDIVISION),
    "Q131541": ("state_of_india", SimplifiedLocationType.SUBDIVISION),
    "Q6465": ("department_france", SimplifiedLocationType.SUBDIVISION),
    "Q50337": ("prefecture_japan", SimplifiedLocationType.SUBDIVISION),
    "Q23058": ("canton_switzerland", SimplifiedLocationType.SUBDIVISION),
    "Q10742": ("autonomous_community_spain", SimplifiedLocationType.SUBDIVISION),
    "Q150093": ("voivodeship_poland", SimplifiedLocationType.SUBDIVISION),
    "Q835714": ("oblast_russia", SimplifiedLocationType.SUBDIVISION),

    # Subdivisions - Generic (map to existing types)
    "Q34876": ("province", SimplifiedLocationType.SUBDIVISION),
    "Q82794": ("region", SimplifiedLocationType.SUBDIVISION),
    "Q28575": ("county", SimplifiedLocationType.SUBDIVISION),
    # Additional generic subdivision types -> map to region/province/county
    "Q10864048": ("region", SimplifiedLocationType.SUBDIVISION),    # first-level admin
    "Q11828004": ("county", SimplifiedLocationType.SUBDIVISION),    # second-level admin
    "Q12483": ("region", SimplifiedLocationType.SUBDIVISION),       # territory
    "Q515716": ("region", SimplifiedLocationType.SUBDIVISION),      # region of Italy
    "Q1132541": ("county", SimplifiedLocationType.SUBDIVISION),     # county of Sweden
    "Q1780990": ("region", SimplifiedLocationType.SUBDIVISION),     # council area Scotland
    "Q211690": ("county", SimplifiedLocationType.SUBDIVISION),      # ceremonial county England
    "Q180673": ("county", SimplifiedLocationType.SUBDIVISION),      # ceremonial county
    "Q1136601": ("county", SimplifiedLocationType.SUBDIVISION),     # metropolitan county
    "Q21451686": ("region", SimplifiedLocationType.SUBDIVISION),    # region of England
    "Q1006876": ("region", SimplifiedLocationType.SUBDIVISION),     # unitary authority Wales
    "Q179872": ("province", SimplifiedLocationType.SUBDIVISION),    # province of Canada
    "Q1352230": ("region", SimplifiedLocationType.SUBDIVISION),     # territory of Canada
    "Q13360155": ("province", SimplifiedLocationType.SUBDIVISION),  # province of China
    "Q842112": ("region", SimplifiedLocationType.SUBDIVISION),      # autonomous region China
    "Q1348006": ("municipality", SimplifiedLocationType.CITY),      # municipality of China (city-level)
    "Q11774097": ("city", SimplifiedLocationType.CITY),             # prefecture-level city

    # Cities/Towns/Municipalities (maps to: city, big_city, capital, town, municipality, village, hamlet)
    "Q515": ("city", SimplifiedLocationType.CITY),
    "Q1549591": ("big_city", SimplifiedLocationType.CITY),
    "Q5119": ("capital", SimplifiedLocationType.CITY),
    "Q3957": ("town", SimplifiedLocationType.CITY),
    "Q15284": ("municipality", SimplifiedLocationType.CITY),
    "Q532": ("village", SimplifiedLocationType.CITY),
    "Q5084": ("hamlet", SimplifiedLocationType.CITY),
    # Country-specific municipalities
    "Q484170": ("commune_france", SimplifiedLocationType.CITY),
    "Q262166": ("municipality_germany", SimplifiedLocationType.CITY),
    "Q1054813": ("municipality_japan", SimplifiedLocationType.CITY),
    # Additional city types -> map to city/town/village
    "Q7930989": ("city", SimplifiedLocationType.CITY),         # city of US
    "Q200250": ("big_city", SimplifiedLocationType.CITY),      # metropolis
    "Q2264924": ("big_city", SimplifiedLocationType.CITY),     # conurbation
    "Q174844": ("big_city", SimplifiedLocationType.CITY),      # megacity
    "Q22865": ("city", SimplifiedLocationType.CITY),           # independent city
    "Q5153359": ("municipality", SimplifiedLocationType.CITY), # commune (generic)
    "Q4286337": ("village", SimplifiedLocationType.CITY),      # locality
    "Q486972": ("village", SimplifiedLocationType.CITY),       # human settlement
    "Q95993392": ("city", SimplifiedLocationType.CITY),        # city or town

    # Districts (maps to: district, borough, neighborhood, ward)
    "Q149621": ("district", SimplifiedLocationType.DISTRICT),
    "Q5765681": ("borough", SimplifiedLocationType.DISTRICT),
    "Q123705": ("neighborhood", SimplifiedLocationType.DISTRICT),
    "Q12813115": ("ward", SimplifiedLocationType.DISTRICT),
    # Additional district types -> map to district/borough
    "Q2198484": ("borough", SimplifiedLocationType.DISTRICT),  # borough of London
    "Q667509": ("district", SimplifiedLocationType.DISTRICT),  # arrondissement
    "Q2100709": ("district", SimplifiedLocationType.DISTRICT), # city district

    # Historic (maps to: former_country, ancient_civilization, historic_territory)
    "Q3024240": ("former_country", SimplifiedLocationType.HISTORIC),
    "Q28171280": ("ancient_civilization", SimplifiedLocationType.HISTORIC),
    "Q1620908": ("historic_territory", SimplifiedLocationType.HISTORIC),
    # Additional historic types
    "Q19953632": ("historic_territory", SimplifiedLocationType.HISTORIC),  # historical region
    "Q1307214": ("historic_territory", SimplifiedLocationType.HISTORIC),   # historical admin region
}


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

DEFAULT_PROGRESS_PATH = Path.home() / ".cache" / "corp-extractor" / "wikidata-dump-progress.json"


@dataclass
class DumpProgress:
    """
    Tracks progress through the Wikidata dump file for resume support.

    Progress is tracked by entity index (number of entities processed).
    On resume, entities are skipped until reaching the saved position.
    """
    # Entity index - number of entities yielded from the dump
    entity_index: int = 0

    # Separate counters for people and orgs import
    people_yielded: int = 0
    orgs_yielded: int = 0

    # Last entity ID processed (for verification)
    last_entity_id: str = ""

    # Timestamp of last update
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Dump file path (to detect if dump changed)
    dump_path: str = ""

    # Dump file size (to detect if dump changed)
    dump_size: int = 0

    def save(self, path: Optional[Path] = None) -> None:
        """Save progress to JSON file."""
        path = path or DEFAULT_PROGRESS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump({
                "entity_index": self.entity_index,
                "people_yielded": self.people_yielded,
                "orgs_yielded": self.orgs_yielded,
                "last_entity_id": self.last_entity_id,
                "last_updated": self.last_updated,
                "dump_path": self.dump_path,
                "dump_size": self.dump_size,
            }, f, indent=2)
        logger.debug(f"Saved progress: entity_index={self.entity_index}, last_id={self.last_entity_id}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["DumpProgress"]:
        """Load progress from JSON file, returns None if not found."""
        path = path or DEFAULT_PROGRESS_PATH
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                entity_index=data.get("entity_index", 0),
                people_yielded=data.get("people_yielded", 0),
                orgs_yielded=data.get("orgs_yielded", 0),
                last_entity_id=data.get("last_entity_id", ""),
                last_updated=data.get("last_updated", ""),
                dump_path=data.get("dump_path", ""),
                dump_size=data.get("dump_size", 0),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load progress from {path}: {e}")
            return None

    @classmethod
    def clear(cls, path: Optional[Path] = None) -> None:
        """Delete the progress file."""
        path = path or DEFAULT_PROGRESS_PATH
        if path.exists():
            path.unlink()
            logger.info(f"Cleared progress file: {path}")

    def matches_dump(self, dump_path: Path) -> bool:
        """Check if this progress matches the given dump file."""
        if str(dump_path) != self.dump_path:
            return False
        if dump_path.exists() and dump_path.stat().st_size != self.dump_size:
            return False
        return True


class WikidataDumpImporter:
    """
    Stream Wikidata JSON dump to extract people and organization records.

    This importer processes the Wikidata dump line-by-line to avoid memory issues
    with the ~100GB compressed file. It filters for:
    - Humans (P31=Q5) with English Wikipedia articles
    - Organizations with English Wikipedia articles

    The dump URL can be customized, and the importer supports both .bz2 and .gz
    compression formats.
    """

    def __init__(self, dump_path: Optional[str] = None):
        """
        Initialize the dump importer.

        Args:
            dump_path: Optional path to a pre-downloaded dump file.
                      If not provided, will need to call download_dump() first.
        """
        self._dump_path = Path(dump_path) if dump_path else None
        # Track discovered organizations from people import
        self._discovered_orgs: dict[str, str] = {}
        # Reverse lookup: person_qid → [(org_qid, role, start_date, end_date), ...] from org executive properties
        # Built during org processing, used to backfill people missing known_for_org
        self._reverse_person_orgs: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]] = {}
        # Cache position item → jurisdiction QID (e.g. Q11696 "President of the United States" → Q30 "United States")
        # Built from all entities with P31=Q4164871 (position) that have P1001 or P17.
        # Used to backfill org context for P39 claims without qualifier-level org.
        self._position_jurisdictions: dict[str, str] = {}
        # Cache role QID → English label. Built as role entities are encountered in the dump.
        # Used to populate known_for_role on person records during pass 1.
        self._role_labels: dict[str, str] = {}

    def download_dump(
        self,
        target_dir: Optional[Path] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_aria2: bool = True,
        aria2_connections: int = 16,
    ) -> Path:
        """
        Download the latest Wikidata dump with progress indicator.

        For fastest downloads, uses aria2c if available (16 parallel connections).
        Falls back to urllib if aria2c is not installed.

        Args:
            target_dir: Directory to save the dump (default: ~/.cache/corp-extractor)
            force: Force re-download even if file exists
            progress_callback: Optional callback(downloaded_bytes, total_bytes) for progress
            use_aria2: Try to use aria2c for faster downloads (default: True)
            aria2_connections: Number of connections for aria2c (default: 16)

        Returns:
            Path to the downloaded dump file
        """
        if target_dir is None:
            target_dir = Path.home() / ".cache" / "corp-extractor"

        target_dir.mkdir(parents=True, exist_ok=True)
        dump_path = target_dir / "wikidata-latest-all.json.bz2"

        if dump_path.exists() and not force:
            logger.info(f"Using cached dump at {dump_path}")
            self._dump_path = dump_path
            return dump_path

        logger.info(f"Target: {dump_path}")

        # Try aria2c first for much faster downloads
        if use_aria2 and shutil.which("aria2c"):
            logger.info("Using aria2c for fast parallel download...")
            try:
                self._download_with_aria2(dump_path, connections=aria2_connections)
                self._dump_path = dump_path
                return dump_path
            except Exception as e:
                logger.warning(f"aria2c download failed: {e}, falling back to urllib")

        # Fallback to urllib
        logger.info(f"Downloading Wikidata dump from {DUMP_URL}...")
        logger.info("TIP: Install aria2c for 10-20x faster downloads: brew install aria2")
        logger.info("This is a large file (~100GB) and will take significant time.")

        # Stream download with progress
        req = urllib.request.Request(
            DUMP_URL,
            headers={"User-Agent": "corp-extractor/1.0 (Wikidata dump importer)"}
        )

        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("content-length", 0))
            total_gb = total / (1024 ** 3) if total else 0

            with open(dump_path, "wb") as f:
                downloaded = 0
                chunk_size = 8 * 1024 * 1024  # 8MB chunks
                last_log_pct = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(downloaded, total)
                    else:
                        # Default logging (every 1%)
                        if total:
                            pct = int((downloaded / total) * 100)
                            if pct > last_log_pct:
                                downloaded_gb = downloaded / (1024 ** 3)
                                logger.info(f"Downloaded {downloaded_gb:.1f}GB / {total_gb:.1f}GB ({pct}%)")
                                last_log_pct = pct
                        elif downloaded % (1024 ** 3) < chunk_size:
                            # Log every GB if total unknown
                            downloaded_gb = downloaded / (1024 ** 3)
                            logger.info(f"Downloaded {downloaded_gb:.1f}GB")

        logger.info(f"Download complete: {dump_path}")
        self._dump_path = dump_path
        return dump_path

    def _download_with_aria2(
        self,
        output_path: Path,
        connections: int = 16,
    ) -> None:
        """
        Download using aria2c with multiple parallel connections.

        aria2c can achieve 10-20x faster downloads by using multiple
        connections to the server.

        Args:
            output_path: Where to save the downloaded file
            connections: Number of parallel connections (default: 16)
        """
        cmd = [
            "aria2c",
            "-x", str(connections),  # Max connections per server
            "-s", str(connections),  # Split file into N parts
            "-k", "10M",  # Min split size
            "--file-allocation=none",  # Faster on SSDs
            "-d", str(output_path.parent),
            "-o", output_path.name,
            "--console-log-level=notice",
            "--summary-interval=10",
            DUMP_URL,
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        # Run aria2c and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output to logger
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"aria2c: {line}")

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"aria2c exited with code {return_code}")

    def get_dump_path(self, target_dir: Optional[Path] = None) -> Path:
        """
        Get the path where the dump would be/is downloaded.

        Args:
            target_dir: Directory for the dump (default: ~/.cache/corp-extractor)

        Returns:
            Path to the dump file location
        """
        if target_dir is None:
            target_dir = Path.home() / ".cache" / "corp-extractor"
        return target_dir / "wikidata-latest-all.json.bz2"

    def iter_entities(
        self,
        dump_path: Optional[Path] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Iterator[dict]:
        """
        Stream entities from dump file, one at a time.

        Handles the Wikidata JSON dump format where each line after the opening
        bracket is a JSON object with a trailing comma (except the last).

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            start_index: Entity index to start yielding from (default 0). Entities
                        before this index are skipped but still cached for label lookups.
            progress_callback: Optional callback(entity_index, entity_id) called for each
                              yielded entity. Useful for tracking progress.

        Yields:
            Parsed entity dictionaries
        """
        path = dump_path or self._dump_path
        if path is None:
            raise ValueError("No dump path provided. Call download_dump() first or pass dump_path.")

        path = Path(path)

        logger.info(f"Opening dump file: {path}")
        logger.info(f"File size: {path.stat().st_size / (1024**3):.1f} GB")
        if _HAVE_ORJSON:
            logger.info("Using orjson for fast JSON parsing")
        if start_index > 0:
            logger.info(f"Resuming from entity index {start_index:,} (skipping earlier entities)")

        # Select opener — prefer indexed_bzip2 for parallel decompression
        file_handle: Any = None
        try:
            if path.suffix == ".bz2" and _HAVE_IBZ2:
                import os
                ncpu = os.cpu_count() or 1
                logger.info(f"Using indexed_bzip2 with {ncpu} cores for parallel decompression")
                raw = _ibz2.open(str(path), parallelization=ncpu)
                file_handle = io.TextIOWrapper(raw, encoding="utf-8")
            elif path.suffix == ".bz2":
                logger.info("Using stdlib bz2 (install indexed_bzip2 for ~6x faster decompression)")
                file_handle = bz2.open(path, "rt", encoding="utf-8")
            elif path.suffix in (".zst", ".zstd"):
                try:
                    import zstandard as zstd
                    logger.info("Using zstandard decompression")
                    fh = open(path, "rb")
                    dctx = zstd.ZstdDecompressor()
                    reader = dctx.stream_reader(fh)
                    file_handle = io.TextIOWrapper(reader, encoding="utf-8")
                except ImportError:
                    raise ImportError("zstandard package required for .zst files: pip install zstandard")
            elif path.suffix == ".gz":
                file_handle = gzip.open(path, "rt", encoding="utf-8")
            else:
                file_handle = open(path, "r", encoding="utf-8")

            logger.info("Dump file opened, reading lines...")
            line_count = 0
            entity_count = 0
            skipped_count = 0
            next_log_threshold = 10_000

            # === Fast skip phase: no JSON parsing, just count entity lines ===
            # Labels from the previous run are already in the DB and loaded into
            # the label cache at startup. Parsing JSON here was the #1 memory
            # killer — each entity dict is large and the label cache grew to
            # 100M+ entries.
            if start_index > 0:
                logger.info(f"Fast-skipping {start_index:,} entities (no JSON parsing)...")
                for line in file_handle:
                    line_count += 1
                    stripped = line.strip()
                    if not stripped or stripped in ("[", "]"):
                        continue

                    entity_count += 1
                    skipped_count += 1

                    if skipped_count >= next_log_threshold:
                        pct = 100 * skipped_count / start_index
                        logger.info(
                            f"Skipping... {skipped_count:,}/{start_index:,} entities ({pct:.1f}%)"
                        )
                        if next_log_threshold < 100_000:
                            next_log_threshold = 100_000
                        elif next_log_threshold < 1_000_000:
                            next_log_threshold = 1_000_000
                        else:
                            next_log_threshold += 1_000_000

                    if entity_count >= start_index:
                        break

                logger.info(f"Skip complete: {skipped_count:,} entities skipped, resuming processing")
                next_log_threshold = entity_count + 100_000

            # === Processing phase: parse JSON and yield entities ===
            for line in file_handle:
                line_count += 1

                if line_count <= 5 and start_index == 0:
                    logger.info(f"Read line {line_count} ({len(line)} chars)")

                line = line.strip()

                if line in ("[", "]"):
                    continue

                if line.endswith(","):
                    line = line[:-1]

                if not line:
                    continue

                try:
                    entity = _json_loads(line)
                    entity_id = entity.get("id", "")

                    # Cache jurisdiction for position items (cheap — just check for P1001/P17)
                    self._cache_position_jurisdiction(entity)
                    entity_count += 1

                    if entity_count >= next_log_threshold:
                        logger.info(f"Processed {entity_count:,} entities")
                        if next_log_threshold < 100_000:
                            next_log_threshold = 100_000
                        elif next_log_threshold < 1_000_000:
                            next_log_threshold = 1_000_000
                        else:
                            next_log_threshold += 1_000_000

                    if progress_callback:
                        progress_callback(entity_count, entity_id)

                    yield entity

                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Line {line_count}: JSON decode error: {e}")
        finally:
            if file_handle is not None:
                file_handle.close()

    def import_people(
        self,
        dump_path: Optional[Path] = None,
        limit: Optional[int] = None,
        require_enwiki: bool = False,
        skip_ids: Optional[set[str]] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> Iterator[PersonRecord]:
        """
        Stream through dump, yielding ALL people (humans with P31=Q5).

        This method filters the dump for:
        - Items with type "item" (not properties)
        - Humans (P31 contains Q5)
        - Optionally: Has English Wikipedia article (enwiki sitelink)

        PersonType is derived from positions (P39) and occupations (P106).
        Parliamentary context (electoral district, term, party) is extracted from P39 qualifiers.

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            limit: Optional maximum number of records to return
            require_enwiki: If True, only include people with English Wikipedia articles
            skip_ids: Optional set of source_ids (Q codes) to skip. Checked early before
                     full processing to avoid unnecessary QID resolution.
            start_index: Entity index to start from (for resume support). Entities
                        before this index are skipped but labels are still cached.
            progress_callback: Optional callback(entity_index, entity_id, records_yielded)
                              called for each yielded record. Useful for saving progress.

        Yields:
            PersonRecord for each qualifying person
        """
        path = dump_path or self._dump_path
        count = 0
        skipped = 0
        current_entity_index = start_index

        logger.info("Starting people import from Wikidata dump...")
        if start_index > 0:
            logger.info(f"Resuming from entity index {start_index:,}")
        if not require_enwiki:
            logger.info("Importing ALL humans (no enwiki filter)")
        if skip_ids:
            logger.info(f"Skipping {len(skip_ids):,} existing Q codes")

        def track_entity(entity_index: int, entity_id: str) -> None:
            nonlocal current_entity_index
            current_entity_index = entity_index

        for entity in self.iter_entities(path, start_index=start_index, progress_callback=track_entity):
            if limit and count >= limit:
                break

            # Check skip_ids early, before full processing (avoids QID resolution)
            entity_id = entity.get("id", "")
            if skip_ids and entity_id in skip_ids:
                skipped += 1
                continue

            person_records = self._process_person_entity(entity, require_enwiki=require_enwiki)
            for record in person_records:
                count += 1
                if count % 10_000 == 0:
                    logger.info(f"Yielded {count:,} people records (skipped {skipped:,})...")

                # Call progress callback with current position
                if progress_callback:
                    progress_callback(current_entity_index, entity_id, count)

                yield record
                if limit and count >= limit:
                    break

        logger.info(f"People import complete: {count:,} records (skipped {skipped:,})")

    def import_organizations(
        self,
        dump_path: Optional[Path] = None,
        limit: Optional[int] = None,
        require_enwiki: bool = False,
        skip_ids: Optional[set[str]] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> Iterator[CompanyRecord]:
        """
        Stream through dump, yielding organizations.

        This method filters the dump for:
        - Items with type "item"
        - Has P31 (instance of) matching an organization type
        - Optionally: Has English Wikipedia article (enwiki sitelink)

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            limit: Optional maximum number of records to return
            require_enwiki: If True, only include orgs with English Wikipedia articles
            skip_ids: Optional set of source_ids (Q codes) to skip. Checked early before
                     full processing to avoid unnecessary QID resolution.
            start_index: Entity index to start from (for resume support). Entities
                        before this index are skipped but labels are still cached.
            progress_callback: Optional callback(entity_index, entity_id, records_yielded)
                              called for each yielded record. Useful for saving progress.

        Yields:
            CompanyRecord for each qualifying organization
        """
        path = dump_path or self._dump_path
        count = 0
        skipped_existing = 0
        skipped_no_type = 0
        skipped_no_enwiki = 0
        skipped_no_label = 0
        current_entity_index = start_index

        logger.info("Starting organization import from Wikidata dump...")
        if start_index > 0:
            logger.info(f"Resuming from entity index {start_index:,}")
        if not require_enwiki:
            logger.info("Importing ALL organizations (no enwiki filter)")
        if skip_ids:
            logger.info(f"Skipping {len(skip_ids):,} existing Q codes")

        def track_entity(entity_index: int, entity_id: str) -> None:
            nonlocal current_entity_index
            current_entity_index = entity_index

        for entity in self.iter_entities(path, start_index=start_index, progress_callback=track_entity):
            if limit and count >= limit:
                break

            # Check skip_ids early, before full processing (avoids QID resolution)
            entity_id = entity.get("id", "")
            if skip_ids and entity_id in skip_ids:
                skipped_existing += 1
                continue

            record = self._process_org_entity(entity, require_enwiki=require_enwiki)
            if record:
                count += 1
                if count % 10_000 == 0:
                    logger.info(f"Yielded {count:,} organization records (skipped {skipped_existing:,} existing)...")

                # Call progress callback with current position
                if progress_callback:
                    progress_callback(current_entity_index, entity_id, count)

                yield record
            elif entity.get("type") == "item":
                # Track skip reasons for debugging
                if self._get_org_type(entity) is None:
                    skipped_no_type += 1
                elif require_enwiki and "enwiki" not in entity.get("sitelinks", {}):
                    skipped_no_enwiki += 1
                else:
                    skipped_no_label += 1

                # Log skip stats periodically
                total_skipped = skipped_no_type + skipped_no_enwiki + skipped_no_label
                if total_skipped > 0 and total_skipped % 1_000_000 == 0:
                    logger.debug(
                        f"Skip stats: no_matching_type={skipped_no_type:,}, "
                        f"no_enwiki={skipped_no_enwiki:,}, no_label={skipped_no_label:,}"
                    )

        logger.info(f"Organization import complete: {count:,} records (skipped {skipped_existing:,} existing)")
        logger.info(
            f"Skipped: no_matching_type={skipped_no_type:,}, "
            f"no_enwiki={skipped_no_enwiki:,}, no_label={skipped_no_label:,}"
        )

    def import_all(
        self,
        dump_path: Optional[Path] = None,
        people_limit: Optional[int] = None,
        orgs_limit: Optional[int] = None,
        locations_limit: Optional[int] = None,
        roles_limit: Optional[int] = None,
        import_people: bool = True,
        import_orgs: bool = True,
        import_locations: bool = True,
        import_roles: bool = True,
        require_enwiki: bool = False,
        skip_people_ids: Optional[set[str]] = None,
        skip_org_ids: Optional[set[str]] = None,
        skip_location_ids: Optional[set[str]] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str, int, int], None]] = None,
    ) -> Iterator[tuple[str, ImportRecord]]:
        """
        Import people, organizations, locations, and roles in a single pass through the dump.

        This is more efficient than calling import methods separately,
        as it only reads the ~100GB dump file once.

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            people_limit: Optional maximum number of people records
            orgs_limit: Optional maximum number of org records
            locations_limit: Optional maximum number of location records
            roles_limit: Optional maximum number of role records
            import_people: Whether to import people (default: True)
            import_orgs: Whether to import organizations (default: True)
            import_locations: Whether to import locations (default: True)
            import_roles: Whether to import roles (default: True)
            require_enwiki: If True, only include entities with English Wikipedia articles
            skip_people_ids: Optional set of people source_ids (Q codes) to skip
            skip_org_ids: Optional set of org source_ids (Q codes) to skip
            skip_location_ids: Optional set of location source_ids (Q codes) to skip
            start_index: Entity index to start from (for resume support)
            progress_callback: Optional callback(entity_index, entity_id, people_count, orgs_count)
                              called periodically. Useful for saving progress.

        Yields:
            Tuples of (record_type, record) where record_type is "person", "org", or "location"
        """
        path = dump_path or self._dump_path
        people_count = 0
        orgs_count = 0
        locations_count = 0
        roles_count = 0
        people_skipped = 0
        orgs_skipped = 0
        locations_skipped = 0
        current_entity_index = start_index

        logger.info("Starting combined import from Wikidata dump...")
        if start_index > 0:
            logger.info(f"Resuming from entity index {start_index:,}")
        if import_people:
            logger.info(f"Importing people (limit: {people_limit or 'none'})")
            if skip_people_ids:
                logger.info(f"  Skipping {len(skip_people_ids):,} existing people Q codes")
        if import_orgs:
            logger.info(f"Importing organizations (limit: {orgs_limit or 'none'})")
            if skip_org_ids:
                logger.info(f"  Skipping {len(skip_org_ids):,} existing org Q codes")
        if import_locations:
            logger.info(f"Importing locations (limit: {locations_limit or 'none'})")
            if skip_location_ids:
                logger.info(f"  Skipping {len(skip_location_ids):,} existing location Q codes")

        # Check if we've hit all limits
        def limits_reached() -> bool:
            people_done = not import_people or (people_limit and people_count >= people_limit)
            orgs_done = not import_orgs or (orgs_limit and orgs_count >= orgs_limit)
            locations_done = not import_locations or (locations_limit and locations_count >= locations_limit)
            roles_done = not import_roles or (roles_limit and roles_count >= roles_limit)
            return bool(people_done and orgs_done and locations_done and roles_done)

        def track_entity(entity_index: int, entity_id: str) -> None:
            nonlocal current_entity_index
            current_entity_index = entity_index

        for entity in self.iter_entities(path, start_index=start_index, progress_callback=track_entity):
            if limits_reached():
                break

            entity_id = entity.get("id", "")

            # Check for role/position entities to cache labels and yield records.
            # Roles are detected first since they tend to have lower QIDs in the dump,
            # so their labels are cached before the people who reference them.
            if import_roles and (not roles_limit or roles_count < roles_limit):
                role_record = self._process_role_entity(entity)
                if role_record:
                    roles_count += 1
                    yield ("role", role_record)
                    if roles_count % 10_000 == 0:
                        logger.info(f"Progress: {roles_count:,} roles cached (entity {current_entity_index:,})")
                    continue  # Role entities aren't people/orgs/locations

            # Try to process as person first (if importing people and not at limit)
            if import_people and (not people_limit or people_count < people_limit):
                # Check skip_ids early
                if skip_people_ids and entity_id in skip_people_ids:
                    people_skipped += 1
                else:
                    person_records = self._process_person_entity(entity, require_enwiki=require_enwiki)
                    if person_records:
                        for person_record in person_records:
                            people_count += 1
                            yield ("person", person_record)
                            if people_limit and people_count >= people_limit:
                                break
                        if people_count % 10_000 == 0:
                            logger.info(
                                f"Progress: {people_count:,} people, {orgs_count:,} orgs, "
                                f"{locations_count:,} locations (entity {current_entity_index:,})"
                            )
                        if progress_callback:
                            progress_callback(current_entity_index, entity_id, people_count, orgs_count)
                        continue  # Entity was a person, don't check for org/location

            # Try to process as organization (if importing orgs and not at limit)
            if import_orgs and (not orgs_limit or orgs_count < orgs_limit):
                # Check skip_ids early
                if skip_org_ids and entity_id in skip_org_ids:
                    orgs_skipped += 1
                else:
                    org_record = self._process_org_entity(entity, require_enwiki=require_enwiki)
                    if org_record:
                        orgs_count += 1
                        if orgs_count % 10_000 == 0:
                            logger.info(
                                f"Progress: {people_count:,} people, {orgs_count:,} orgs, "
                                f"{locations_count:,} locations (entity {current_entity_index:,})"
                            )
                        if progress_callback:
                            progress_callback(current_entity_index, entity_id, people_count, orgs_count)
                        yield ("org", org_record)
                        # Also check if entity is a location (countries are both orgs and locations)
                        if import_locations and (not locations_limit or locations_count < locations_limit):
                            if not (skip_location_ids and entity_id in skip_location_ids):
                                location_record = self._process_location_entity(entity, require_enwiki=require_enwiki)
                                if location_record:
                                    locations_count += 1
                                    yield ("location", location_record)
                        continue

            # Try to process as location (if importing locations and not at limit)
            if import_locations and (not locations_limit or locations_count < locations_limit):
                if skip_location_ids and entity_id in skip_location_ids:
                    locations_skipped += 1
                else:
                    location_record = self._process_location_entity(entity, require_enwiki=require_enwiki)
                    if location_record:
                        locations_count += 1
                        if locations_count % 10_000 == 0:
                            logger.info(
                                f"Progress: {people_count:,} people, {orgs_count:,} orgs, "
                                f"{locations_count:,} locations (entity {current_entity_index:,})"
                            )
                        if progress_callback:
                            progress_callback(current_entity_index, entity_id, people_count, orgs_count)
                        yield ("location", location_record)

        logger.info(
            f"Combined import complete: {people_count:,} people, {orgs_count:,} orgs, "
            f"{locations_count:,} locations, {roles_count:,} roles "
            f"(skipped {people_skipped:,} people, {orgs_skipped:,} orgs, {locations_skipped:,} locations)"
        )

    def _process_person_entity(
        self,
        entity: dict,
        require_enwiki: bool = False,
    ) -> list[PersonRecord]:
        """
        Process a single entity, return PersonRecord(s) if it's a human.

        Returns multiple records when a person has multiple positions with
        different orgs (e.g. CEO of company A and board member of company B).

        Args:
            entity: Parsed Wikidata entity dictionary
            require_enwiki: If True, only include people with English Wikipedia articles

        Returns:
            List of PersonRecords (empty if entity doesn't qualify)
        """
        # Must be an item (not property)
        if entity.get("type") != "item":
            return []

        # Must be human (P31 contains Q5)
        if not self._is_human(entity):
            return []

        # Optionally require English Wikipedia article
        if require_enwiki:
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" not in sitelinks:
                return []

        # Extract person records (one per position with org)
        return self._extract_person_records(entity)

    def _process_org_entity(
        self,
        entity: dict,
        require_enwiki: bool = False,
    ) -> Optional[CompanyRecord]:
        """
        Process a single entity, return CompanyRecord if it's an organization.

        Args:
            entity: Parsed Wikidata entity dictionary
            require_enwiki: If True, only include orgs with English Wikipedia articles

        Returns:
            CompanyRecord if entity qualifies, None otherwise
        """
        # Must be an item (not property)
        if entity.get("type") != "item":
            return None

        # Get organization type from P31
        entity_type = self._get_org_type(entity)
        if entity_type is None:
            return None

        # Optionally require English Wikipedia article
        if require_enwiki:
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" not in sitelinks:
                return None

        # Extract organization data
        return self._extract_org_data(entity, entity_type)

    def _is_human(self, entity: dict) -> bool:
        """
        Check if entity has P31 (instance of) = Q5 (human).

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            True if entity is a human
        """
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict) and value.get("id") == "Q5":
                return True
        return False

    def _get_org_type(self, entity: dict) -> Optional[EntityType]:
        """
        Check if entity has P31 (instance of) matching an organization type.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            EntityType if entity is an organization, None otherwise
        """
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id", "")
                if qid in ORG_TYPE_TO_ENTITY_TYPE:
                    return ORG_TYPE_TO_ENTITY_TYPE[qid]
        return None

    def _get_location_type(self, entity: dict) -> Optional[tuple[str, SimplifiedLocationType]]:
        """
        Check if entity has P31 (instance of) matching a location type.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            Tuple of (location_type_name, SimplifiedLocationType) if entity is a location, None otherwise
        """
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id", "")
                if qid in LOCATION_TYPE_QIDS:
                    return LOCATION_TYPE_QIDS[qid]
        return None

    def _is_role_entity(self, entity: dict) -> bool:
        """Check if entity has P31 matching a role/position type."""
        claims = entity.get("claims", {})
        for claim in claims.get("P31", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id", "")
                if qid in ROLE_TYPE_QIDS:
                    return True
        return False

    def _process_role_entity(self, entity: dict) -> Optional[RoleRecord]:
        """Process an entity as a role/position, return RoleRecord if it qualifies.

        Only includes entities with English labels.
        """
        if entity.get("type") != "item":
            return None

        if not self._is_role_entity(entity):
            return None

        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value", "")
        if not label or not qid:
            return None

        # Cache the label for use in person processing
        self._role_labels[qid] = label

        qid_int = int(qid[1:]) if qid.startswith("Q") and qid[1:].isdigit() else None

        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        return RoleRecord(
            name=label,
            source="wikidata",
            source_id=qid,
            qid=qid_int,
            record={
                "wikidata_id": qid,
                "label": label,
                "description": description,
            },
        )

    def _get_claim_values(self, entity: dict, prop: str) -> list[str]:
        """
        Get all QID values for a property (e.g., P39, P106).

        Args:
            entity: Parsed Wikidata entity dictionary
            prop: Property ID (e.g., "P39", "P106")

        Returns:
            List of QID strings
        """
        claims = entity.get("claims", {})
        values = []
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                qid = value.get("id")
                if qid:
                    values.append(qid)
        return values

    def _get_qid_qualifier(self, qualifiers: dict, prop: str) -> Optional[str]:
        """Extract first QID from a qualifier property."""
        for qual in qualifiers.get(prop, []):
            qual_datavalue = qual.get("datavalue", {})
            qual_value = qual_datavalue.get("value", {})
            if isinstance(qual_value, dict):
                return qual_value.get("id")
        return None

    def _get_time_qualifier(self, qualifiers: dict, prop: str) -> Optional[str]:
        """Extract first time value from a qualifier property."""
        for qual in qualifiers.get(prop, []):
            qual_datavalue = qual.get("datavalue", {})
            qual_value = qual_datavalue.get("value", {})
            if isinstance(qual_value, dict):
                time_str = qual_value.get("time", "")
                return self._parse_time_value(time_str)
        return None

    def _get_positions_with_org(self, claims: dict) -> list[dict]:
        """
        Extract P39 positions with qualifiers for org, dates, and parliamentary context.

        Qualifiers extracted per WikiProject Parliaments guidelines:
        - P580 (start time) - when the position started
        - P582 (end time) - when the position ended
        - P108 (employer) - organization they work for
        - P642 (of) - the organization (legacy/fallback)
        - P768 (electoral district) - constituency for MPs
        - P2937 (parliamentary term) - which term they served in
        - P4100 (parliamentary group) - political party/faction
        - P1001 (applies to jurisdiction) - jurisdiction they represent
        - P2715 (elected in) - which election elected them

        Args:
            claims: Claims dictionary from entity

        Returns:
            List of position dictionaries with position metadata
        """
        positions = []
        for claim in claims.get("P39", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            pos_value = datavalue.get("value", {})
            pos_qid = pos_value.get("id") if isinstance(pos_value, dict) else None
            if not pos_qid:
                continue

            qualifiers = claim.get("qualifiers", {})

            # Extract organization from multiple possible qualifiers
            # Priority: P108 (employer) > P642 (of) > P1001 (jurisdiction) > position item's P1001/P17
            org_qid = (
                self._get_qid_qualifier(qualifiers, "P108") or  # employer
                self._get_qid_qualifier(qualifiers, "P642") or  # of (legacy)
                self._get_qid_qualifier(qualifiers, "P1001") or  # applies to jurisdiction
                self._position_jurisdictions.get(pos_qid, "")   # position item's own jurisdiction
            )

            # Extract dates
            start_date = self._get_time_qualifier(qualifiers, "P580")
            end_date = self._get_time_qualifier(qualifiers, "P582")

            # Extract parliamentary/political qualifiers
            electoral_district = self._get_qid_qualifier(qualifiers, "P768")
            parliamentary_term = self._get_qid_qualifier(qualifiers, "P2937")
            parliamentary_group = self._get_qid_qualifier(qualifiers, "P4100")
            elected_in = self._get_qid_qualifier(qualifiers, "P2715")

            positions.append({
                "position_qid": pos_qid,
                "org_qid": org_qid,
                "start_date": start_date,
                "end_date": end_date,
                # Parliamentary context
                "electoral_district": electoral_district,
                "parliamentary_term": parliamentary_term,
                "parliamentary_group": parliamentary_group,
                "elected_in": elected_in,
            })
        return positions

    def _parse_time_value(self, time_str: str) -> Optional[str]:
        """
        Parse Wikidata time value to ISO date string.

        Args:
            time_str: Wikidata time format like "+2020-01-15T00:00:00Z"

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        if not time_str:
            return None
        # Remove leading + and extract date part
        time_str = time_str.lstrip("+")
        if "T" in time_str:
            return time_str.split("T")[0]
        return None

    def _classify_person_type(
        self,
        positions: list[dict],
        occupations: list[str],
    ) -> PersonType:
        """
        Determine PersonType from P39 positions and P106 occupations.

        Priority order:
        1. Check positions (more specific)
        2. Check occupations
        3. Default to UNKNOWN

        Args:
            positions: List of position dictionaries from _get_positions_with_org
            occupations: List of occupation QIDs from P106

        Returns:
            Classified PersonType
        """
        # Check positions first (more specific)
        for pos in positions:
            pos_qid = pos.get("position_qid", "")
            if pos_qid in EXECUTIVE_POSITION_QIDS:
                return PersonType.EXECUTIVE
            if pos_qid in POLITICIAN_POSITION_QIDS:
                return PersonType.POLITICIAN

        # Then check occupations
        for occ in occupations:
            if occ in OCCUPATION_TO_TYPE:
                return OCCUPATION_TO_TYPE[occ]

        # Default
        return PersonType.UNKNOWN

    def _get_org_or_context(self, pos: dict) -> str:
        """Get org QID from position, falling back to electoral district or parliamentary group."""
        return (
            pos.get("org_qid") or
            pos.get("electoral_district") or
            pos.get("parliamentary_group") or
            ""
        )

    def _get_best_role_org(
        self,
        positions: list[dict],
    ) -> tuple[str, str, str, Optional[str], Optional[str], dict]:
        """
        Select best position for role/org display.

        Priority:
        1. Positions with org/context and dates
        2. Positions with org/context
        3. Positions with dates
        4. Any position

        Args:
            positions: List of position dictionaries

        Returns:
            Tuple of (role_qid, org_label, org_qid, start_date, end_date, extra_context)
            Note: In dump mode, we return QIDs since we don't have labels
            extra_context contains parliamentary metadata
        """
        def has_context(pos: dict) -> bool:
            return bool(
                pos.get("org_qid") or
                pos.get("electoral_district") or
                pos.get("parliamentary_group")
            )

        def get_extra_context(pos: dict) -> dict:
            return {
                k: v for k, v in {
                    "electoral_district": pos.get("electoral_district"),
                    "parliamentary_term": pos.get("parliamentary_term"),
                    "parliamentary_group": pos.get("parliamentary_group"),
                    "elected_in": pos.get("elected_in"),
                }.items() if v
            }

        # Priority 1: Position with org/context and dates
        for pos in positions:
            if has_context(pos) and (pos.get("start_date") or pos.get("end_date")):
                return (
                    pos["position_qid"],
                    "",
                    self._get_org_or_context(pos),
                    pos.get("start_date"),
                    pos.get("end_date"),
                    get_extra_context(pos),
                )

        # Priority 2: Position with org/context
        for pos in positions:
            if has_context(pos):
                return (
                    pos["position_qid"],
                    "",
                    self._get_org_or_context(pos),
                    pos.get("start_date"),
                    pos.get("end_date"),
                    get_extra_context(pos),
                )

        # Priority 3: Position with dates
        for pos in positions:
            if pos.get("start_date") or pos.get("end_date"):
                return (
                    pos["position_qid"],
                    "",
                    self._get_org_or_context(pos),
                    pos.get("start_date"),
                    pos.get("end_date"),
                    get_extra_context(pos),
                )

        # Priority 4: Any position
        if positions:
            pos = positions[0]
            return (
                pos["position_qid"],
                "",
                self._get_org_or_context(pos),
                pos.get("start_date"),
                pos.get("end_date"),
                get_extra_context(pos),
            )

        return "", "", "", None, None, {}

    def _extract_person_records(self, entity: dict) -> list[PersonRecord]:
        """
        Extract one or more PersonRecords from entity dict.

        Creates a separate record for each position that has an org/context,
        so a CEO of 5 companies gets 5 entries linked by canonicalization.
        Falls back to a single record when no positions have orgs.

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            List of PersonRecords (empty if essential data is missing)
        """
        MAX_RECORDS_PER_PERSON = 10

        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        # Try English label first, fall back to any available label
        label = labels.get("en", {}).get("value", "")
        if not label:
            # Try to get any label
            for lang_data in labels.values():
                if isinstance(lang_data, dict) and lang_data.get("value"):
                    label = lang_data["value"]
                    break

        if not label or not qid:
            return []

        claims = entity.get("claims", {})

        # Get positions (P39) with qualifiers for org
        positions = self._get_positions_with_org(claims)
        # Get occupations (P106)
        occupations = self._get_claim_values(entity, "P106")

        # Classify person type from positions + occupations
        person_type = self._classify_person_type(positions, occupations)

        # Get country (P27 - country of citizenship)
        countries = self._get_claim_values(entity, "P27")
        country_qid = countries[0] if countries else ""

        # Get birth and death dates (P569, P570)
        birth_date = self._get_time_claim(claims, "P569")
        death_date = self._get_time_claim(claims, "P570")

        # Get description
        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        # Shared record data (position-specific fields added per record below)
        base_record_data = {
            "wikidata_id": qid,
            "label": label,
            "description": description,
            "positions": [p["position_qid"] for p in positions],
            "occupations": occupations,
            "country_qid": country_qid,
            "birth_date": birth_date,
            "death_date": death_date,
        }

        def make_record(
            role_qid: str, org_qid: str, start_date: Optional[str], end_date: Optional[str],
            extra_context: Optional[dict] = None,
            role_label_override: str = "",
        ) -> PersonRecord:
            role_label = role_label_override
            org_label = ""

            # Track discovered org
            if org_qid:
                self._discovered_orgs[org_qid] = org_qid

            record_data = {
                **base_record_data,
                "org_qid": org_qid,
                "role_qid": role_qid,
            }
            if extra_context:
                record_data.update(extra_context)

            return PersonRecord(
                name=label,
                source="wikidata",
                source_id=qid,
                country="",
                person_type=person_type,
                known_for_role=role_label,
                known_for_org_name=org_label,
                from_date=start_date,
                to_date=end_date,
                birth_date=birth_date,
                death_date=death_date,
                record=record_data,
            )

        # --- Multi-record path: one record per position with an org ---
        records: list[PersonRecord] = []
        seen_role_org: set[tuple[str, str]] = set()

        for pos in positions:
            org_qid_pos = self._get_org_or_context(pos)
            if not org_qid_pos:
                continue
            role_qid_pos = pos["position_qid"]
            key = (role_qid_pos, org_qid_pos)
            if key in seen_role_org:
                continue
            seen_role_org.add(key)

            extra_context = {
                k: v for k, v in {
                    "electoral_district": pos.get("electoral_district"),
                    "parliamentary_term": pos.get("parliamentary_term"),
                    "parliamentary_group": pos.get("parliamentary_group"),
                    "elected_in": pos.get("elected_in"),
                }.items() if v
            }
            records.append(make_record(
                role_qid_pos, org_qid_pos, pos.get("start_date"), pos.get("end_date"), extra_context,
            ))
            if len(records) >= MAX_RECORDS_PER_PERSON:
                break

        # --- Supplement with reverse org→person mappings (P169 CEO, P488 chair, etc.) ---
        # These come from org entities that list this person as an executive.
        # Creates additional records for roles not already covered by P39 qualifiers.
        if len(records) < MAX_RECORDS_PER_PERSON and qid in self._reverse_person_orgs:
            for rev_org_qid, rev_role, rev_start, rev_end in self._reverse_person_orgs[qid]:
                if len(records) >= MAX_RECORDS_PER_PERSON:
                    break
                # Deduplicate: skip if we already have a record for this org
                # (use org only since reverse role is a label string, not a QID)
                if any(rev_org_qid == key[1] for key in seen_role_org):
                    continue
                # Use a sentinel role_qid for reverse-mapped records
                seen_role_org.add(("_reverse", rev_org_qid))
                records.append(make_record(
                    "", rev_org_qid, rev_start, rev_end,
                    role_label_override=rev_role,
                ))
                logger.debug(
                    f"Added reverse-mapped record for {qid} ({label}): "
                    f"{rev_org_qid} as {rev_role}"
                )

        if records:
            logger.debug(f"{qid} ({label}): {len(records)} position records")
            return records

        # --- Fallback: single record (no positions had orgs, no reverse mappings) ---
        # Use _get_best_role_org for the single best position
        role_qid, _, org_qid, start_date, end_date, extra_context = self._get_best_role_org(positions)

        # Fallback: P108 employer
        if not org_qid:
            employers = self._get_claim_values(entity, "P108")
            if employers:
                org_qid = employers[0]
                logger.debug(f"Using top-level P108 employer for {qid}: {org_qid}")

        # Fallback chain: type-specific properties
        if not org_qid:
            fallback_props = [
                ("P1830", "owner-of"),
                ("P54", "sports-team"),
                ("P102", "political-party"),
                ("P241", "military-branch"),
                ("P264", "record-label"),
                ("P1416", "affiliation"),
                ("P463", "member-of"),
                ("P69", "educated-at"),
            ]
            for prop, prop_label in fallback_props:
                values = self._get_claim_values(entity, prop)
                if values:
                    org_qid = values[0]
                    logger.debug(f"Using {prop} ({prop_label}) for {qid}: {org_qid}")
                    break

        return [make_record(role_qid, org_qid, start_date, end_date, extra_context)]

    def _extract_org_data(
        self,
        entity: dict,
        entity_type: EntityType,
    ) -> Optional[CompanyRecord]:
        """
        Extract CompanyRecord from entity dict.

        Args:
            entity: Parsed Wikidata entity dictionary
            entity_type: Determined EntityType

        Returns:
            CompanyRecord or None if essential data is missing
        """
        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value", "")

        if not label or not qid:
            return None

        claims = entity.get("claims", {})

        # Get country (P17 - country)
        countries = self._get_claim_values(entity, "P17")
        country_qid = countries[0] if countries else ""

        # Get LEI (P1278)
        lei = self._get_string_claim(claims, "P1278")

        # Get ticker (P249)
        ticker = self._get_string_claim(claims, "P249")

        # Get description
        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        # Get inception date (P571)
        inception = self._get_time_claim(claims, "P571")

        # Get dissolution date (P576)
        dissolution = self._get_time_claim(claims, "P576")

        # Extract executive/leadership properties → reverse person lookup
        # These properties on the org point to people (person_qid)
        # Iterate claims directly to extract P580/P582 date qualifiers per claim
        executive_props = [
            ("P169", "chief executive officer"),
            ("P488", "chairperson"),
            ("P112", "founded by"),
            ("P1037", "director/manager"),
            ("P3320", "board member"),
        ]
        executives: list[dict[str, Any]] = []
        for prop, role_desc in executive_props:
            for claim in claims.get(prop, []):
                mainsnak = claim.get("mainsnak", {})
                person_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                if not person_qid:
                    continue
                qualifiers = claim.get("qualifiers", {})
                start_date = self._get_time_qualifier(qualifiers, "P580")
                end_date = self._get_time_qualifier(qualifiers, "P582")
                self._reverse_person_orgs.setdefault(person_qid, []).append(
                    (qid, role_desc, start_date, end_date)
                )
                executives.append({
                    "person_qid": person_qid,
                    "role": role_desc,
                    "start_date": start_date,
                    "end_date": end_date,
                })
                logger.debug(
                    f"Reverse mapping: {person_qid} → {qid} ({label}) as {role_desc} "
                    f"({start_date or '?'} – {end_date or '?'})"
                )

        record_data: dict[str, Any] = {
            "wikidata_id": qid,
            "label": label,
            "description": description,
            "lei": lei,
            "ticker": ticker,
            "country_qid": country_qid,
        }
        if executives:
            record_data["executives"] = executives

        return CompanyRecord(
            name=label,
            source="wikipedia",  # Use "wikipedia" per existing convention
            source_id=qid,
            region="",
            entity_type=entity_type,
            from_date=inception,
            to_date=dissolution,
            record=record_data,
        )

    def _process_location_entity(
        self,
        entity: dict,
        require_enwiki: bool = False,
    ) -> Optional[LocationRecord]:
        """
        Process a single entity, return LocationRecord if it's a location.

        Args:
            entity: Parsed Wikidata entity dictionary
            require_enwiki: If True, only include locations with English Wikipedia articles

        Returns:
            LocationRecord if entity qualifies, None otherwise
        """
        # Must be an item (not property)
        if entity.get("type") != "item":
            return None

        # Get location type from P31
        location_type_info = self._get_location_type(entity)
        if location_type_info is None:
            return None

        location_type_name, simplified_type = location_type_info

        # Optionally require English Wikipedia article
        if require_enwiki:
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" not in sitelinks:
                return None

        # Extract location data
        return self._extract_location_data(entity, location_type_name, simplified_type)

    def _extract_location_data(
        self,
        entity: dict,
        location_type: str,
        simplified_type: SimplifiedLocationType,
    ) -> Optional[LocationRecord]:
        """
        Extract LocationRecord from entity dict.

        Args:
            entity: Parsed Wikidata entity dictionary
            location_type: Detailed location type name
            simplified_type: Simplified location type enum

        Returns:
            LocationRecord or None if essential data is missing
        """
        qid = entity.get("id", "")
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value", "")

        if not label or not qid:
            return None

        claims = entity.get("claims", {})

        # Get parent locations from P131 (located in administrative territorial entity)
        # This gives us the full hierarchy (city -> state -> country)
        parent_qids = self._get_claim_values(entity, "P131")

        # Get country from P17 as fallback/additional parent
        country_qids = self._get_claim_values(entity, "P17")

        # Get coordinates from P625 (coordinate location)
        coordinates = self._get_coordinates(claims)

        # Get description
        descriptions = entity.get("descriptions", {})
        description = descriptions.get("en", {}).get("value", "")

        # Get inception date (P571) - when location was established
        inception = self._get_time_claim(claims, "P571")

        # Get dissolution date (P576) - when location ceased to exist
        dissolution = self._get_time_claim(claims, "P576")

        # Parse QID to integer
        qid_int = int(qid[1:]) if qid.startswith("Q") and qid[1:].isdigit() else None

        # Build record with extra details
        record_data = {
            "wikidata_id": qid,
            "label": label,
            "description": description,
            "parent_qids": parent_qids,
            "country_qids": country_qids,
        }
        if coordinates:
            record_data["coordinates"] = coordinates

        return LocationRecord(
            name=label,
            source="wikidata",
            source_id=qid,
            qid=qid_int,
            location_type=location_type,
            simplified_type=simplified_type,
            parent_ids=[],  # Will be resolved later by looking up parent QIDs in the database
            from_date=inception,
            to_date=dissolution,
            record=record_data,
        )

    def _get_coordinates(self, claims: dict) -> Optional[dict]:
        """
        Get coordinates from P625 (coordinate location).

        Args:
            claims: Claims dictionary

        Returns:
            Dict with lat/lon or None
        """
        for claim in claims.get("P625", []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                lat = value.get("latitude")
                lon = value.get("longitude")
                if lat is not None and lon is not None:
                    return {"lat": lat, "lon": lon}
        return None

    def import_locations(
        self,
        dump_path: Optional[Path] = None,
        limit: Optional[int] = None,
        require_enwiki: bool = False,
        skip_ids: Optional[set[str]] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> Iterator[LocationRecord]:
        """
        Stream through dump, yielding locations (geopolitical entities).

        This method filters the dump for:
        - Items with type "item"
        - Has P31 (instance of) matching a location type
        - Optionally: Has English Wikipedia article (enwiki sitelink)

        Args:
            dump_path: Path to dump file (uses self._dump_path if not provided)
            limit: Optional maximum number of records to return
            require_enwiki: If True, only include locations with English Wikipedia articles
            skip_ids: Optional set of source_ids (Q codes) to skip
            start_index: Entity index to start from (for resume support)
            progress_callback: Optional callback(entity_index, entity_id, records_yielded)

        Yields:
            LocationRecord for each qualifying location
        """
        path = dump_path or self._dump_path
        count = 0
        skipped_existing = 0
        current_entity_index = start_index

        logger.info("Starting location import from Wikidata dump...")
        if start_index > 0:
            logger.info(f"Resuming from entity index {start_index:,}")
        if not require_enwiki:
            logger.info("Importing ALL locations (no enwiki filter)")
        if skip_ids:
            logger.info(f"Skipping {len(skip_ids):,} existing Q codes")

        def track_entity(entity_index: int, entity_id: str) -> None:
            nonlocal current_entity_index
            current_entity_index = entity_index

        for entity in self.iter_entities(path, start_index=start_index, progress_callback=track_entity):
            if limit and count >= limit:
                break

            # Check skip_ids early, before full processing
            entity_id = entity.get("id", "")
            if skip_ids and entity_id in skip_ids:
                skipped_existing += 1
                continue

            record = self._process_location_entity(entity, require_enwiki=require_enwiki)
            if record:
                count += 1
                if count % 10_000 == 0:
                    logger.info(f"Yielded {count:,} location records (skipped {skipped_existing:,})...")

                # Call progress callback with current position
                if progress_callback:
                    progress_callback(current_entity_index, entity_id, count)

                yield record

        logger.info(f"Location import complete: {count:,} records (skipped {skipped_existing:,})")

    def _get_string_claim(self, claims: dict, prop: str) -> str:
        """
        Get first string value for a property.

        Args:
            claims: Claims dictionary
            prop: Property ID

        Returns:
            String value or empty string
        """
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value")
            if isinstance(value, str):
                return value
        return ""

    def _get_time_claim(self, claims: dict, prop: str) -> Optional[str]:
        """
        Get first time value for a property as ISO date string.

        Args:
            claims: Claims dictionary
            prop: Property ID

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        for claim in claims.get(prop, []):
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                time_str = value.get("time", "")
                # Format: +2020-01-15T00:00:00Z
                if time_str:
                    # Remove leading + and extract date part
                    time_str = time_str.lstrip("+")
                    if "T" in time_str:
                        return time_str.split("T")[0]
        return None

    def get_discovered_organizations(self) -> list[CompanyRecord]:
        """
        Get organizations discovered during the people import.

        These are organizations associated with people (from P39 P642 qualifiers)
        that can be inserted into the organizations database if not already present.

        Note: In dump mode, we only have QIDs, not labels.

        Returns:
            List of CompanyRecord objects for discovered organizations
        """
        records = []
        for org_qid in self._discovered_orgs:
            record = CompanyRecord(
                name=org_qid,  # Only have QID, not label
                source="wikipedia",
                source_id=org_qid,
                region="",
                entity_type=EntityType.BUSINESS,  # Default
                record={
                    "wikidata_id": org_qid,
                    "discovered_from": "people_import",
                    "needs_label_resolution": True,
                },
            )
            records.append(record)
        logger.info(f"Discovered {len(records)} organizations from people import")
        return records

    def clear_discovered_organizations(self) -> None:
        """Clear the discovered organizations cache."""
        self._discovered_orgs.clear()

    def get_reverse_person_orgs(self) -> dict[str, list[tuple[str, str, Optional[str], Optional[str]]]]:
        """
        Get the reverse person→org mappings built during org import.

        Maps person_qid → [(org_qid, role_description, start_date, end_date), ...] from org
        executive properties (P169 CEO, P488 chairperson, P112 founded by, P1037 director,
        P3320 board member).

        Use this after import to backfill people whose org entity appeared after their
        person entity in the dump (since the dump is processed in QID order).
        """
        return self._reverse_person_orgs

    def import_fk_relations(
        self,
        dump_path: Optional[Path] = None,
        start_index: int = 0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Iterator[tuple[str, str, dict]]:
        """
        Second pass: yield FK relationship data for each entity.

        Re-reads the dump and extracts only cross-table FK references
        (country QIDs, parent QIDs, org QIDs) without building full records.
        Also rebuilds _reverse_person_orgs for backfill.

        Yields:
            (entity_type, qid, fk_data) where fk_data contains QID references
        """
        path = dump_path or self._dump_path
        count = 0

        # Clear and rebuild reverse person orgs during pass 2
        self._reverse_person_orgs.clear()

        logger.info("Starting FK relations pass (pass 2)...")

        for entity in self.iter_entities(path, start_index=start_index, progress_callback=progress_callback):
            if entity.get("type") != "item":
                continue

            entity_id = entity.get("id", "")
            if not entity_id:
                continue

            claims = entity.get("claims", {})

            # Check entity type using same detection as pass 1
            if self._is_human(entity):
                # Person: extract country (P27) and org QID from positions/employer
                countries = self._get_claim_values(entity, "P27")
                country_qid = countries[0] if countries else ""

                # Get best org QID from positions
                positions = self._get_positions_with_org(claims)
                _, _, org_qid, _, _, _ = self._get_best_role_org(positions)

                # Fallback: P108 employer
                if not org_qid:
                    employers = self._get_claim_values(entity, "P108")
                    if employers:
                        org_qid = employers[0]

                if country_qid or org_qid:
                    fk_data: dict[str, str] = {}
                    if country_qid:
                        fk_data["country_qid"] = country_qid
                    if org_qid:
                        fk_data["org_qid"] = org_qid
                    count += 1
                    yield ("person", entity_id, fk_data)

            elif self._get_org_type(entity) is not None:
                # Org: extract country (P17) and rebuild reverse person orgs
                countries = self._get_claim_values(entity, "P17")
                country_qid = countries[0] if countries else ""

                # Rebuild reverse person orgs from executive properties
                executive_props = [
                    ("P169", "chief executive officer"),
                    ("P488", "chairperson"),
                    ("P112", "founded by"),
                    ("P1037", "director/manager"),
                    ("P3320", "board member"),
                ]
                for prop, role_desc in executive_props:
                    for claim in claims.get(prop, []):
                        mainsnak = claim.get("mainsnak", {})
                        person_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                        if not person_qid:
                            continue
                        qualifiers = claim.get("qualifiers", {})
                        start_date = self._get_time_qualifier(qualifiers, "P580")
                        end_date = self._get_time_qualifier(qualifiers, "P582")
                        self._reverse_person_orgs.setdefault(person_qid, []).append(
                            (entity_id, role_desc, start_date, end_date)
                        )

                if country_qid:
                    count += 1
                    yield ("org", entity_id, {"country_qid": country_qid})

            elif self._get_location_type(entity) is not None:
                # Location: extract parent QIDs (P131) and country QIDs (P17)
                parent_qids = self._get_claim_values(entity, "P131")
                country_qids = self._get_claim_values(entity, "P17")

                if parent_qids or country_qids:
                    fk_data = {}
                    if parent_qids:
                        fk_data["parent_qids"] = parent_qids
                    if country_qids:
                        fk_data["country_qids"] = country_qids
                    count += 1
                    yield ("location", entity_id, fk_data)

            if count % 100_000 == 0 and count > 0:
                logger.info(f"FK relations pass: {count:,} entities with FKs extracted")

        logger.info(f"FK relations pass complete: {count:,} entities with FK data")

    def _cache_position_jurisdiction(self, entity: dict) -> None:
        """
        Cache P1001 (applies to jurisdiction) or P17 (country) for position items.

        This enables backfilling org context for P39 claims where the position
        item inherently implies a jurisdiction (e.g. Q11696 "President of the
        United States" → Q30 "United States") but has no qualifier-level org.

        Only caches entities that have P1001 or P17 — the check is trivially
        cheap for entities that don't.
        """
        claims = entity.get("claims", {})
        # Only bother if the entity has P1001 or P17
        p1001_claims = claims.get("P1001", [])
        p17_claims = claims.get("P17", [])
        if not p1001_claims and not p17_claims:
            return

        qid = entity.get("id", "")
        if not qid:
            return

        # Prefer P1001 (applies to jurisdiction), fall back to P17 (country)
        for claim_list in (p1001_claims, p17_claims):
            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                value = mainsnak.get("datavalue", {}).get("value", {})
                jur_qid = value.get("id") if isinstance(value, dict) else None
                if jur_qid:
                    self._position_jurisdictions[qid] = jur_qid
                    return

