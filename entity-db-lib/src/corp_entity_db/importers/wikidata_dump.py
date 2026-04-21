"""
Wikidata dump importer for people and organizations.

Uses the Wikidata JSON dump (~100GB compressed) to import:
1. People: All humans (P31=Q5)
2. Organizations: All organizations (optionally filtered to those with English Wikipedia articles)

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
import time
import urllib.error
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
    # =========================================================================
    # POLITICIANS (elected officials)
    # =========================================================================
    "Q82955": PersonType.POLITICIAN,      # politician
    "Q193391": PersonType.POLITICIAN,     # member of parliament
    "Q372436": PersonType.POLITICIAN,     # statesperson
    "Q116852": PersonType.POLITICIAN,     # head of state (occupation)
    "Q2285706": PersonType.POLITICIAN,    # head of government (occupation)

    # =========================================================================
    # GOVERNMENT (civil servants, diplomats, appointed officials)
    # =========================================================================
    "Q212238": PersonType.GOVERNMENT,     # civil servant
    "Q806798": PersonType.GOVERNMENT,     # diplomat
    "Q15627169": PersonType.GOVERNMENT,   # trade unionist
    "Q889821": PersonType.GOVERNMENT,     # ambassador
    "Q15966511": PersonType.GOVERNMENT,   # diplomat (alternative)
    "Q294414": PersonType.GOVERNMENT,     # public office holder
    "Q3099732": PersonType.GOVERNMENT,    # ombudsman
    "Q11612": PersonType.GOVERNMENT,      # spy/intelligence officer
    "Q3380760": PersonType.GOVERNMENT,    # police officer

    # =========================================================================
    # MILITARY
    # =========================================================================
    "Q189290": PersonType.MILITARY,       # military officer
    "Q47064": PersonType.MILITARY,        # military personnel
    "Q4991371": PersonType.MILITARY,      # soldier
    "Q10669499": PersonType.MILITARY,     # naval officer
    "Q11974939": PersonType.MILITARY,     # air force officer
    "Q10974448": PersonType.MILITARY,     # army officer
    "Q36834": PersonType.MILITARY,        # general (rank)
    "Q467598": PersonType.MILITARY,       # admiral
    "Q156839": PersonType.MILITARY,       # colonel
    "Q130278": PersonType.MILITARY,       # lieutenant
    "Q19100": PersonType.MILITARY,        # captain (military)
    "Q55983715": PersonType.MILITARY,     # military leader

    # =========================================================================
    # LEGAL PROFESSIONALS
    # =========================================================================
    "Q16533": PersonType.LEGAL,           # judge
    "Q40348": PersonType.LEGAL,           # lawyer
    "Q185351": PersonType.LEGAL,          # jurist
    "Q3242871": PersonType.LEGAL,         # prosecutor
    "Q1792450": PersonType.LEGAL,         # barrister
    "Q3406182": PersonType.LEGAL,         # solicitor
    "Q1234634": PersonType.LEGAL,         # magistrate
    "Q1402561": PersonType.LEGAL,         # notary
    "Q188539": PersonType.LEGAL,          # advocate

    # =========================================================================
    # ATHLETES — root types and major sport-specific occupations
    # P279 subclass resolution will catch niche sport variants
    # =========================================================================
    "Q2066131": PersonType.ATHLETE,       # athlete (root type)
    "Q18536342": PersonType.ATHLETE,      # competitive sports person

    # Football/Soccer
    "Q937857": PersonType.ATHLETE,        # association football player
    "Q13382519": PersonType.ATHLETE,      # association football player (alternative QID)
    "Q628099": PersonType.ATHLETE,        # association football manager

    # Major team sports
    "Q3665646": PersonType.ATHLETE,       # basketball player
    "Q10871364": PersonType.ATHLETE,      # baseball player
    "Q19204627": PersonType.ATHLETE,      # ice hockey player
    "Q10833314": PersonType.ATHLETE,      # cricket player
    "Q12299841": PersonType.ATHLETE,      # cricketer (alternative)
    "Q13141064": PersonType.ATHLETE,      # rugby player
    "Q14089670": PersonType.ATHLETE,      # rugby union player
    "Q13415036": PersonType.ATHLETE,      # rugby league player
    "Q10843263": PersonType.ATHLETE,      # volleyball player
    "Q12840545": PersonType.ATHLETE,      # handball player
    "Q10871621": PersonType.ATHLETE,      # field hockey player
    "Q4009406": PersonType.ATHLETE,       # water polo player
    "Q11774891": PersonType.ATHLETE,      # American football player

    # Racket sports
    "Q10843402": PersonType.ATHLETE,      # tennis player
    "Q15117395": PersonType.ATHLETE,      # badminton player
    "Q15117302": PersonType.ATHLETE,      # table tennis player
    "Q13219637": PersonType.ATHLETE,      # squash player

    # Combat sports
    "Q11338576": PersonType.ATHLETE,      # boxer
    "Q847517": PersonType.ATHLETE,        # martial artist
    "Q13381689": PersonType.ATHLETE,      # judoka
    "Q11607585": PersonType.ATHLETE,      # wrestler
    "Q14128148": PersonType.ATHLETE,      # fencer (sport)
    "Q10873567": PersonType.ATHLETE,      # karateka
    "Q11422780": PersonType.ATHLETE,      # taekwondo practitioner

    # Track & field / Athletics
    "Q11513337": PersonType.ATHLETE,      # athletics competitor
    "Q4009182": PersonType.ATHLETE,       # sprinter
    "Q11296761": PersonType.ATHLETE,      # marathon runner

    # Water sports
    "Q10873124": PersonType.ATHLETE,      # swimmer
    "Q13382576": PersonType.ATHLETE,      # canoeist
    "Q13382608": PersonType.ATHLETE,      # rower
    "Q13382122": PersonType.ATHLETE,      # sailor (sport)
    "Q13561328": PersonType.ATHLETE,      # surfer
    "Q16029547": PersonType.ATHLETE,      # diver (sport)

    # Winter sports
    "Q4270517": PersonType.ATHLETE,       # skier
    "Q13381753": PersonType.ATHLETE,      # speed skater
    "Q13219587": PersonType.ATHLETE,      # figure skater
    "Q2340674": PersonType.ATHLETE,       # biathlete
    "Q14625788": PersonType.ATHLETE,      # cross-country skier
    "Q13382700": PersonType.ATHLETE,      # ski jumper
    "Q13382566": PersonType.ATHLETE,      # bobsledder
    "Q15117415": PersonType.ATHLETE,      # curler

    # Motor sports
    "Q11303721": PersonType.ATHLETE,      # racing driver
    "Q10843958": PersonType.ATHLETE,      # motorcycle racer

    # Individual sports
    "Q13381376": PersonType.ATHLETE,      # golfer
    "Q2309784": PersonType.ATHLETE,       # cyclist
    "Q13382981": PersonType.ATHLETE,      # sport shooter
    "Q13382363": PersonType.ATHLETE,      # weightlifter
    "Q11513339": PersonType.ATHLETE,      # gymnast
    "Q1647623": PersonType.ATHLETE,       # jockey
    "Q13381863": PersonType.ATHLETE,      # archer (sport)
    "Q14373094": PersonType.ATHLETE,      # triathlete
    "Q13474373": PersonType.ATHLETE,      # polo player
    "Q10842936": PersonType.ATHLETE,      # chess player

    # Coaching / Management
    "Q41583": PersonType.ATHLETE,         # coach (sport)
    "Q3303330": PersonType.ATHLETE,       # sports manager

    # =========================================================================
    # ARTISTS (traditional creative professions)
    # =========================================================================
    # Acting
    "Q33999": PersonType.ARTIST,          # actor
    "Q10800557": PersonType.ARTIST,       # film actor
    "Q10798782": PersonType.ARTIST,       # television actor
    "Q2405480": PersonType.ARTIST,        # voice actor
    "Q3387717": PersonType.ARTIST,        # stage actor
    "Q2259451": PersonType.ARTIST,        # stand-up comedian
    "Q6625963": PersonType.ARTIST,        # comedian
    "Q2490358": PersonType.ARTIST,        # choreographer

    # Music
    "Q177220": PersonType.ARTIST,         # singer
    "Q639669": PersonType.ARTIST,         # musician
    "Q488205": PersonType.ARTIST,         # singer-songwriter
    "Q753110": PersonType.ARTIST,         # songwriter
    "Q130857": PersonType.ARTIST,         # composer
    "Q183945": PersonType.ARTIST,         # record producer
    "Q806349": PersonType.ARTIST,         # band leader
    "Q855091": PersonType.ARTIST,         # rapper
    "Q158852": PersonType.ARTIST,         # conductor (music)
    "Q486748": PersonType.ARTIST,         # pianist
    "Q1415090": PersonType.ARTIST,        # guitarist
    "Q2722764": PersonType.ARTIST,        # DJ (disc jockey)
    "Q3658608": PersonType.ARTIST,        # opera singer
    "Q16145150": PersonType.ARTIST,       # drummer
    "Q1198887": PersonType.ARTIST,        # violinist
    "Q12800682": PersonType.ARTIST,       # saxophonist
    "Q1075651": PersonType.ARTIST,        # bassist
    "Q3076272": PersonType.ARTIST,        # organist
    "Q12377274": PersonType.ARTIST,       # trumpeter
    "Q1639825": PersonType.ARTIST,        # cellist
    "Q12902372": PersonType.ARTIST,       # flautist

    # Film / TV direction & production
    "Q2526255": PersonType.ARTIST,        # film director
    "Q3455803": PersonType.ARTIST,        # director
    "Q1053574": PersonType.ARTIST,        # television director
    "Q3282637": PersonType.ARTIST,        # film producer
    "Q28389": PersonType.ARTIST,          # screenwriter
    "Q578109": PersonType.ARTIST,         # television producer (also MEDIA, but creative)
    "Q4220892": PersonType.ARTIST,        # cinematographer
    "Q7042855": PersonType.ARTIST,        # film editor

    # Writing / Literature
    "Q36180": PersonType.ARTIST,          # writer
    "Q49757": PersonType.ARTIST,          # poet
    "Q4351403": PersonType.ARTIST,        # novelist
    "Q214917": PersonType.ARTIST,         # dramatist/playwright
    "Q15949613": PersonType.ARTIST,       # short story writer
    "Q333634": PersonType.ARTIST,         # translator
    "Q11774202": PersonType.ARTIST,       # essayist
    "Q4853732": PersonType.ARTIST,        # lyricist
    "Q482994": PersonType.ARTIST,         # manga artist
    "Q1114448": PersonType.ARTIST,        # cartoonist

    # Visual arts
    "Q483501": PersonType.ARTIST,         # artist (root type)
    "Q1028181": PersonType.ARTIST,        # painter
    "Q1281618": PersonType.ARTIST,        # sculptor
    "Q33231": PersonType.ARTIST,          # photographer
    "Q28640": PersonType.ARTIST,          # illustrator
    "Q644687": PersonType.ARTIST,         # graphic designer
    "Q5322166": PersonType.ARTIST,        # engraver
    "Q3391743": PersonType.ARTIST,        # visual artist
    "Q15296811": PersonType.ARTIST,       # video artist
    "Q17505902": PersonType.ARTIST,       # graffiti artist
    "Q627325": PersonType.ARTIST,         # graphic novelist

    # Design / Fashion
    "Q3501317": PersonType.ARTIST,        # fashion designer

    # Dance
    "Q5716684": PersonType.ARTIST,        # dancer
    "Q10843527": PersonType.ARTIST,       # ballet dancer

    # =========================================================================
    # MEDIA (internet/social media personalities, TV/radio hosts)
    # =========================================================================
    "Q6168364": PersonType.MEDIA,         # YouTuber
    "Q15077007": PersonType.MEDIA,        # podcaster
    "Q17125263": PersonType.MEDIA,        # social media influencer
    "Q15981151": PersonType.MEDIA,        # internet celebrity
    "Q2059704": PersonType.MEDIA,         # television personality
    "Q4610556": PersonType.MEDIA,         # model
    "Q2516866": PersonType.MEDIA,         # publisher
    "Q93191800": PersonType.MEDIA,        # content creator
    "Q105756498": PersonType.MEDIA,       # streamer (Twitch etc.)
    "Q19844021": PersonType.MEDIA,        # fashion model
    "Q3286043": PersonType.MEDIA,         # radio presenter
    "Q18844224": PersonType.MEDIA,        # radio personality
    "Q11631093": PersonType.MEDIA,        # TV host
    "Q15077008": PersonType.MEDIA,        # vlogger

    # =========================================================================
    # PROFESSIONALS (known for their profession/work)
    # =========================================================================
    # Medical
    "Q39631": PersonType.PROFESSIONAL,    # physician/doctor
    "Q774306": PersonType.PROFESSIONAL,   # surgeon
    "Q1234713": PersonType.PROFESSIONAL,  # dentist
    "Q15924224": PersonType.PROFESSIONAL, # psychiatrist
    "Q212980": PersonType.PROFESSIONAL,   # psychologist
    "Q3621491": PersonType.PROFESSIONAL,  # nurse
    "Q18805": PersonType.PROFESSIONAL,    # pharmacist
    "Q15895020": PersonType.PROFESSIONAL, # veterinarian
    "Q205375": PersonType.PROFESSIONAL,   # pharmacologist
    "Q1650260": PersonType.PROFESSIONAL,  # midwife

    # Engineering
    "Q81096": PersonType.PROFESSIONAL,    # engineer
    "Q5323050": PersonType.PROFESSIONAL,  # electrical engineer
    "Q13582652": PersonType.PROFESSIONAL, # civil engineer
    "Q81965": PersonType.PROFESSIONAL,    # software engineer
    "Q5482740": PersonType.PROFESSIONAL,  # data scientist
    "Q511093": PersonType.PROFESSIONAL,   # mechanical engineer

    # Architecture / Design
    "Q432386": PersonType.PROFESSIONAL,   # architect
    "Q3816358": PersonType.PROFESSIONAL,  # urban planner

    # Religious
    "Q42603": PersonType.PROFESSIONAL,    # priest/clergy
    "Q250867": PersonType.PROFESSIONAL,   # Catholic priest
    "Q611644": PersonType.PROFESSIONAL,   # Catholic bishop
    "Q1144754": PersonType.PROFESSIONAL,  # imam
    "Q133485": PersonType.PROFESSIONAL,   # rabbi
    "Q191808": PersonType.PROFESSIONAL,   # pastor
    "Q152002": PersonType.PROFESSIONAL,   # Anglican bishop
    "Q42857": PersonType.PROFESSIONAL,    # monk
    "Q219477": PersonType.PROFESSIONAL,   # missionary

    # Education
    "Q37226": PersonType.PROFESSIONAL,    # teacher
    "Q1607826": PersonType.PROFESSIONAL,  # school principal
    "Q23833535": PersonType.PROFESSIONAL, # school teacher

    # Other professions
    "Q131512": PersonType.PROFESSIONAL,   # chef
    "Q3499072": PersonType.PROFESSIONAL,  # pilot
    "Q15895449": PersonType.PROFESSIONAL, # accountant
    "Q806750": PersonType.PROFESSIONAL,   # consultant
    "Q584301": PersonType.PROFESSIONAL,   # economist
    "Q188094": PersonType.PROFESSIONAL,   # economist (alternative)
    "Q1371925": PersonType.PROFESSIONAL,  # real estate agent
    "Q266569": PersonType.PROFESSIONAL,   # librarian
    "Q3140857": PersonType.PROFESSIONAL,  # farmer
    "Q15839134": PersonType.PROFESSIONAL, # explorer
    "Q205862": PersonType.PROFESSIONAL,   # inventor
    "Q16323414": PersonType.PROFESSIONAL, # cartographer

    # =========================================================================
    # ACADEMICS (professors, researchers, scientists)
    # =========================================================================
    "Q121594": PersonType.ACADEMIC,       # professor
    "Q3400985": PersonType.ACADEMIC,      # academic
    "Q1622272": PersonType.ACADEMIC,      # university professor

    # Scientists (root + major disciplines)
    "Q901": PersonType.ACADEMIC,          # scientist
    "Q1650915": PersonType.ACADEMIC,      # researcher
    "Q169470": PersonType.ACADEMIC,       # physicist
    "Q593644": PersonType.ACADEMIC,       # chemist
    "Q864503": PersonType.ACADEMIC,       # biologist
    "Q11063": PersonType.ACADEMIC,        # astronomer
    "Q170790": PersonType.ACADEMIC,       # mathematician
    "Q2374149": PersonType.ACADEMIC,      # botanist
    "Q350979": PersonType.ACADEMIC,       # zoologist
    "Q520549": PersonType.ACADEMIC,       # geographer
    "Q11900058": PersonType.ACADEMIC,     # anthropologist
    "Q17167049": PersonType.ACADEMIC,     # sociologist
    "Q16267607": PersonType.ACADEMIC,     # astrophysicist
    "Q2306091": PersonType.ACADEMIC,      # political scientist
    "Q15632617": PersonType.ACADEMIC,     # computer scientist
    "Q201788": PersonType.ACADEMIC,       # historian
    "Q13570226": PersonType.ACADEMIC,     # geologist
    "Q2374463": PersonType.ACADEMIC,      # entomologist
    "Q11634": PersonType.ACADEMIC,        # art historian
    "Q2259532": PersonType.ACADEMIC,      # palaeontologist
    "Q15976092": PersonType.ACADEMIC,     # archaeologist
    "Q3126128": PersonType.ACADEMIC,      # philosopher
    "Q13418253": PersonType.ACADEMIC,     # linguist
    "Q488111": PersonType.ACADEMIC,       # philologist
    "Q11569986": PersonType.ACADEMIC,     # sinologist
    "Q11631524": PersonType.ACADEMIC,     # Egyptologist
    "Q10732476": PersonType.ACADEMIC,     # mycologist
    "Q3055126": PersonType.ACADEMIC,      # oceanographer
    "Q15839610": PersonType.ACADEMIC,     # epidemiologist
    "Q1662561": PersonType.ACADEMIC,      # microbiologist
    "Q7188": PersonType.ACADEMIC,         # biochemist
    "Q3560872": PersonType.ACADEMIC,      # neuroscientist
    "Q3178547": PersonType.ACADEMIC,      # geneticist
    "Q2919046": PersonType.ACADEMIC,      # ecologist

    # =========================================================================
    # JOURNALISTS
    # =========================================================================
    "Q1930187": PersonType.JOURNALIST,    # journalist
    "Q13590141": PersonType.JOURNALIST,   # news presenter
    "Q947873": PersonType.JOURNALIST,     # television presenter
    "Q4263842": PersonType.JOURNALIST,    # columnist
    "Q1086863": PersonType.JOURNALIST,    # war correspondent
    "Q13382487": PersonType.JOURNALIST,   # photojournalist
    "Q15978307": PersonType.JOURNALIST,   # broadcast journalist
    "Q15978631": PersonType.JOURNALIST,   # investigative journalist

    # =========================================================================
    # ACTIVISTS
    # =========================================================================
    "Q15253558": PersonType.ACTIVIST,     # activist
    "Q11631410": PersonType.ACTIVIST,     # human rights activist
    "Q18939491": PersonType.ACTIVIST,     # environmental activist
    "Q1476215": PersonType.ACTIVIST,      # trade union leader
    "Q2135538": PersonType.ACTIVIST,      # social activist
    "Q21072834": PersonType.ACTIVIST,     # women's rights activist
    "Q974144": PersonType.ACTIVIST,       # philanthropist
    "Q39894720": PersonType.ACTIVIST,     # peace activist

    # =========================================================================
    # EXECUTIVES / BUSINESS
    # =========================================================================
    "Q131524": PersonType.EXECUTIVE,      # entrepreneur
    "Q43845": PersonType.EXECUTIVE,       # businessperson
    "Q484876": PersonType.EXECUTIVE,      # CEO (as occupation)
    "Q1208543": PersonType.EXECUTIVE,     # landowner
    "Q3918409": PersonType.EXECUTIVE,     # investor
}

# =============================================================================
# ROLE/POSITION TYPE MAPPING (P31 - instance of)
# Entities with these P31 values are position/office items referenced by P39.
# =============================================================================

ROLE_TYPE_QIDS: frozenset[str] = frozenset({
    # Core position/occupation types
    "Q4164871",    # position
    "Q294414",     # public office
    "Q12737077",   # occupation
    "Q28640",      # profession

    # Occupation sub-types (by domain)
    "Q66715801",   # musical profession
    "Q88789639",   # artistic profession
    "Q4220920",    # filmmaking occupation
    "Q15839299",   # theatrical occupation
    "Q58635633",   # media profession
    "Q66666607",   # academic profession
    "Q137841866",  # foreign language profession
    "Q63188683",   # Christian religious occupation
    "Q63188808",   # Catholic vocation
    "Q138348066",  # female occupation
    "Q63187345",   # religious occupation
    "Q66811410",   # medical profession
    "Q15987302",   # legal profession
    "Q3139516",    # ecclesiastical occupation
    "Q91188763",   # Eastern Orthodox religious occupation
    "Q6857706",    # military profession
    "Q103810966",  # circus profession
    "Q110749524",  # show business profession
    "Q83856136",   # legal position
    "Q56604560",   # wood working profession
    "Q17279032",   # elective office
    "Q108377574",  # Islamic religious occupation
    "Q135106813",  # musical occupation (alternate)
    "Q137847894",  # clerical occupation
    "Q138024519",  # Roman Catholic episcopal title
    "Q138061574",  # Orthodox episcopal title
    "Q138348131",  # male occupation
    "Q16335296",   # historical profession
    "Q16631188",   # military position
    "Q3922583",    # health profession
    "Q349843",     # allied health profession
    "Q57260825",   # skilled trade
    "Q828803",     # job title
    "Q214339",     # role
    "Q355567",     # noble title
    "Q3320743",    # title of honor
    "Q486983",     # academic rank
    "Q4226220",    # naval officer rank
    "Q1474521",    # function
    "Q621504",     # police rank (Germany)
    "Q1221033",    # rank group
    "Q108392111",  # university head
    "Q98038492",   # membership type

    # Occupation classification systems
    "Q108300140",  # occupation group (ISCO-08)
    "Q119982961",  # profession and socioprofessional category (France)

    # Professional/title types
    "Q11488158",   # corporate title
    "Q20827480",   # certified professional
    "Q3529618",    # academic title
    "Q480319",     # title of authority
    "Q136649946",  # type of position

    # Social status (used by P106 for aristocrat, pensioner, etc.)
    "Q187588",     # social class
    "Q189970",     # social status

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
    - Humans (P31=Q5)
    - Organizations (optionally filtered to those with English Wikipedia articles)

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
        # Track ALL occupation QIDs referenced by people (from P106 claims).
        # Used after import to backfill any missing role records.
        self._needed_role_qids: set[str] = set()
        # Dynamic occupation → PersonType cache. Initialized from OCCUPATION_TO_TYPE,
        # then expanded via P279 (subclass of) chains as role entities are encountered.
        self._occupation_type_cache: dict[str, PersonType] = dict(OCCUPATION_TO_TYPE)

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

        p279_resolved = len(self._occupation_type_cache) - len(OCCUPATION_TO_TYPE)
        logger.info(
            f"Combined import complete: {people_count:,} people, {orgs_count:,} orgs, "
            f"{locations_count:,} locations, {roles_count:,} roles "
            f"(skipped {people_skipped:,} people, {orgs_skipped:,} orgs, {locations_skipped:,} locations)"
        )
        logger.info(
            f"Occupation type cache: {len(self._occupation_type_cache):,} total "
            f"({len(OCCUPATION_TO_TYPE)} static + {p279_resolved:,} resolved via P279 subclass chains)"
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
        """Check if entity is a role/position/occupation.

        Matches entities whose P31 is in ROLE_TYPE_QIDS (position, occupation, etc.)
        or whose QID is directly listed in OCCUPATION_TO_TYPE (known occupations like
        politician, entrepreneur, actor that may have different P31 types).
        """
        # Direct match: entity QID is a known occupation
        entity_qid = entity.get("id", "")
        if entity_qid in OCCUPATION_TO_TYPE:
            return True

        # P31-based match: entity is an instance of a role/position type
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

        # Resolve PersonType via P279 (subclass of) chain if not already known.
        # This expands the occupation→type mapping dynamically so that e.g.
        # "football coach" (subclass of "coach" subclass of "athlete") gets ATHLETE.
        if qid not in self._occupation_type_cache:
            self._resolve_occupation_subclass(entity, qid)

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

    def _resolve_occupation_subclass(self, entity: dict, qid: str) -> Optional[PersonType]:
        """Resolve PersonType for a role entity via its P279 (subclass of) parents.

        Checks direct P279 parents against the occupation type cache. If any parent
        is already resolved, this QID inherits that type. Deeper chains resolve
        transitively: parent entities resolve their own parents when processed
        (lower QIDs are processed first), so by the time we see a child, the
        chain is already built up in the cache.

        Args:
            entity: Parsed Wikidata entity dictionary for the role
            qid: QID of the role entity

        Returns:
            Resolved PersonType if found, None otherwise
        """
        parent_qids = self._get_claim_values(entity, "P279")
        if not parent_qids:
            return None

        for parent_qid in parent_qids:
            if parent_qid in self._occupation_type_cache:
                person_type = self._occupation_type_cache[parent_qid]
                self._occupation_type_cache[qid] = person_type
                return person_type

        return None

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

        # Then check occupations (using dynamic cache which includes P279-resolved subclasses)
        for occ in occupations:
            if occ in self._occupation_type_cache:
                return self._occupation_type_cache[occ]

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

    def _extract_person_records(self, entity: dict) -> list[PersonRecord]:
        """
        Extract one or more PersonRecords from entity dict.

        Creates a separate record for each distinct role+org combo from:
        1. Positions (P39) — with or without org
        2. Reverse org→person mappings (P169, P488, etc.)
        3. Occupations (P106) — role-only records for any not already covered
        4. Org fallback chain — only if steps 1-3 produced nothing

        Args:
            entity: Parsed Wikidata entity dictionary

        Returns:
            List of PersonRecords (empty if essential data is missing)
        """
        MAX_RECORDS_PER_PERSON = 5000

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
        # Get occupations (P106) and track for role backfill
        occupations = self._get_claim_values(entity, "P106")
        self._needed_role_qids.update(occupations)
        # Also track P39 position QIDs as needed roles
        for pos in positions:
            if pos.get("position_qid"):
                self._needed_role_qids.add(pos["position_qid"])

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

        # --- Step 1: One record per position (P39), with or without org ---
        records: list[PersonRecord] = []
        seen_role_org: set[tuple[str, str]] = set()
        seen_role_qids: set[str] = set()

        for pos in positions:
            if len(records) >= MAX_RECORDS_PER_PERSON:
                break
            org_qid_pos = self._get_org_or_context(pos)
            role_qid_pos = pos["position_qid"]
            key = (role_qid_pos, org_qid_pos or "")
            if key in seen_role_org:
                continue
            seen_role_org.add(key)
            seen_role_qids.add(role_qid_pos)

            extra_context = {
                k: v for k, v in {
                    "electoral_district": pos.get("electoral_district"),
                    "parliamentary_term": pos.get("parliamentary_term"),
                    "parliamentary_group": pos.get("parliamentary_group"),
                    "elected_in": pos.get("elected_in"),
                }.items() if v
            }
            records.append(make_record(
                role_qid_pos, org_qid_pos or "", pos.get("start_date"), pos.get("end_date"), extra_context,
            ))

        # --- Step 2: Supplement with reverse org→person mappings (P169 CEO, P488 chair, etc.) ---
        if len(records) < MAX_RECORDS_PER_PERSON and qid in self._reverse_person_orgs:
            for rev_org_qid, rev_role, rev_start, rev_end in self._reverse_person_orgs[qid]:
                if len(records) >= MAX_RECORDS_PER_PERSON:
                    break
                # Deduplicate: skip if we already have a record for this org
                if any(rev_org_qid == key[1] for key in seen_role_org):
                    continue
                seen_role_org.add(("_reverse", rev_org_qid))
                records.append(make_record(
                    "", rev_org_qid, rev_start, rev_end,
                    role_label_override=rev_role,
                ))
                logger.debug(
                    f"Added reverse-mapped record for {qid} ({label}): "
                    f"{rev_org_qid} as {rev_role}"
                )

        # --- Step 3: One record per occupation (P106) not already covered by positions ---
        for occ_qid in occupations:
            if len(records) >= MAX_RECORDS_PER_PERSON:
                break
            if occ_qid in seen_role_qids:
                continue
            key = (occ_qid, "")
            if key in seen_role_org:
                continue
            seen_role_org.add(key)
            seen_role_qids.add(occ_qid)
            records.append(make_record(occ_qid, "", None, None))

        if records:
            logger.debug(f"{qid} ({label}): {len(records)} records")
            return records

        # --- Step 4: No positions or occupations — org fallback chain ---
        org_qid = ""
        fallback_props = [
            ("P108", "employer"),
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

        return [make_record("", org_qid, None, None)]

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

        # Extract Wikidata aliases (alternative names in English)
        aliases_data = entity.get("aliases", {})
        en_aliases = aliases_data.get("en", [])
        wikidata_aliases: list[str] = []
        if en_aliases:
            label_lower = label.lower()
            for alias_entry in en_aliases:
                alias_val = alias_entry.get("value", "").strip() if isinstance(alias_entry, dict) else str(alias_entry).strip()
                if alias_val and alias_val.lower() != label_lower:
                    wikidata_aliases.append(alias_val)

        record_data: dict[str, Any] = {
            "wikidata_id": qid,
            "label": label,
            "description": description,
            "lei": lei,
            "ticker": ticker,
            "country_qid": country_qid,
        }
        if wikidata_aliases:
            record_data["wikidata_aliases"] = wikidata_aliases
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

    def get_discovered_org_qids(self) -> set[str]:
        """Return org QID strings discovered during people import."""
        return set(self._discovered_orgs.keys())

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

                # Get first org QID from positions
                positions = self._get_positions_with_org(claims)
                org_qid = ""
                for pos in positions:
                    org_qid = self._get_org_or_context(pos)
                    if org_qid:
                        break

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

    def backfill_aliases(
        self,
        conn: "sqlite3.Connection",
        dump_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> int:
        """
        Scan the dump and backfill wikidata_aliases into existing org record JSON.

        This is a lightweight pass that only reads aliases from entities whose QID
        matches an existing org. It updates the record JSON in-place (adds/updates
        the "wikidata_aliases" key) without modifying any other fields.

        Run this followed by `populate-aliases` and `build-index` to get alias
        search working without a full re-import.

        Args:
            conn: Writable SQLite connection
            dump_path: Path to dump file (uses self._dump_path if not provided)
            progress_callback: Optional callback(entity_index, entity_id)

        Returns:
            Number of org records updated with aliases
        """
        import sqlite3

        path = dump_path or self._dump_path
        if path is None:
            raise ValueError("No dump path provided. Call download_dump() first or pass dump_path.")

        # Load existing org QIDs
        cursor = conn.execute("SELECT qid FROM organizations WHERE qid IS NOT NULL AND alias_source_id IS NULL")
        org_qids: set[int] = {row[0] for row in cursor}
        logger.info(f"Loaded {len(org_qids):,} org QIDs to match against")

        updated = 0
        scanned = 0

        for entity in self.iter_entities(path, progress_callback=progress_callback):
            if entity.get("type") != "item":
                continue

            qid_str = entity.get("id", "")
            if not qid_str or not qid_str.startswith("Q"):
                continue

            qid_int_str = qid_str[1:]
            if not qid_int_str.isdigit():
                continue
            qid_int = int(qid_int_str)

            if qid_int not in org_qids:
                continue

            scanned += 1

            # Extract English aliases
            aliases_data = entity.get("aliases", {})
            en_aliases = aliases_data.get("en", [])
            if not en_aliases:
                continue

            label = entity.get("labels", {}).get("en", {}).get("value", "")
            label_lower = label.lower() if label else ""

            alias_list: list[str] = []
            for alias_entry in en_aliases:
                val = alias_entry.get("value", "").strip() if isinstance(alias_entry, dict) else str(alias_entry).strip()
                if val and val.lower() != label_lower:
                    alias_list.append(val)

            if not alias_list:
                continue

            # Read current record JSON, merge aliases, write back
            row = conn.execute(
                "SELECT id, record FROM organizations WHERE qid = ? AND alias_source_id IS NULL",
                (qid_int,),
            ).fetchone()
            if not row:
                continue

            record_data = json.loads(row[1]) if row[1] and row[1] != "{}" else {}
            record_data["wikidata_aliases"] = alias_list

            conn.execute(
                "UPDATE organizations SET record = ? WHERE id = ?",
                (json.dumps(record_data), row[0]),
            )
            updated += 1

            if updated % 10_000 == 0:
                conn.commit()
                logger.info(f"Updated {updated:,} org records with aliases (scanned {scanned:,} matching entities)...")

        conn.commit()
        logger.info(f"Alias backfill complete: {updated:,} orgs updated with aliases (scanned {scanned:,} matching entities)")
        return updated

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
        # Only bother if the entity has P1001, P17, or P131
        p1001_claims = claims.get("P1001", [])
        p17_claims = claims.get("P17", [])
        p131_claims = claims.get("P131", [])
        if not p1001_claims and not p17_claims and not p131_claims:
            return

        qid = entity.get("id", "")
        if not qid:
            return

        # Priority: P1001 (applies to jurisdiction) > P17 (country) > P131 (located in)
        for claim_list in (p1001_claims, p17_claims, p131_claims):
            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                value = mainsnak.get("datavalue", {}).get("value", {})
                jur_qid = value.get("id") if isinstance(value, dict) else None
                if jur_qid:
                    self._position_jurisdictions[qid] = jur_qid
                    return

    def get_missing_role_qids(self, existing_role_qids: set[int]) -> set[int]:
        """Return occupation/position QIDs referenced by people but not in the roles table.

        Args:
            existing_role_qids: Set of integer QIDs already in the roles table.

        Returns:
            Set of integer QIDs that need role records created.
        """
        missing = set()
        for qid_str in self._needed_role_qids:
            if qid_str.startswith("Q") and qid_str[1:].isdigit():
                qid_int = int(qid_str[1:])
                if qid_int not in existing_role_qids:
                    missing.add(qid_int)
        return missing

    @staticmethod
    def fetch_qid_labels(
        qids: set[int],
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[int, str]:
        """Fetch labels for QIDs from the Wikidata API.

        Prefers English labels, falls back to any available language.
        For QIDs with no label at all, uses the QID string as the name.

        Args:
            qids: Set of integer QIDs to look up.
            batch_size: Number of QIDs per API request (max 50).
            progress_callback: Optional callback(fetched_count, total_count) called after each batch.

        Returns:
            Mapping of QID int → label string (guaranteed to cover all input QIDs).
        """
        labels: dict[int, str] = {}
        qid_list = sorted(qids)
        total_batches = (len(qid_list) + batch_size - 1) // batch_size
        total_count = len(qid_list)
        for batch_idx, i in enumerate(range(0, len(qid_list), batch_size)):
            batch = qid_list[i:i + batch_size]
            ids_param = "|".join(f"Q{q}" for q in batch)
            url = (
                f"https://www.wikidata.org/w/api.php?action=wbgetentities"
                f"&ids={ids_param}&props=labels&format=json"
            )

            # Retry with exponential backoff on 429s
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "corp-entity-db/1.0 (Wikidata role backfill)"
                    })
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read())
                    for qid_str, ent in data.get("entities", {}).items():
                        if not (qid_str.startswith("Q") and qid_str[1:].isdigit()):
                            continue
                        all_labels = ent.get("labels", {})
                        # Prefer English, fall back to any language
                        label = all_labels.get("en", {}).get("value", "")
                        if not label:
                            for lang_data in all_labels.values():
                                if isinstance(lang_data, dict) and lang_data.get("value"):
                                    label = lang_data["value"]
                                    break
                        if label:
                            labels[int(qid_str[1:])] = label
                    break  # Success
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                        logger.info(f"Rate limited (429), waiting {wait}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        logger.warning(f"Failed to fetch labels for batch starting at Q{batch[0]}: {e}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to fetch labels for batch starting at Q{batch[0]}: {e}")
                    break

            # Rate limit: pause between batches to stay under Wikidata's limits
            if batch_idx < total_batches - 1:
                time.sleep(0.5)

            if progress_callback:
                progress_callback(min((batch_idx + 1) * batch_size, total_count), total_count)
            elif (batch_idx + 1) % 100 == 0:
                logger.info(f"  Fetched labels: {batch_idx + 1}/{total_batches} batches, {len(labels):,} labels so far")

        # Use QID as name for anything still missing
        for qid_int in qids - set(labels.keys()):
            labels[qid_int] = f"Q{qid_int}"

        return labels

