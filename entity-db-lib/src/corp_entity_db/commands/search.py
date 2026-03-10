"""Database search commands."""

from typing import Any, Optional

import click

from ._common import _configure_logging, _resolve_db_path


@click.command("search-people")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--role", type=str, default=None, help="Role/job title for composite search (e.g. 'CEO')")
@click.option("--org", type=str, default=None, help="Organization for composite search (e.g. 'Apple')")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search_people(query: str, db_path: Optional[str], top_k: int, role: Optional[str], org: Optional[str], verbose: bool):
    """
    Search for a person in the database.

    Uses composite embeddings: name, role, and org are embedded as separate
    256-dim segments and concatenated into a 768-dim vector. Use --role and
    --org to constrain results (e.g. find a specific CEO at a specific company).

    \b
    Examples:
        corp-entity-db search-people "Tim Cook"
        corp-entity-db search-people "Tim Cook" --role CEO --org Apple
        corp-entity-db search-people "Elon Musk" --top-k 5
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database
    from corp_entity_db.embeddings import CompanyEmbedder

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    click.echo(f"Searching for '{query}' in {db_path_obj}...", err=True)

    # Initialize components
    database = get_person_database(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    # Composite embedding search (primary) + name_normalized fallback
    query_embedding = embedder.embed_composite_person(query, role=role, org=org)

    results = database.search(
        query_embedding, top_k=top_k,
        query_name=query,
        embedder=embedder,
        query_role=role,
        query_org=org,
    )

    if not results:
        click.echo("No results found.", err=True)
        return

    click.echo(f"\nFound {len(results)} results:\n")
    for i, (record, similarity) in enumerate(results, 1):
        role_str = f" ({record.known_for_role})" if record.known_for_role else ""
        org_str = f" at {record.known_for_org_name}" if record.known_for_org_name else ""
        country_str = f" [{record.country}]" if record.country else ""
        click.echo(f"  {i}. {record.name}{role_str}{org_str}{country_str}")
        click.echo(f"     Source: wikidata:{record.source_id}, Type: {record.person_type.value}, Score: {similarity:.3f}")
        click.echo()

    database.close()


@click.command("people-test")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of results per query")
@click.option("--type", "person_type_filter", type=str, help="Only test a single person type (e.g. 'executive')")
@click.option("--for-llm", is_flag=True, help="Output structured results for LLM review (failures + ambiguous matches)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_people_test(
    db_path: Optional[str], top_k: int, person_type_filter: Optional[str],
    for_llm: bool, verbose: bool,
):
    """
    Run person searches across all PersonType categories (20 per type) to measure
    latency and accuracy. Each query has a known expected name — accuracy is scored
    by checking whether the expected person appears in the top-k results.

    \b
    Examples:
        corp-entity-db people-test
        corp-entity-db people-test --type politician
    """
    import time as _time

    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database
    from corp_entity_db.embeddings import CompanyEmbedder

    # Each entry: (name, expected_name_substring, query_kwargs)
    # query_kwargs contain person_type, role, org — used for composite embedding.
    test_queries_by_type: dict[str, list[tuple[str, str, dict[str, str]]]] = {
        "executive": [
            ("Tim Cook", "Tim Cook", {"person_type": "executive", "role": "CEO", "org": "Apple"}),
            ("Satya Nadella", "Satya Nadella", {"person_type": "executive", "role": "CEO", "org": "Microsoft"}),
            ("Andy Jassy", "Andy Jassy", {"person_type": "executive", "role": "CEO", "org": "Amazon"}),
            ("Sundar Pichai", "Sundar Pichai", {"person_type": "executive", "role": "CEO", "org": "Google", "expected_org": "Alphabet"}),
            ("Jensen Huang", "Jensen Huang", {"person_type": "executive", "role": "CEO", "org": "NVIDIA"}),
            ("Mark Zuckerberg", "Mark Zuckerberg", {"person_type": "executive", "role": "CEO", "org": "Meta", "expected_org": "Chan Zuckerberg"}),
            ("Jamie Dimon", "Jamie Dimon", {"person_type": "executive", "role": "CEO", "org": "JPMorgan Chase"}),
            ("Warren Buffett", "Warren Buffett", {"person_type": "executive", "role": "CEO", "org": "Berkshire Hathaway"}),
            ("Elon Musk", "Elon Musk", {"person_type": "executive", "role": "CEO", "org": "Tesla", "expected_org": ""}),
            ("Lisa Su", "Lisa Su", {"person_type": "executive", "role": "CEO", "org": "AMD", "expected_org": "Cisco"}),
            ("Mary Barra", "Mary Barra", {"person_type": "executive", "role": "CEO", "org": "General Motors", "expected_org": "The Walt Disney"}),
            ("David Solomon", "David M. Solomon", {"person_type": "executive", "role": "CEO", "org": "Goldman Sachs"}),
            ("Brian Moynihan", "Brian Moynihan", {"person_type": "executive", "role": "CEO", "org": "Bank of America"}),
            ("Arvind Krishna", "Arvind Krishna", {"person_type": "executive", "role": "CEO", "org": "IBM", "expected_type": "unknown"}),
            ("Pat Gelsinger", "Pat Gelsinger", {"person_type": "executive", "role": "CEO", "org": "Intel", "expected_type": "professional", "expected_org": "Intel Corporation"}),
            ("Chuck Robbins", "Chuck Robbins", {"person_type": "executive", "role": "CEO", "org": "Cisco"}),
            ("Safra Catz", "Safra Catz", {"person_type": "executive", "role": "CEO", "org": "Oracle", "expected_org": "The Walt Disney"}),
            ("Shantanu Narayen", "Shantanu Narayen", {"person_type": "executive", "role": "CEO", "org": "Adobe", "expected_org": "Pfizer"}),
            ("Marc Benioff", "Marc Benioff", {"person_type": "executive", "role": "CEO", "org": "Salesforce"}),
            ("Darius Adamczyk", "Darius Adamczyk", {"person_type": "executive", "role": "CEO", "org": "Honeywell", "expected_org": "Johnson & Johnson"}),
            # Entrepreneurs (merged from entrepreneur type)
            ("Jeff Bezos", "Jeff Bezos", {"person_type": "executive", "role": "founder", "org": "Amazon", "expected_org": ""}),
            ("Bill Gates", "Bill Gates", {"person_type": "executive", "role": "founder", "org": "Microsoft", "expected_org": ""}),
            ("Larry Page", "Larry Page", {"person_type": "executive", "role": "co-founder", "org": "Google", "expected_org": ""}),
            ("Sergey Brin", "Sergey Brin", {"person_type": "executive", "role": "co-founder", "org": "Google", "expected_type": "unknown", "expected_org": ""}),
            ("Jack Dorsey", "Jack Dorsey", {"person_type": "executive", "role": "founder", "org": "Twitter", "expected_org": ""}),
            ("Reid Hoffman", "Reid Hoffman", {"person_type": "executive", "role": "co-founder", "org": "LinkedIn", "expected_org": ""}),
            ("Peter Thiel", "Peter Thiel", {"person_type": "executive", "role": "co-founder", "org": "PayPal", "expected_type": "government", "expected_org": ""}),
            ("Travis Kalanick", "Travis Kalanick", {"person_type": "executive", "role": "co-founder", "org": "Uber", "expected_org": ""}),
            ("Brian Chesky", "Brian Chesky", {"person_type": "executive", "role": "CEO", "org": "Airbnb", "expected_org": ""}),
            ("Jack Ma", "Jack Ma", {"person_type": "executive", "role": "founder", "org": "Alibaba", "expected_org": ""}),
            ("Richard Branson", "Richard Branson", {"person_type": "executive", "role": "founder", "org": "Virgin Group", "expected_org": ""}),
            ("Sam Altman", "Sam Altman", {"person_type": "executive", "role": "CEO", "org": "OpenAI", "expected_org": ""}),
            ("Evan Spiegel", "Evan Spiegel", {"person_type": "executive", "role": "CEO", "org": "Snap", "expected_org": ""}),
            ("Daniel Ek", "Daniel Ek", {"person_type": "executive", "role": "CEO", "org": "Spotify", "expected_org": ""}),
            ("Patrick Collison", "Patrick Collison", {"person_type": "executive", "role": "CEO", "org": "Stripe", "expected_type": "professional", "expected_org": ""}),
            ("Whitney Wolfe Herd", "Whitney Wolfe Herd", {"person_type": "executive", "role": "CEO", "org": "Bumble", "expected_org": ""}),
            ("Stewart Butterfield", "Stewart Butterfield", {"person_type": "executive", "role": "co-founder", "org": "Slack", "expected_org": ""}),
            ("Drew Houston", "Drew Houston", {"person_type": "executive", "role": "CEO", "org": "Dropbox", "expected_org": ""}),
            ("Tony Hsieh", "Tony Hsieh", {"person_type": "executive", "role": "CEO", "org": "Zappos", "expected_org": ""}),
            ("Steve Jobs", "Steve Jobs", {"person_type": "executive", "role": "founder", "org": "Apple", "expected_org": ""}),
        ],
        "politician": [
            ("Joe Biden", "Joe Biden", {"person_type": "politician", "role": "President", "org": "United States"}),
            ("Donald Trump", "Donald Trump", {"person_type": "politician", "role": "President", "org": "United States", "expected_type": "executive", "expected_org": ""}),
            ("Emmanuel Macron", "Emmanuel Macron", {"person_type": "politician", "role": "President", "org": "France", "expected_org": ""}),
            ("Rishi Sunak", "Rishi Sunak", {"person_type": "politician", "role": "Prime Minister", "org": "United Kingdom", "expected_org": ""}),
            ("Olaf Scholz", "Olaf Scholz", {"person_type": "politician", "role": "Chancellor", "org": "Germany"}),
            ("Justin Trudeau", "Justin Trudeau", {"person_type": "politician", "role": "Prime Minister", "org": "Canada", "expected_type": "legal", "expected_org": ""}),
            ("Narendra Modi", "Narendra Modi", {"person_type": "politician", "role": "Prime Minister", "org": "India", "expected_org": ""}),
            ("Fumio Kishida", "Fumio Kishida", {"person_type": "politician", "role": "Prime Minister", "org": "Japan"}),
            ("Volodymyr Zelenskyy", "Volodymyr Zelenskyy", {"person_type": "politician", "role": "President", "org": "Ukraine"}),
            ("Pedro Sánchez", "Pedro Sánchez", {"person_type": "politician", "role": "Prime Minister", "org": "Spain", "expected_org": "Madrid"}),
            ("Anthony Albanese", "Anthony Albanese", {"person_type": "politician", "role": "Prime Minister", "org": "Australia", "expected_org": ""}),
            ("Giorgia Meloni", "Giorgia Meloni", {"person_type": "politician", "role": "Prime Minister", "org": "Italy"}),
            ("Luiz Inácio Lula da Silva", "Luiz Inácio Lula da Silva", {"person_type": "politician", "role": "President", "org": "Brazil", "expected_type": "government", "expected_org": ""}),
            ("Jacinda Ardern", "Jacinda Ardern", {"person_type": "politician", "role": "Prime Minister", "org": "New Zealand", "expected_org": ""}),
            ("Xi Jinping", "Xi Jinping", {"person_type": "politician", "role": "President", "org": "China", "expected_org": ""}),
            ("Recep Tayyip Erdoğan", "Recep Tayyip Erdoğan", {"person_type": "politician", "role": "President", "org": "Turkey", "expected_org": ""}),
            ("Benjamin Netanyahu", "Benjamin Netanyahu", {"person_type": "politician", "role": "Prime Minister", "org": "Israel", "expected_org": ""}),
            ("Yoon Suk Yeol", "Yoon Suk Yeol", {"person_type": "legal", "role": "President", "org": "South Korea"}),
            ("Mark Rutte", "Mark Rutte", {"person_type": "politician", "role": "Prime Minister", "org": "Netherlands", "expected_org": ""}),
            ("Keir Starmer", "Keir Starmer", {"person_type": "politician", "role": "Prime Minister", "org": "United Kingdom", "expected_org": ""}),
        ],
        "government": [
            ("Janet Yellen", "Janet Yellen", {"person_type": "government", "role": "Secretary of the Treasury", "org": "United States", "expected_type": "professional", "expected_org": ""}),
            ("Antony Blinken", "Antony Blinken", {"person_type": "government", "role": "Secretary of State", "org": "United States", "expected_type": "politician", "expected_org": ""}),
            ("Jerome Powell", "Jerome Powell", {"person_type": "government", "role": "Chair", "org": "Federal Reserve", "expected_type": "legal", "expected_org": ""}),
            ("Christine Lagarde", "Christine Lagarde", {"person_type": "government", "role": "President", "org": "European Central Bank", "expected_type": "politician", "expected_org": ""}),
            ("Ursula von der Leyen", "Ursula von der Leyen", {"person_type": "government", "role": "President", "org": "European Commission", "expected_type": "professional", "expected_org": ""}),
            ("António Guterres", "António Guterres", {"person_type": "government", "role": "Secretary-General", "org": "United Nations", "expected_type": "politician"}),
            ("Tedros Adhanom Ghebreyesus", "Tedros Adhanom Ghebreyesus", {"person_type": "government", "role": "Director-General", "org": "WHO", "expected_type": "politician", "expected_org": "World Health Organization"}),
            ("Kristalina Georgieva", "Kristalina Georgieva", {"person_type": "government", "role": "Managing Director", "org": "IMF", "expected_type": "executive", "expected_org": ""}),
            ("Gary Gensler", "Gary Gensler", {"person_type": "government", "role": "Chair", "org": "SEC", "expected_type": "politician", "expected_org": ""}),
            ("Merrick Garland", "Merrick Garland", {"person_type": "government", "role": "Attorney General", "org": "United States", "expected_type": "legal"}),
            ("Andrew Bailey", "Andrew Bailey", {"person_type": "government", "role": "Governor", "org": "Bank of England", "expected_org": "Financial Stability Board"}),
            ("Haruhiko Kuroda", "Haruhiko Kuroda", {"person_type": "government", "role": "Governor", "org": "Bank of Japan", "expected_type": "academic", "expected_org": "Asian Development Bank"}),
            ("Lloyd Austin", "Lloyd J. Austin III", {"person_type": "government", "role": "Secretary of Defense", "org": "United States", "expected_type": "military", "expected_org": ""}),
            ("Avril Haines", "Avril Haines", {"person_type": "government", "role": "Director of National Intelligence", "org": "United States", "expected_type": "legal"}),
            ("Alejandro Mayorkas", "Alejandro Mayorkas", {"person_type": "government", "role": "Secretary", "org": "DHS", "expected_type": "legal", "expected_org": ""}),
            ("Gina Raimondo", "Gina Raimondo", {"person_type": "government", "role": "Secretary of Commerce", "org": "United States", "expected_type": "politician", "expected_org": ""}),
            ("Janet Woodcock", "Janet Woodcock", {"person_type": "government", "role": "Commissioner", "org": "FDA", "expected_type": "academic", "expected_org": ""}),
            ("Ajay Banga", "Ajay K Banga", {"person_type": "government", "role": "President", "org": "World Bank", "expected_type": "academic", "expected_org": ""}),
            ("Ngozi Okonjo-Iweala", "Ngozi Okonjo-Iweala", {"person_type": "government", "role": "Director-General", "org": "WTO", "expected_type": "politician", "expected_org": ""}),
            ("Charles Michel", "Charles Michel", {"person_type": "government", "role": "President", "org": "European Council", "expected_type": "politician"}),
        ],
        "military": [
            ("Mark A. Milley", "Mark A. Milley", {"person_type": "military", "role": "Chairman Joint Chiefs of Staff", "org": "United States", "expected_type": "unknown"}),
            ("Valerii Zaluzhnyi", "Valerii Zaluzhnyi", {"person_type": "military", "role": "Commander-in-Chief", "org": "Ukraine Armed Forces", "expected_org": ""}),
            ("Tony Radakin", "Tony Radakin", {"person_type": "military", "role": "Chief of Defence Staff", "org": "United Kingdom"}),
            ("Thierry Burkhard", "Thierry Burkhard", {"person_type": "military", "role": "Chief of Defence Staff", "org": "France"}),
            ("Rob Bauer", "Rob Bauer", {"person_type": "military", "role": "Chair", "org": "NATO Military Committee", "expected_org": ""}),
            ("Christopher Cavoli", "Christopher G. Cavoli", {"person_type": "military", "role": "SACEUR", "org": "NATO", "expected_type": "unknown", "expected_org": ""}),
            ("Michael Kurilla", "Michael Kurilla", {"person_type": "military", "role": "Commander", "org": "CENTCOM", "expected_org": ""}),
            ("Charles Q. Brown Jr", "Charles Q. Brown Jr.", {"person_type": "military", "role": "Chairman Joint Chiefs of Staff", "org": "United States", "expected_type": "unknown"}),
            ("Eberhard Zorn", "Eberhard Zorn", {"person_type": "military", "role": "Inspector General", "org": "Germany Bundeswehr", "expected_org": ""}),
            ("Koji Yamazaki", "Koji Yamazaki", {"person_type": "military", "role": "Chief of Staff", "org": "Japan Self-Defense Forces", "expected_org": "Japan"}),
            ("Angus Campbell", "Angus Campbell", {"person_type": "military", "role": "Chief of Defence Force", "org": "Australia", "expected_org": ""}),
            ("Bipin Rawat", "Bipin Rawat", {"person_type": "military", "role": "Chief of Defence Staff", "org": "India"}),
            ("Wayne Eyre", "Wayne Eyre", {"person_type": "military", "role": "Chief of Defence Staff", "org": "Canada"}),
            ("Sergei Shoigu", "Sergei Shoigu", {"person_type": "military", "role": "Minister of Defence", "org": "Russia", "expected_type": "politician", "expected_org": ""}),
            ("James Hecker", "James Hecker", {"person_type": "military", "role": "Commander", "org": "US Air Forces in Europe", "expected_type": "unknown", "expected_org": ""}),
            ("Samuel Paparo", "Samuel Paparo", {"person_type": "military", "role": "Commander", "org": "US Indo-Pacific Command", "expected_org": ""}),
            ("Laura Richardson", "Laura Richardson", {"person_type": "military", "role": "Commander", "org": "US Southern Command", "expected_type": "politician", "expected_org": ""}),
            ("Mauro Del Vecchio", "Mauro Del Vecchio", {"person_type": "military", "role": "Commander", "org": "NATO Joint Force Command", "expected_type": "politician", "expected_org": ""}),
            ("Stuart Peach, Baron Peach", "Stuart Peach, Baron Peach", {"person_type": "military", "role": "Chair", "org": "NATO Military Committee", "expected_org": ""}),
            ("Eirik Kristoffersen", "Eirik Kristoffersen", {"person_type": "military", "role": "Chief of Defence", "org": "Norway"}),
        ],
        "legal": [
            ("John Roberts", "John Roberts", {"person_type": "legal", "role": "Chief Justice", "org": "Supreme Court of the United States", "expected_org": "United States"}),
            ("Sonia Sotomayor", "Sonia Sotomayor", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Ketanji Brown Jackson", "Ketanji Brown Jackson", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Clarence Thomas", "Clarence Thomas", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Elena Kagan", "Elena Kagan", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Neil Gorsuch", "Neil Gorsuch", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": "United States"}),
            ("Brett Kavanaugh", "Brett Kavanaugh", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Amy Coney Barrett", "Amy Coney Barrett", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Samuel Alito", "Samuel Alito", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Ruth Bader Ginsburg", "Ruth Bader Ginsburg", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States", "expected_org": ""}),
            ("Brenda Hale", "Brenda Hale", {"person_type": "legal", "role": "President", "org": "Supreme Court, United Kingdom", "expected_type": "politician", "expected_org": ""}),
            ("Merrick Garland", "Merrick Garland", {"person_type": "legal", "role": "Attorney General", "org": "United States Department of Justice", "expected_org": ""}),
            ("Karim Khan", "Karim Khan", {"person_type": "legal", "role": "Prosecutor", "org": "International Criminal Court", "expected_type": "professional", "expected_org": ""}),
            ("Joan Donoghue", "Joan Donoghue", {"person_type": "legal", "role": "President", "org": "International Court of Justice", "expected_org": ""}),
            ("Didier Reynders", "Didier Reynders", {"person_type": "legal", "role": "Commissioner for Justice", "org": "European Commission", "expected_type": "politician", "expected_org": "European Union"}),
            ("Fatou Bensouda", "Fatou Bensouda", {"person_type": "legal", "role": "Prosecutor", "org": "International Criminal Court", "expected_type": "politician", "expected_org": ""}),
            ("Loretta Lynch", "Loretta Lynch", {"person_type": "legal", "role": "Attorney General", "org": "United States", "expected_org": ""}),
            ("Eric Holder", "Eric Holder", {"person_type": "legal", "role": "Attorney General", "org": "United States", "expected_org": ""}),
            ("Robert Mueller", "Robert Mueller", {"person_type": "legal", "role": "Special Counsel", "org": "Department of Justice", "expected_type": "military", "expected_org": ""}),
            ("Jack Smith", "Jack Smith", {"person_type": "legal", "role": "Special Counsel", "org": "Department of Justice"}),
        ],
        "professional": [
            ("Atul Gawande", "Atul Gawande", {"person_type": "professional", "role": "surgeon", "org": "Brigham and Women's Hospital", "expected_type": "journalist", "expected_org": ""}),
            ("Sanjay Gupta", "Sanjay Gupta", {"person_type": "professional", "role": "neurosurgeon", "org": "Emory University Hospital", "expected_org": ""}),
            ("Anthony Fauci", "Anthony Fauci", {"person_type": "professional", "role": "immunologist", "org": "NIAID", "expected_org": ""}),
            ("Devi Shetty", "Devi Shetty", {"person_type": "professional", "role": "cardiac surgeon", "org": "Narayana Health", "expected_org": ""}),
            ("Norman Foster", "Norman Foster", {"person_type": "professional", "role": "architect", "org": "Foster + Partners", "expected_type": "politician", "expected_org": ""}),
            ("Bjarke Ingels", "Bjarke Ingels", {"person_type": "professional", "role": "architect", "org": "BIG", "expected_type": "academic", "expected_org": ""}),
            ("Zaha Hadid", "Zaha Hadid", {"person_type": "professional", "role": "architect", "org": "Zaha Hadid Architects", "expected_type": "academic"}),
            ("Renzo Piano", "Renzo Piano", {"person_type": "professional", "role": "architect", "org": "Renzo Piano Building Workshop", "expected_type": "politician", "expected_org": ""}),
            ("Frank Gehry", "Frank Gehry", {"person_type": "professional", "role": "architect", "org": "Gehry Partners", "expected_type": "artist", "expected_org": ""}),
            ("Tadao Ando", "Tadao Ando", {"person_type": "professional", "role": "architect", "expected_type": "athlete"}),
            ("I. M. Pei", "I. M. Pei", {"person_type": "unknown", "role": "architect", "org": "Pei Cobb Freed", "expected_org": ""}),
            ("Santiago Calatrava", "Santiago Calatrava", {"person_type": "professional", "role": "architect", "expected_type": "artist"}),
            ("Rem Koolhaas", "Rem Koolhaas", {"person_type": "professional", "role": "architect", "org": "OMA", "expected_type": "artist", "expected_org": ""}),
            ("Thomas Heatherwick", "Thomas Heatherwick", {"person_type": "professional", "role": "designer", "org": "Heatherwick Studio", "expected_type": "artist", "expected_org": ""}),
            ("Jony Ive", "Jony Ive", {"person_type": "professional", "role": "designer", "org": "Apple", "expected_type": "unknown", "expected_org": ""}),
            ("Dieter Rams", "Dieter Rams", {"person_type": "professional", "role": "designer", "org": "Braun", "expected_type": "academic", "expected_org": ""}),
            ("Philippe Starck", "Philippe Starck", {"person_type": "professional", "role": "designer", "expected_type": "executive"}),
            ("Toyo Ito", "Toyo Ito", {"person_type": "unknown", "role": "architect"}),
            ("Daniel Libeskind", "Daniel Libeskind", {"person_type": "professional", "role": "architect", "expected_type": "artist"}),
            ("Peter Zumthor", "Peter Zumthor", {"person_type": "professional", "role": "architect", "expected_type": "academic"}),
        ],
        "academic": [
            ("Noam Chomsky", "Noam Chomsky", {"person_type": "academic", "role": "professor", "org": "MIT", "expected_org": ""}),
            ("Steven Pinker", "Steven Pinker", {"person_type": "academic", "role": "professor", "org": "Harvard", "expected_type": "professional", "expected_org": ""}),
            ("Paul Krugman", "Paul Krugman", {"person_type": "academic", "role": "professor", "org": "Princeton", "expected_type": "professional", "expected_org": ""}),
            ("Joseph Stiglitz", "Joseph E. Stiglitz", {"person_type": "academic", "role": "professor", "org": "Columbia", "expected_type": "professional", "expected_org": ""}),
            ("Thomas Piketty", "Thomas Piketty", {"person_type": "academic", "role": "professor", "org": "Paris School of Economics", "expected_type": "professional"}),
            ("Yuval Noah Harari", "Yuval Noah Harari", {"person_type": "academic", "role": "professor", "org": "Hebrew University", "expected_type": "artist", "expected_org": ""}),
            ("Niall Ferguson", "Niall Ferguson", {"person_type": "academic", "role": "professor", "org": "Stanford", "expected_org": ""}),
            ("Lawrence Lessig", "Lawrence Lessig", {"person_type": "academic", "role": "professor", "org": "Harvard Law School", "expected_type": "legal", "expected_org": ""}),
            ("Cornel West", "Cornel West", {"person_type": "academic", "role": "professor", "org": "Union Theological Seminary", "expected_type": "professional", "expected_org": ""}),
            ("Nassim Nicholas Taleb", "Nassim Nicholas Taleb", {"person_type": "academic", "role": "professor", "org": "NYU", "expected_type": "professional", "expected_org": ""}),
            ("Jordan Peterson", "Jordan Peterson", {"person_type": "academic", "role": "professor", "org": "University of Toronto"}),
            ("Richard Dawkins", "Richard Dawkins", {"person_type": "academic", "role": "professor", "org": "Oxford", "expected_type": "artist", "expected_org": ""}),
            ("Amy Cuddy", "Amy Cuddy", {"person_type": "academic", "role": "professor", "org": "Harvard Business School", "expected_org": ""}),
            ("Brené Brown", "Brené Brown", {"person_type": "academic", "role": "professor", "org": "University of Houston", "expected_org": ""}),
            ("Henry Kissinger", "Henry Kissinger", {"person_type": "academic", "role": "professor", "org": "Georgetown", "expected_type": "politician", "expected_org": ""}),
            ("Daron Acemoğlu", "Daron Acemoğlu", {"person_type": "academic", "role": "professor", "org": "MIT", "expected_type": "professional", "expected_org": ""}),
            ("Tyler Cowen", "Tyler Cowen", {"person_type": "academic", "role": "professor", "org": "George Mason University", "expected_type": "professional", "expected_org": ""}),
            ("Esther Duflo", "Esther Duflo", {"person_type": "academic", "role": "professor", "org": "MIT", "expected_type": "professional", "expected_org": ""}),
            ("Abhijit Banerjee", "Abhijit Banerjee", {"person_type": "academic", "role": "professor", "org": "MIT", "expected_type": "professional", "expected_org": ""}),
            ("Jeffrey Sachs", "Jeffrey Sachs", {"person_type": "academic", "role": "professor", "org": "Columbia", "expected_type": "professional", "expected_org": ""}),
            # Scientists (merged from scientist type)
            ("Albert Einstein", "Albert Einstein", {"person_type": "academic", "role": "physicist", "org": "Princeton", "expected_org": ""}),
            ("Stephen Hawking", "Stephen Hawking", {"person_type": "academic", "role": "physicist", "org": "University of Cambridge", "expected_org": ""}),
            ("Marie Curie", "Marie Curie", {"person_type": "academic", "role": "physicist"}),
            ("Jennifer Doudna", "Jennifer Doudna", {"person_type": "academic", "role": "biochemist", "org": "UC Berkeley", "expected_org": ""}),
            ("Emmanuelle Charpentier", "Emmanuelle Charpentier", {"person_type": "academic", "role": "microbiologist"}),
            ("Katalin Karikó", "Katalin Karikó", {"person_type": "academic", "role": "biochemist", "org": "BioNTech", "expected_org": ""}),
            ("Demis Hassabis", "Demis Hassabis", {"person_type": "academic", "role": "AI researcher", "org": "DeepMind", "expected_type": "professional", "expected_org": "Google DeepMind"}),
            ("Geoffrey Hinton", "Geoffrey Hinton", {"person_type": "academic", "role": "computer scientist"}),
            ("Yann LeCun", "Yann Le Cun", {"person_type": "academic", "role": "AI researcher", "org": "Meta", "expected_org": ""}),
            ("Yoshua Bengio", "Yoshua Bengio", {"person_type": "academic", "role": "computer scientist", "org": "Mila"}),
            ("Andrew Ng", "Andrew Ng", {"person_type": "academic", "role": "computer scientist", "org": "Stanford", "expected_type": "academic", "expected_org": ""}),
            ("Fei-Fei Li", "Fei-Fei Li", {"person_type": "academic", "role": "computer scientist", "org": "Stanford", "expected_org": ""}),
            ("Neil deGrasse Tyson", "Neil deGrasse Tyson", {"person_type": "academic", "role": "astrophysicist", "org": "Hayden Planetarium", "expected_org": ""}),
            ("Jane Goodall", "Jane Goodall", {"person_type": "academic", "role": "primatologist"}),
            ("Francis Collins", "Francis Collins", {"person_type": "academic", "role": "geneticist", "org": "NIH", "expected_type": "professional", "expected_org": ""}),
            ("Kip Thorne", "Kip S. Thorne", {"person_type": "academic", "role": "physicist", "org": "Caltech", "expected_org": ""}),
            ("Roger Penrose", "Roger Penrose", {"person_type": "academic", "role": "mathematician", "org": "Oxford", "expected_org": ""}),
            ("Tu Youyou", "Tu Youyou", {"person_type": "academic", "role": "pharmacologist"}),
            ("James Watson", "James Watson", {"person_type": "academic", "role": "molecular biologist", "expected_type": "politician"}),
            ("Tim Berners-Lee", "Tim Berners-Lee", {"person_type": "academic", "role": "computer scientist"}),
        ],
        "artist": [
            ("Taylor Swift", "Taylor Swift", {"person_type": "artist", "role": "singer"}),
            ("Beyoncé", "Beyoncé", {"person_type": "artist", "role": "singer"}),
            ("Ed Sheeran", "Ed Sheeran", {"person_type": "artist", "role": "singer-songwriter"}),
            ("Adele", "Adele", {"person_type": "artist", "role": "singer"}),
            ("Drake", "Drake", {"person_type": "artist", "role": "rapper"}),
            ("Tom Hanks", "Tom Hanks", {"person_type": "artist", "role": "actor"}),
            ("Meryl Streep", "Meryl Streep", {"person_type": "artist", "role": "actress"}),
            ("Leonardo DiCaprio", "Leonardo DiCaprio", {"person_type": "artist", "role": "actor"}),
            ("Cate Blanchett", "Cate Blanchett", {"person_type": "artist", "role": "actress"}),
            ("Denzel Washington", "Denzel Washington", {"person_type": "artist", "role": "actor"}),
            ("Christopher Nolan", "Christopher Nolan", {"person_type": "artist", "role": "director"}),
            ("Martin Scorsese", "Martin Scorsese", {"person_type": "artist", "role": "director"}),
            ("Steven Spielberg", "Steven Spielberg", {"person_type": "artist", "role": "director"}),
            ("Banksy", "Banksy", {"person_type": "artist", "role": "street artist"}),
            ("Ai Weiwei", "Ai Weiwei", {"person_type": "artist", "role": "artist"}),
            ("Damien Hirst", "Damien Hirst", {"person_type": "artist", "role": "artist"}),
            ("J.K. Rowling", "Joanne K. Rowling", {"person_type": "artist", "role": "writer"}),
            ("Stephen King", "Stephen King", {"person_type": "artist", "role": "writer", "expected_type": "media"}),
            ("Haruki Murakami", "Haruki Murakami", {"person_type": "artist", "role": "novelist"}),
            ("Bob Dylan", "Bob Dylan", {"person_type": "artist", "role": "singer-songwriter"}),
        ],
        "media": [
            ("PewDiePie", "PewDiePie", {"person_type": "media", "role": "YouTuber"}),
            ("MrBeast", "MrBeast", {"person_type": "media", "role": "YouTuber"}),
            ("Joe Rogan", "Joe Rogan", {"person_type": "media", "role": "podcaster", "expected_type": "artist"}),
            ("Kim Kardashian", "Kim Kardashian", {"person_type": "media", "role": "reality TV star"}),
            ("Kylie Jenner", "Kylie Jenner", {"person_type": "media", "role": "influencer"}),
            ("Logan Paul", "Logan Paul", {"person_type": "media", "role": "YouTuber", "expected_type": "artist"}),
            ("Markiplier", "Markiplier", {"person_type": "media", "role": "YouTuber"}),
            ("Liza Koshy", "Liza Koshy", {"person_type": "media", "role": "YouTuber"}),
            ("Marques Brownlee", "Marques Brownlee", {"person_type": "media", "role": "tech reviewer"}),
            ("Emma Chamberlain", "Emma Chamberlain", {"person_type": "media", "role": "YouTuber"}),
            ("Casey Neistat", "Casey Neistat", {"person_type": "artist", "role": "YouTuber"}),
            ("Lilly Singh", "Lilly Singh", {"person_type": "media", "role": "YouTuber"}),
            ("David Dobrik", "David Dobrik", {"person_type": "media", "role": "YouTuber", "expected_type": "artist"}),
            ("Charli D'Amelio", "Charli D'Amelio", {"person_type": "artist", "role": "TikTok star"}),
            ("Addison Rae", "Addison Rae", {"person_type": "media", "role": "TikTok star", "expected_type": "artist"}),
            ("Ninja", "Ninja", {"person_type": "media", "role": "streamer"}),
            ("Pokimane", "Pokimane", {"person_type": "media", "role": "streamer"}),
            ("Linus Sebastian", "Linus Sebastian", {"person_type": "media", "role": "tech YouTuber", "expected_type": "executive"}),
            ("Philip DeFranco", "Philip DeFranco", {"person_type": "media", "role": "YouTuber"}),
            ("Rhett McLaughlin", "Rhett McLaughlin", {"person_type": "media", "role": "YouTuber"}),
        ],
        "athlete": [
            ("LeBron James", "LeBron James", {"person_type": "athlete", "role": "basketball player", "org": "Lakers", "expected_org": ""}),
            ("Lionel Messi", "Lionel Messi", {"person_type": "athlete", "role": "footballer"}),
            ("Cristiano Ronaldo", "Cristiano Ronaldo", {"person_type": "athlete", "role": "footballer"}),
            ("Serena Williams", "Serena Williams", {"person_type": "athlete", "role": "tennis player"}),
            ("Roger Federer", "Roger Federer", {"person_type": "athlete", "role": "tennis player"}),
            ("Novak Djokovic", "Novak Djokovic", {"person_type": "athlete", "role": "tennis player"}),
            ("Usain Bolt", "Usain Bolt", {"person_type": "athlete", "role": "sprinter"}),
            ("Michael Phelps", "Michael Phelps", {"person_type": "athlete", "role": "swimmer"}),
            ("Simone Biles", "Simone Biles", {"person_type": "athlete", "role": "gymnast", "expected_type": "unknown"}),
            ("Lewis Hamilton", "Lewis Hamilton", {"person_type": "athlete", "role": "racing driver", "org": "Mercedes", "expected_org": ""}),
            ("Max Verstappen", "Max Verstappen", {"person_type": "athlete", "role": "racing driver", "org": "Red Bull", "expected_type": "artist", "expected_org": ""}),
            ("Tom Brady", "Tom Brady", {"person_type": "athlete", "role": "quarterback"}),
            ("Patrick Mahomes", "Patrick Mahomes", {"person_type": "athlete", "role": "quarterback", "org": "Kansas City Chiefs", "expected_org": ""}),
            ("Kylian Mbappé", "Kylian Mbappé", {"person_type": "athlete", "role": "footballer"}),
            ("Erling Haaland", "Erling Haaland", {"person_type": "athlete", "role": "footballer", "org": "Manchester City", "expected_org": ""}),
            ("Stephen Curry", "Stephen Curry", {"person_type": "athlete", "role": "basketball player", "org": "Golden State Warriors", "expected_type": "athlete", "expected_org": ""}),
            ("Naomi Osaka", "Naomi Osaka", {"person_type": "athlete", "role": "tennis player"}),
            ("Katie Ledecky", "Katie Ledecky", {"person_type": "athlete", "role": "swimmer"}),
            ("Eliud Kipchoge", "Eliud Kipchoge", {"person_type": "athlete", "role": "marathon runner", "expected_type": "unknown"}),
            ("Neymar", "Neymar", {"person_type": "athlete", "role": "footballer"}),
        ],
        "journalist": [
            ("Anderson Cooper", "Anderson Cooper", {"person_type": "journalist", "role": "anchor", "org": "CNN", "expected_org": ""}),
            ("Christiane Amanpour", "Christiane Amanpour", {"person_type": "journalist", "role": "journalist", "org": "CNN", "expected_org": ""}),
            ("Bob Woodward", "Bob Woodward", {"person_type": "journalist", "role": "journalist", "org": "Washington Post", "expected_type": "artist", "expected_org": ""}),
            ("Kara Swisher", "Kara Swisher", {"person_type": "journalist", "role": "tech journalist"}),
            ("Tucker Carlson", "Tucker Carlson", {"person_type": "journalist", "role": "host", "org": "Fox News", "expected_org": ""}),
            ("Rachel Maddow", "Rachel Maddow", {"person_type": "journalist", "role": "host", "org": "MSNBC", "expected_org": ""}),
            ("Lester Holt", "Lester Holt", {"person_type": "journalist", "role": "anchor", "org": "NBC Nightly News", "expected_org": ""}),
            ("David Muir", "David Muir", {"person_type": "journalist", "role": "anchor", "org": "ABC World News Tonight", "expected_org": ""}),
            ("Norah O'Donnell", "Norah O'Donnell", {"person_type": "journalist", "role": "anchor", "org": "CBS Evening News", "expected_org": ""}),
            ("Wolf Blitzer", "Wolf Blitzer", {"person_type": "journalist", "role": "anchor", "org": "CNN", "expected_org": ""}),
            ("Fareed Zakaria", "Fareed Zakaria", {"person_type": "journalist", "role": "journalist", "org": "CNN", "expected_org": ""}),
            ("Maggie Haberman", "Maggie Haberman", {"person_type": "journalist", "role": "journalist", "org": "New York Times", "expected_org": ""}),
            ("Glenn Greenwald", "Glenn Greenwald", {"person_type": "journalist", "role": "journalist", "org": "The Intercept", "expected_type": "legal"}),
            ("Ronan Farrow", "Ronan Farrow", {"person_type": "journalist", "role": "journalist", "org": "The New Yorker", "expected_type": "legal", "expected_org": ""}),
            ("Savannah Guthrie", "Savannah Guthrie", {"person_type": "journalist", "role": "anchor", "org": "Today Show", "expected_org": ""}),
            ("Jake Tapper", "Jake Tapper", {"person_type": "journalist", "role": "anchor", "org": "CNN", "expected_org": ""}),
            ("Jorge Ramos", "Jorge Ramos", {"person_type": "journalist", "role": "anchor", "org": "Univision", "expected_org": ""}),
            ("Lesley Stahl", "Lesley Stahl", {"person_type": "journalist", "role": "correspondent", "org": "60 Minutes", "expected_org": ""}),
            ("Scott Pelley", "Scott Pelley", {"person_type": "journalist", "role": "correspondent", "org": "60 Minutes", "expected_org": ""}),
            ("Gayle King", "Gayle King", {"person_type": "journalist", "role": "anchor", "org": "CBS Mornings", "expected_type": "artist", "expected_org": ""}),
        ],
        "activist": [
            ("Greta Thunberg", "Greta Thunberg", {"person_type": "activist", "role": "climate activist"}),
            ("Malala Yousafzai", "Malala Yousafzai", {"person_type": "activist", "role": "education activist", "expected_type": "artist"}),
            ("Naomi Klein", "Naomi Klein", {"person_type": "activist", "role": "author", "expected_type": "journalist"}),
            ("Ai Weiwei", "Ai Weiwei", {"person_type": "activist", "role": "artist and activist", "expected_type": "artist"}),
            ("Desmond Tutu", "Desmond Tutu", {"person_type": "activist", "role": "archbishop", "expected_type": "unknown"}),
            ("Gloria Steinem", "Gloria Steinem", {"person_type": "activist", "role": "feminist activist", "expected_type": "journalist"}),
            ("Angela Davis", "Angela Davis", {"person_type": "activist", "role": "civil rights activist", "expected_type": "artist"}),
            ("Wangari Muta Maathai", "Wangari Muta Maathai", {"person_type": "politician", "role": "environmentalist"}),
            ("Vandana Shiva", "Vandana Shiva", {"person_type": "activist", "role": "environmental activist", "expected_type": "artist"}),
            ("Bryan Stevenson", "Bryan Stevenson", {"person_type": "activist", "role": "civil rights lawyer", "expected_type": "legal"}),
            ("Tarana Burke", "Tarana Burke", {"person_type": "activist", "role": "MeToo founder", "expected_type": "unknown"}),
            ("Patrisse Cullors", "Patrisse Khan-Cullors", {"person_type": "activist", "role": "BLM co-founder"}),
            ("Luisa Neubauer", "Luisa Neubauer", {"person_type": "media", "role": "climate activist"}),
            ("Joshua Wong", "Joshua Wong", {"person_type": "activist", "role": "pro-democracy activist", "expected_type": "politician"}),
            ("Alexei Navalny", "Alexei Navalny", {"person_type": "activist", "role": "opposition leader", "expected_type": "politician"}),
            ("Aung San Suu Kyi", "Aung San Suu Kyi", {"person_type": "politician", "role": "political leader"}),
            ("Nelson Mandela", "Nelson Mandela", {"person_type": "activist", "role": "anti-apartheid leader", "expected_type": "politician"}),
            ("Martin Luther King Jr", "Martin Luther King Jr.", {"person_type": "activist", "role": "civil rights leader", "expected_type": "professional"}),
            ("Rosa Parks", "Rosa Parks", {"person_type": "activist", "role": "civil rights activist", "expected_type": "unknown"}),
            ("Cesar Chavez", "Cesar Chavez", {"person_type": "activist", "role": "labor leader", "expected_type": "government"}),
        ],
    }

    # Filter to single type if requested
    if person_type_filter:
        person_type_filter = person_type_filter.lower()
        if person_type_filter not in test_queries_by_type:
            valid = ", ".join(sorted(test_queries_by_type.keys()))
            raise click.UsageError(f"Unknown type '{person_type_filter}'. Valid: {valid}")
        test_queries_by_type = {person_type_filter: test_queries_by_type[person_type_filter]}

    total_queries = sum(len(qs) for qs in test_queries_by_type.values())
    db_path_obj = _resolve_db_path(db_path)

    click.echo(
        f"Person search perf+accuracy test — "
        f"{len(test_queries_by_type)} types, {total_queries} queries, top_k={top_k}",
        err=True,
    )
    click.echo(f"Database: {db_path_obj}", err=True)

    database = get_person_database(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    # Track results per type
    type_stats: dict[str, dict[str, Any]] = {}
    all_timings: list[float] = []
    global_hits_at_1 = 0
    global_hits_in_topk = 0
    global_total = 0
    # For --for-llm: collect every non-top1 result for review
    llm_issues: list[dict[str, Any]] = []

    for ptype, queries in test_queries_by_type.items():
        if not for_llm:
            click.echo(f"\n{'=' * 80}", err=True)
            click.echo(f"  {ptype.upper()} ({len(queries)} queries)", err=True)
            click.echo(f"{'=' * 80}", err=True)

        hits_at_1 = 0
        hits_in_topk = 0
        type_timings: list[float] = []

        for i, (name, expected, query_kwargs) in enumerate(queries, 1):
            expected_lower = expected.lower()

            role = query_kwargs.get("role")
            org = query_kwargs.get("org")
            person_type = query_kwargs.get("person_type")
            query = f"{name}" + (f", {role}" if role else "") + (f" at {org}" if org else "")
            t0 = _time.perf_counter()
            query_embedding = embedder.embed_composite_person(name, role=role, org=org)

            t1 = _time.perf_counter()
            results = database.search(
                query_embedding, top_k=top_k,
                query_name=name,
                embedder=embedder,
                query_person_type=person_type,
                query_role=role,
                query_org=org,
            )

            total_elapsed = _time.perf_counter() - t0
            type_timings.append(total_elapsed)
            all_timings.append(total_elapsed)

            # Accuracy: check if expected person appears in results.
            # Match by name AND verify person_type/role/org to ensure it's the right person
            # (not a different person with the same name).
            # Checks both the canonical record and the matched_record (the actual
            # indexed record that triggered the match, which may have different role/org).
            def _is_expected(rec: "PersonRecord") -> bool:
                # Check canonical record and matched_record (if present)
                candidates = [rec]
                if rec.matched_record is not None:
                    candidates.append(rec.matched_record)

                for c in candidates:
                    if c.name.lower() != expected_lower:
                        continue
                    exp_type = query_kwargs.get("expected_type", person_type)
                    if exp_type and c.person_type.value != exp_type:
                        continue
                    exp_org = query_kwargs.get("expected_org", org)
                    if exp_org and exp_org.lower() not in c.known_for_org_name.lower():
                        continue
                    return True
                return False

            top1_match = False
            topk_match = False
            topk_rank = -1
            if results:
                if _is_expected(results[0][0]):
                    top1_match = True
                    topk_match = True
                    topk_rank = 1
                else:
                    for rank, (rec, _score) in enumerate(results, 1):
                        if _is_expected(rec):
                            topk_match = True
                            topk_rank = rank
                            break

            if top1_match:
                hits_at_1 += 1
            if topk_match:
                hits_in_topk += 1

            # Collect for --for-llm output (all non-top1 hits are worth reviewing)
            if for_llm and not top1_match:
                from corp_names import normalize_name as _corp_normalize_name

                top_results = [
                    {"rank": r, "name": rec.name,
                     "name_normalized": _corp_normalize_name(rec.name).normalized,
                     "score": round(sc, 4),
                     "person_type": rec.person_type.value, "role": rec.known_for_role, "org": rec.known_for_org_name}
                    for r, (rec, sc) in enumerate(results, 1)
                ]
                # DB diagnostics: check if expected person exists and is indexed
                db_matches = database._conn.execute(
                    "SELECT id, name FROM people WHERE LOWER(name) = LOWER(?) LIMIT 5",
                    (expected,),
                ).fetchall()
                db_like_matches = []
                if not db_matches:
                    db_like_matches_raw = database._conn.execute(
                        "SELECT id, name FROM people WHERE name LIKE ? LIMIT 5",
                        (f"%{expected}%",),
                    ).fetchall()
                    db_like_matches = [{"id": r["id"], "name": r["name"]} for r in db_like_matches_raw]
                in_composite = []
                for row in db_matches:
                    pid = row["id"]
                    in_composite.append(pid in database._hnsw_index if database._hnsw_index else False)

                issue: dict[str, Any] = {
                    "type": ptype,
                    "query": query,
                    "query_name_normalized": _corp_normalize_name(name).normalized,
                    "expected": expected,
                    "status": "wrong_rank" if topk_match else "missing",
                    "found_at_rank": topk_rank if topk_match else None,
                    "db_diagnostic": {
                        "exact_matches": [{"id": r["id"], "name": r["name"]} for r in db_matches],
                        "in_composite_index": in_composite,
                    },
                    "top_results": top_results,
                }
                if db_like_matches:
                    issue["db_diagnostic"]["like_matches"] = db_like_matches
                llm_issues.append(issue)

            # Display (skip in --for-llm mode)
            if not for_llm:
                hit_marker = "✓" if top1_match else ("~" if topk_match else "✗")
                top_name = results[0][0].name if results else "—"
                top_score = f"{results[0][1]:.3f}" if results else "—"
                rank_info = f"@{topk_rank}" if topk_match and not top1_match else ""

                click.echo(
                    f"  {hit_marker} {i:2d}. {total_elapsed * 1000:6.1f}ms  "
                    f"top: {top_name} ({top_score})  "
                    f"expect: {expected}{rank_info}",
                    err=True,
                )

        n = len(queries)
        acc1 = hits_at_1 / n * 100 if n else 0
        acck = hits_in_topk / n * 100 if n else 0
        mean_ms = sum(type_timings) / n * 1000 if n else 0
        type_stats[ptype] = {
            "n": n, "hits_at_1": hits_at_1, "hits_in_topk": hits_in_topk,
            "acc1": acc1, "acck": acck, "mean_ms": mean_ms,
        }
        global_hits_at_1 += hits_at_1
        global_hits_in_topk += hits_in_topk
        global_total += n

        if not for_llm:
            click.echo(
                f"  → {ptype}: acc@1={acc1:.0f}%  acc@{top_k}={acck:.0f}%  "
                f"mean={mean_ms:.1f}ms",
                err=True,
            )

    # Summary
    global_acc1 = global_hits_at_1 / global_total * 100 if global_total else 0
    global_acck = global_hits_in_topk / global_total * 100 if global_total else 0
    global_mean = sum(all_timings) / len(all_timings) * 1000 if all_timings else 0

    if not for_llm:
        click.echo(f"\n{'=' * 80}", err=True)
        click.echo("  SUMMARY", err=True)
        click.echo(f"{'=' * 80}", err=True)
        click.echo(f"  {'Type':<16s} {'N':>4s} {'Acc@1':>7s} {'Acc@k':>7s} {'Mean':>8s}", err=True)
        click.echo(f"  {'-' * 44}", err=True)
        for ptype, stats in type_stats.items():
            click.echo(
                f"  {ptype:<16s} {stats['n']:4d} "
                f"{stats['acc1']:6.0f}% {stats['acck']:6.0f}% "
                f"{stats['mean_ms']:7.1f}ms",
                err=True,
            )
        click.echo(f"  {'-' * 44}", err=True)
        click.echo(
            f"  {'TOTAL':<16s} {global_total:4d} "
            f"{global_acc1:6.1f}% {global_acck:6.1f}% "
            f"{global_mean:7.1f}ms",
            err=True,
        )
        click.echo(f"\n  Total time: {sum(all_timings):.2f}s  |  "
                   f"Min: {min(all_timings) * 1000:.1f}ms  |  "
                   f"Max: {max(all_timings) * 1000:.1f}ms", err=True)

    # --for-llm: write structured output to a temp file for LLM review
    if for_llm:
        import json as _json
        import tempfile
        n_missing = sum(1 for i in llm_issues if i["status"] == "missing")
        n_wrong_rank = sum(1 for i in llm_issues if i["status"] == "wrong_rank")
        llm_output = {
            "summary": {
                "total_queries": global_total,
                "acc_at_1": round(global_acc1, 1),
                "acc_at_k": round(global_acck, 1),
                "top_k": top_k,
                "mode": "embeddings-only",
                "failures": n_missing,
                "wrong_rank": n_wrong_rank,
            },
            "instructions": (
                "Review each issue below. Name matching uses exact equality (case-insensitive). "
                "Each issue includes query_name_normalized (the normalized query name used for "
                "SQL fallback) and db_diagnostic with: exact_matches (records whose name equals "
                "the expected value), like_matches (if no exact match, records containing the expected "
                "as a substring), and whether those records are in the composite HNSW index. "
                "Each result includes name_normalized showing what the Levenshtein fallback matches against. "
                "For 'missing' items: if db_diagnostic.exact_matches is empty, the expected name is "
                "wrong — check like_matches for the correct DB name and update the expected value. "
                "If exact_matches exist but in_composite_index is false, the record wasn't indexed. "
                "For 'wrong_rank' items: a different person with a similar name outranks the correct one. "
                "Propose fixes to the test_queries_by_type dict in search.py if the expected value "
                "is incorrect. Do NOT change the test if the search result is genuinely wrong."
            ),
            "issues": llm_issues,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="people-test-", delete=False,
        ) as f:
            _json.dump(llm_output, f, indent=2, ensure_ascii=False)
            f.write("\n")
            llm_path = f.name
        click.echo(f"LLM review output written to: {llm_path}", err=True)

    database.close()


@click.command("search")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--source", type=click.Choice(["gleif", "sec_edgar", "companies_house", "wikipedia"]), help="Filter by source")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search(query: str, db_path: Optional[str], top_k: int, source: Optional[str], verbose: bool):
    """
    Search for an organization in the database.

    \b
    Examples:
        corp-entity-db search "Apple Inc"
        corp-entity-db search "Microsoft" --source sec_edgar
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder

    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj)

    click.echo(f"Searching for '{query}'...", err=True)

    # Embed query
    query_embedding = embedder.embed(query)

    # Search
    results = database.search(query_embedding, top_k=top_k, source_filter=source)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop {len(results)} matches:")
    click.echo("-" * 60)

    for i, (record, similarity) in enumerate(results, 1):
        click.echo(f"{i}. {record.name}")
        click.echo(f"   Source: {record.source} | ID: {record.source_id}")
        click.echo(f"   Canonical ID: {record.canonical_id}")
        click.echo(f"   Similarity: {similarity:.4f}")
        if verbose and record.record:
            if record.record.get("ticker"):
                click.echo(f"   Ticker: {record.record['ticker']}")
            if record.record.get("jurisdiction"):
                click.echo(f"   Jurisdiction: {record.record['jurisdiction']}")
        click.echo()

    database.close()


@click.command("org-test")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of results per query")
@click.option("--type", "org_type_filter", type=str, help="Only test a single org type (e.g. 'business')")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_org_test(
    db_path: Optional[str], top_k: int, org_type_filter: Optional[str], verbose: bool,
):
    """
    Run organization searches across entity types to measure latency and accuracy.
    Each query has a known expected name — accuracy is scored by checking whether
    the expected organization appears in the top-k results.

    \b
    Examples:
        corp-entity-db org-test
        corp-entity-db org-test --type business --top-k 10
    """
    import time as _time

    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder

    test_queries_by_type: dict[str, list[tuple[str, str]]] = {
        "business": [
            ("Apple", "Apple Inc."),
            ("Microsoft", "Microsoft Corporation"),
            ("Amazon", "Amazon.com, Inc."),
            ("Google", "Alphabet Inc."),
            ("Tesla", "Tesla, Inc."),
            ("NVIDIA", "NVIDIA CORPORATION"),
            ("Meta", "Meta Platforms, Inc."),
            ("JPMorgan Chase", "JPMORGAN CHASE & CO"),
            ("Berkshire Hathaway", "Berkshire Hathaway Inc."),
            ("Samsung Electronics", "Samsung Electronics Co., Ltd."),
            ("Toyota", "Toyota Motor Corporation"),
            ("LVMH", "LVMH Moët Hennessy Louis Vuitton"),
            ("Nestlé", "Nestlé S.A."),
            ("Goldman Sachs", "The Goldman Sachs Group, Inc."),
            ("Shell", "Shell plc"),
            ("Pfizer", "PFIZER INC"),
            ("Intel", "Intel Corporation"),
            ("Adobe", "Adobe Inc."),
            ("Salesforce", "Salesforce, Inc."),
            ("Netflix", "Netflix, Inc."),
        ],
        "fund": [
            ("Vanguard S&P 500", "Vanguard S&P 500 ETF"),
            ("BlackRock", "BlackRock, Inc."),
            ("Fidelity", "Fidelity Investments"),
            ("PIMCO", "PIMCO"),
            ("Bridgewater Associates", "Bridgewater Associates"),
        ],
        "government": [
            ("SEC", "U.S. Securities and Exchange Commission"),
            ("FDA", "Food and Drug Administration"),
            ("Federal Reserve", "Board of Governors of the Federal Reserve System"),
            ("Bank of England", "Bank of England"),
            ("European Central Bank", "European Central Bank"),
        ],
        "educational": [
            ("MIT", "Massachusetts Institute of Technology"),
            ("Stanford", "Stanford University"),
            ("Harvard", "Harvard University"),
            ("Oxford", "University of Oxford"),
            ("Cambridge", "University of Cambridge"),
        ],
        "international_org": [
            ("United Nations", "United Nations"),
            ("World Bank", "World Bank"),
            ("IMF", "International Monetary Fund"),
            ("WHO", "World Health Organization"),
            ("NATO", "NATO"),
        ],
        "nonprofit": [
            ("Red Cross", "International Committee of the Red Cross"),
            ("Wikimedia Foundation", "Wikimedia Foundation, Inc."),
            ("Mozilla", "Mozilla Foundation"),
            ("Wikipedia", "Wikimedia Foundation, Inc."),
            ("Linux Foundation", "The Linux Foundation"),
        ],
    }

    if org_type_filter:
        org_type_filter = org_type_filter.lower()
        if org_type_filter not in test_queries_by_type:
            valid = ", ".join(sorted(test_queries_by_type.keys()))
            raise click.UsageError(f"Unknown type '{org_type_filter}'. Valid: {valid}")
        test_queries_by_type = {org_type_filter: test_queries_by_type[org_type_filter]}

    total_queries = sum(len(qs) for qs in test_queries_by_type.values())
    db_path_obj = _resolve_db_path(db_path)

    click.echo(
        f"Org search test — {len(test_queries_by_type)} types, "
        f"{total_queries} queries, top_k={top_k}",
        err=True,
    )
    click.echo(f"Database: {db_path_obj}", err=True)

    database = OrganizationDatabase(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    type_stats: dict[str, dict[str, Any]] = {}
    all_timings: list[float] = []
    global_hits_at_1 = 0
    global_hits_in_topk = 0
    global_total = 0

    for otype, queries in test_queries_by_type.items():
        click.echo(f"\n{'=' * 80}", err=True)
        click.echo(f"  {otype.upper()} ({len(queries)} queries)", err=True)
        click.echo(f"{'=' * 80}", err=True)

        hits_at_1 = 0
        hits_in_topk = 0
        type_timings: list[float] = []

        for i, (query, expected) in enumerate(queries, 1):
            expected_lower = expected.lower()

            t0 = _time.perf_counter()
            query_embedding = embedder.embed(query)
            results = database.search(query_embedding, top_k=top_k)
            total_elapsed = _time.perf_counter() - t0

            type_timings.append(total_elapsed)
            all_timings.append(total_elapsed)

            top1_match = False
            topk_match = False
            topk_rank = -1
            if results:
                if results[0][0].name.lower() == expected_lower:
                    top1_match = True
                    topk_match = True
                    topk_rank = 1
                else:
                    for rank, (rec, _score) in enumerate(results, 1):
                        if rec.name.lower() == expected_lower:
                            topk_match = True
                            topk_rank = rank
                            break

            if top1_match:
                hits_at_1 += 1
            if topk_match:
                hits_in_topk += 1

            hit_marker = "✓" if top1_match else ("~" if topk_match else "✗")
            top_name = results[0][0].name if results else "—"
            top_score = f"{results[0][1]:.3f}" if results else "—"
            rank_info = f"@{topk_rank}" if topk_match and not top1_match else ""

            click.echo(
                f"  {hit_marker} {i:2d}. {total_elapsed * 1000:6.1f}ms  "
                f"top: {top_name} ({top_score})  "
                f"expect: {expected}{rank_info}",
                err=True,
            )

        n = len(queries)
        acc1 = hits_at_1 / n * 100 if n else 0
        acck = hits_in_topk / n * 100 if n else 0
        mean_ms = sum(type_timings) / n * 1000 if n else 0
        type_stats[otype] = {
            "n": n, "hits_at_1": hits_at_1, "hits_in_topk": hits_in_topk,
            "acc1": acc1, "acck": acck, "mean_ms": mean_ms,
        }
        global_hits_at_1 += hits_at_1
        global_hits_in_topk += hits_in_topk
        global_total += n

        click.echo(
            f"  → {otype}: acc@1={acc1:.0f}%  acc@{top_k}={acck:.0f}%  "
            f"mean={mean_ms:.1f}ms",
            err=True,
        )

    # Summary
    click.echo(f"\n{'=' * 80}", err=True)
    click.echo("  SUMMARY", err=True)
    click.echo(f"{'=' * 80}", err=True)
    click.echo(f"  {'Type':<20s} {'N':>4s} {'Acc@1':>7s} {'Acc@k':>7s} {'Mean':>8s}", err=True)
    click.echo(f"  {'-' * 48}", err=True)
    for otype, stats in type_stats.items():
        click.echo(
            f"  {otype:<20s} {stats['n']:4d} "
            f"{stats['acc1']:6.0f}% {stats['acck']:6.0f}% "
            f"{stats['mean_ms']:7.1f}ms",
            err=True,
        )
    click.echo(f"  {'-' * 48}", err=True)
    global_acc1 = global_hits_at_1 / global_total * 100 if global_total else 0
    global_acck = global_hits_in_topk / global_total * 100 if global_total else 0
    global_mean = sum(all_timings) / len(all_timings) * 1000 if all_timings else 0
    click.echo(
        f"  {'TOTAL':<20s} {global_total:4d} "
        f"{global_acc1:6.1f}% {global_acck:6.1f}% "
        f"{global_mean:7.1f}ms",
        err=True,
    )
    click.echo(f"\n  Total time: {sum(all_timings):.2f}s  |  "
               f"Min: {min(all_timings) * 1000:.1f}ms  |  "
               f"Max: {max(all_timings) * 1000:.1f}ms", err=True)

    database.close()


@click.command("search-roles")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--limit", default=10, help="Maximum results to return")
def db_search_roles(query: str, db_path: Optional[str], limit: int):
    """
    Search for roles by name.

    \b
    Examples:
        corp-entity-db search-roles "CEO"
        corp-entity-db search-roles "Chief Executive" --limit 5
    """
    from corp_entity_db.store import get_roles_database

    roles_db = get_roles_database(db_path)
    results = roles_db.search(query, top_k=limit)

    if not results:
        click.echo(f"No roles found matching '{query}'")
        return

    click.echo(f"Found {len(results)} role(s) matching '{query}':")
    for role_id, name, score in results:
        click.echo(f"  [{role_id}] {name} (score: {score:.2f})")


@click.command("search-locations")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--type", "location_type", type=str, help="Filter by simplified type (country, city, etc.)")
@click.option("--limit", default=10, help="Maximum results to return")
def db_search_locations(query: str, db_path: Optional[str], location_type: Optional[str], limit: int):
    """
    Search for locations by name.

    \b
    Examples:
        corp-entity-db search-locations "California"
        corp-entity-db search-locations "Paris" --type city
        corp-entity-db search-locations "Germany" --type country
    """
    from corp_entity_db.store import get_locations_database

    locations_db = get_locations_database(db_path)
    results = locations_db.search(query, top_k=limit, simplified_type=location_type)

    if not results:
        click.echo(f"No locations found matching '{query}'")
        return

    click.echo(f"Found {len(results)} location(s) matching '{query}':")
    for loc_id, name, score in results:
        click.echo(f"  [{loc_id}] {name} (score: {score:.2f})")
