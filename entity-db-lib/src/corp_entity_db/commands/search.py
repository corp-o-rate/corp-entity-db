"""Database search commands."""

from typing import Any, Optional

import click

from ._common import _configure_logging, _resolve_db_path


@click.command("search-people")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search (default is embeddings-only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search_people(query: str, db_path: Optional[str], top_k: int, hybrid: bool, verbose: bool):
    """
    Search for a person in the database.

    \b
    Examples:
        corp-entity-db search-people "Tim Cook"
        corp-entity-db search-people "Elon Musk" --top-k 5
        corp-entity-db search-people "Elon Musk" --hybrid
    """
    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database
    from corp_entity_db.embeddings import CompanyEmbedder

    # Default database path
    db_path_obj = _resolve_db_path(db_path)

    mode = "hybrid (text + embeddings)" if hybrid else "embeddings-only"
    click.echo(f"Searching for '{query}' in {db_path_obj} [{mode}]...", err=True)

    # Initialize components
    database = get_person_database(db_path=db_path_obj)
    embedder = CompanyEmbedder()

    # Embed query and search
    query_embedding = embedder.embed(query)
    query_text = query if hybrid else None
    results = database.search(query_embedding, top_k=top_k, query_text=query_text)

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


@click.command("search-people-perf-test")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=5, help="Number of results per query")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search")
@click.option("--type", "person_type_filter", type=str, help="Only test a single person type (e.g. 'executive')")
@click.option("--for-llm", is_flag=True, help="Output structured results for LLM review (failures + ambiguous matches)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search_people_perf_test(
    db_path: Optional[str], top_k: int, hybrid: bool, person_type_filter: Optional[str],
    for_llm: bool, verbose: bool,
):
    """
    Run person searches across all PersonType categories (20 per type) to measure
    latency and accuracy. Each query has a known expected name — accuracy is scored
    by checking whether the expected person appears in the top-k results.

    \b
    Examples:
        corp-entity-db search-people-perf-test
        corp-entity-db search-people-perf-test --hybrid
        corp-entity-db search-people-perf-test --type politician
    """
    import time as _time

    _configure_logging(verbose)

    from corp_entity_db.store import get_person_database, format_person_query
    from corp_entity_db.embeddings import CompanyEmbedder

    # Each entry: (name, expected_name_substring, query_kwargs)
    # query_kwargs are passed to format_person_query (person_type, role, org).
    test_queries_by_type: dict[str, list[tuple[str, str, dict[str, str]]]] = {
        "executive": [
            ("Tim Cook", "Tim Cook", {"person_type": "executive", "role": "CEO", "org": "Apple"}),
            ("Satya Nadella", "Satya Nadella", {"person_type": "executive", "role": "CEO", "org": "Microsoft"}),
            ("Andy Jassy", "Andy Jassy", {"person_type": "executive", "role": "CEO", "org": "Amazon"}),
            ("Sundar Pichai", "Sundar Pichai", {"person_type": "executive", "role": "CEO", "org": "Google"}),
            ("Jensen Huang", "Jensen Huang", {"person_type": "executive", "role": "CEO", "org": "NVIDIA"}),
            ("Mark Zuckerberg", "Mark Zuckerberg", {"person_type": "executive", "role": "CEO", "org": "Meta"}),
            ("Jamie Dimon", "Jamie Dimon", {"person_type": "executive", "role": "CEO", "org": "JPMorgan Chase"}),
            ("Warren Buffett", "Warren Buffett", {"person_type": "executive", "role": "CEO", "org": "Berkshire Hathaway"}),
            ("Elon Musk", "Elon Musk", {"person_type": "executive", "role": "CEO", "org": "Tesla"}),
            ("Lisa Su", "Lisa Su", {"person_type": "executive", "role": "CEO", "org": "AMD"}),
            ("Mary Barra", "Mary Barra", {"person_type": "executive", "role": "CEO", "org": "General Motors"}),
            ("David Solomon", "Solomon", {"person_type": "executive", "role": "CEO", "org": "Goldman Sachs"}),
            ("Brian Moynihan", "Brian Moynihan", {"person_type": "executive", "role": "CEO", "org": "Bank of America"}),
            ("Arvind Krishna", "Arvind Krishna", {"person_type": "executive", "role": "CEO", "org": "IBM"}),
            ("Pat Gelsinger", "Pat Gelsinger", {"person_type": "executive", "role": "CEO", "org": "Intel"}),
            ("Chuck Robbins", "Chuck Robbins", {"person_type": "executive", "role": "CEO", "org": "Cisco"}),
            ("Safra Catz", "Safra Catz", {"person_type": "executive", "role": "CEO", "org": "Oracle"}),
            ("Shantanu Narayen", "Shantanu Narayen", {"person_type": "executive", "role": "CEO", "org": "Adobe"}),
            ("Marc Benioff", "Marc Benioff", {"person_type": "executive", "role": "CEO", "org": "Salesforce"}),
            ("Darius Adamczyk", "Darius Adamczyk", {"person_type": "executive", "role": "CEO", "org": "Honeywell"}),
        ],
        "politician": [
            ("Joe Biden", "Joe Biden", {"person_type": "politician", "role": "President", "org": "United States"}),
            ("Donald Trump", "Donald Trump", {"person_type": "politician", "role": "President", "org": "United States"}),
            ("Emmanuel Macron", "Emmanuel Macron", {"person_type": "politician", "role": "President", "org": "France"}),
            ("Rishi Sunak", "Rishi Sunak", {"person_type": "politician", "role": "Prime Minister", "org": "United Kingdom"}),
            ("Olaf Scholz", "Olaf Scholz", {"person_type": "politician", "role": "Chancellor", "org": "Germany"}),
            ("Justin Trudeau", "Justin Trudeau", {"person_type": "politician", "role": "Prime Minister", "org": "Canada"}),
            ("Narendra Modi", "Narendra Modi", {"person_type": "politician", "role": "Prime Minister", "org": "India"}),
            ("Fumio Kishida", "Fumio Kishida", {"person_type": "politician", "role": "Prime Minister", "org": "Japan"}),
            ("Volodymyr Zelenskyy", "Zelenskyy", {"person_type": "politician", "role": "President", "org": "Ukraine"}),
            ("Pedro Sánchez", "Pedro S", {"person_type": "politician", "role": "Prime Minister", "org": "Spain"}),
            ("Anthony Albanese", "Albanese", {"person_type": "politician", "role": "Prime Minister", "org": "Australia"}),
            ("Giorgia Meloni", "Meloni", {"person_type": "politician", "role": "Prime Minister", "org": "Italy"}),
            ("Lula da Silva", "Lula", {"person_type": "politician", "role": "President", "org": "Brazil"}),
            ("Jacinda Ardern", "Ardern", {"person_type": "politician", "role": "Prime Minister", "org": "New Zealand"}),
            ("Xi Jinping", "Xi Jinping", {"person_type": "politician", "role": "President", "org": "China"}),
            ("Recep Tayyip Erdogan", "Erdo", {"person_type": "politician", "role": "President", "org": "Turkey"}),
            ("Benjamin Netanyahu", "Netanyahu", {"person_type": "politician", "role": "Prime Minister", "org": "Israel"}),
            ("Yoon Suk-yeol", "Yoon", {"person_type": "politician", "role": "President", "org": "South Korea"}),
            ("Mark Rutte", "Rutte", {"person_type": "politician", "role": "Prime Minister", "org": "Netherlands"}),
            ("Keir Starmer", "Starmer", {"person_type": "politician", "role": "Prime Minister", "org": "United Kingdom"}),
        ],
        "government": [
            ("Janet Yellen", "Yellen", {"person_type": "government", "role": "Secretary of the Treasury", "org": "United States"}),
            ("Antony Blinken", "Blinken", {"person_type": "government", "role": "Secretary of State", "org": "United States"}),
            ("Jerome Powell", "Powell", {"person_type": "government", "role": "Chair", "org": "Federal Reserve"}),
            ("Christine Lagarde", "Lagarde", {"person_type": "government", "role": "President", "org": "European Central Bank"}),
            ("Ursula von der Leyen", "von der Leyen", {"person_type": "government", "role": "President", "org": "European Commission"}),
            ("António Guterres", "Guterres", {"person_type": "government", "role": "Secretary-General", "org": "United Nations"}),
            ("Tedros Adhanom", "Tedros", {"person_type": "government", "role": "Director-General", "org": "WHO"}),
            ("Kristalina Georgieva", "Georgieva", {"person_type": "government", "role": "Managing Director", "org": "IMF"}),
            ("Gary Gensler", "Gensler", {"person_type": "government", "role": "Chair", "org": "SEC"}),
            ("Merrick Garland", "Garland", {"person_type": "government", "role": "Attorney General", "org": "United States"}),
            ("Andrew Bailey", "Bailey", {"person_type": "government", "role": "Governor", "org": "Bank of England"}),
            ("Haruhiko Kuroda", "Kuroda", {"person_type": "government", "role": "Governor", "org": "Bank of Japan"}),
            ("Lloyd Austin", "Austin", {"person_type": "government", "role": "Secretary of Defense", "org": "United States"}),
            ("Avril Haines", "Haines", {"person_type": "government", "role": "Director of National Intelligence", "org": "United States"}),
            ("Alejandro Mayorkas", "Mayorkas", {"person_type": "government", "role": "Secretary", "org": "DHS"}),
            ("Gina Raimondo", "Raimondo", {"person_type": "government", "role": "Secretary of Commerce", "org": "United States"}),
            ("Janet Woodcock", "Woodcock", {"person_type": "government", "role": "Commissioner", "org": "FDA"}),
            ("Ajay Banga", "Banga", {"person_type": "government", "role": "President", "org": "World Bank"}),
            ("Ngozi Okonjo-Iweala", "Okonjo", {"person_type": "government", "role": "Director-General", "org": "WTO"}),
            ("Charles Michel", "Michel", {"person_type": "government", "role": "President", "org": "European Council"}),
        ],
        "military": [
            ("Mark Milley", "Milley", {"person_type": "military", "role": "Chairman Joint Chiefs of Staff", "org": "United States"}),
            ("Valerii Zaluzhnyi", "Zaluzhnyi", {"person_type": "military", "role": "Commander-in-Chief", "org": "Ukraine Armed Forces"}),
            ("Tony Radakin", "Radakin", {"person_type": "military", "role": "Chief of Defence Staff", "org": "United Kingdom"}),
            ("Thierry Burkhard", "Burkhard", {"person_type": "military", "role": "Chief of Defence Staff", "org": "France"}),
            ("Rob Bauer", "Bauer", {"person_type": "military", "role": "Chair", "org": "NATO Military Committee"}),
            ("Christopher Cavoli", "Cavoli", {"person_type": "military", "role": "SACEUR", "org": "NATO"}),
            ("Michael Kurilla", "Kurilla", {"person_type": "military", "role": "Commander", "org": "CENTCOM"}),
            ("Charles Brown", "Charles E. Brown", {"person_type": "military", "role": "Chairman Joint Chiefs of Staff", "org": "United States"}),
            ("Eberhard Zorn", "Zorn", {"person_type": "military", "role": "Inspector General", "org": "Germany Bundeswehr"}),
            ("Koji Yamazaki", "Yamazaki", {"person_type": "military", "role": "Chief of Staff", "org": "Japan Self-Defense Forces"}),
            ("Angus Campbell", "Campbell", {"person_type": "military", "role": "Chief of Defence Force", "org": "Australia"}),
            ("Bipin Rawat", "Rawat", {"person_type": "military", "role": "Chief of Defence Staff", "org": "India"}),
            ("Wayne Eyre", "Eyre", {"person_type": "military", "role": "Chief of Defence Staff", "org": "Canada"}),
            ("Sergei Shoigu", "Shoigu", {"person_type": "military", "role": "Minister of Defence", "org": "Russia"}),
            ("James Hecker", "Hecker", {"person_type": "military", "role": "Commander", "org": "US Air Forces in Europe"}),
            ("Samuel Paparo", "Paparo", {"person_type": "military", "role": "Commander", "org": "US Indo-Pacific Command"}),
            ("Laura Richardson", "Richardson", {"person_type": "military", "role": "Commander", "org": "US Southern Command"}),
            ("Mauro Del Vecchio", "Del Vecchio", {"person_type": "military", "role": "Commander", "org": "NATO Joint Force Command"}),
            ("Stuart Peach", "Peach", {"person_type": "military", "role": "Chair", "org": "NATO Military Committee"}),
            ("Eirik Kristoffersen", "Kristoffersen", {"person_type": "military", "role": "Chief of Defence", "org": "Norway"}),
        ],
        "legal": [
            ("John Roberts", "Roberts", {"person_type": "legal", "role": "Chief Justice", "org": "Supreme Court of the United States"}),
            ("Sonia Sotomayor", "Sotomayor", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Ketanji Brown Jackson", "Ketanji", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Clarence Thomas", "Clarence Thomas", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Elena Kagan", "Kagan", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Neil Gorsuch", "Gorsuch", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Brett Kavanaugh", "Kavanaugh", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Amy Coney Barrett", "Barrett", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Samuel Alito", "Alito", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Ruth Bader Ginsburg", "Ginsburg", {"person_type": "legal", "role": "Justice", "org": "Supreme Court of the United States"}),
            ("Brenda Hale", "Hale", {"person_type": "legal", "role": "President", "org": "Supreme Court, United Kingdom"}),
            ("Robert Reed", "Reed", {"person_type": "legal", "role": "President", "org": "UK Supreme Court"}),
            ("Karim Khan", "Khan", {"person_type": "legal", "role": "Prosecutor", "org": "International Criminal Court"}),
            ("Joan Donoghue", "Donoghue", {"person_type": "legal", "role": "President", "org": "International Court of Justice"}),
            ("Didier Reynders", "Reynders", {"person_type": "legal", "role": "Commissioner for Justice", "org": "European Commission"}),
            ("Fatou Bensouda", "Bensouda", {"person_type": "legal", "role": "Prosecutor", "org": "International Criminal Court"}),
            ("Loretta Lynch", "Lynch", {"person_type": "legal", "role": "Attorney General", "org": "United States"}),
            ("Eric Holder", "Holder", {"person_type": "legal", "role": "Attorney General", "org": "United States"}),
            ("Robert Mueller", "Mueller", {"person_type": "legal", "role": "Special Counsel", "org": "Department of Justice"}),
            ("Jack Smith", "Jack Smith", {"person_type": "legal", "role": "Special Counsel", "org": "Department of Justice"}),
        ],
        "professional": [
            ("Atul Gawande", "Gawande", {"person_type": "professional", "role": "surgeon", "org": "Brigham and Women's Hospital"}),
            ("Sanjay Gupta", "Sanjay Gupta", {"person_type": "professional", "role": "neurosurgeon", "org": "Emory University Hospital"}),
            ("Anthony Fauci", "Fauci", {"person_type": "professional", "role": "immunologist", "org": "NIAID"}),
            ("Devi Shetty", "Shetty", {"person_type": "professional", "role": "cardiac surgeon", "org": "Narayana Health"}),
            ("Norman Foster", "Norman Foster", {"person_type": "professional", "role": "architect", "org": "Foster + Partners"}),
            ("Bjarke Ingels", "Bjarke Ingels", {"person_type": "professional", "role": "architect", "org": "BIG"}),
            ("Zaha Hadid", "Zaha Hadid", {"person_type": "professional", "role": "architect", "org": "Zaha Hadid Architects"}),
            ("Renzo Piano", "Renzo Piano", {"person_type": "professional", "role": "architect", "org": "Renzo Piano Building Workshop"}),
            ("Frank Gehry", "Gehry", {"person_type": "professional", "role": "architect", "org": "Gehry Partners"}),
            ("Tadao Ando", "Tadao Ando", {"person_type": "professional", "role": "architect"}),
            ("I. M. Pei", "Pei", {"person_type": "professional", "role": "architect", "org": "Pei Cobb Freed"}),
            ("Santiago Calatrava", "Calatrava", {"person_type": "professional", "role": "architect"}),
            ("Rem Koolhaas", "Koolhaas", {"person_type": "professional", "role": "architect", "org": "OMA"}),
            ("Thomas Heatherwick", "Heatherwick", {"person_type": "professional", "role": "designer", "org": "Heatherwick Studio"}),
            ("Jony Ive", "Ive", {"person_type": "professional", "role": "designer", "org": "Apple"}),
            ("Dieter Rams", "Dieter Rams", {"person_type": "professional", "role": "designer", "org": "Braun"}),
            ("Philippe Starck", "Starck", {"person_type": "professional", "role": "designer"}),
            ("Toyo Ito", "Toyo Ito", {"person_type": "professional", "role": "architect"}),
            ("Daniel Libeskind", "Libeskind", {"person_type": "professional", "role": "architect"}),
            ("Peter Zumthor", "Zumthor", {"person_type": "professional", "role": "architect"}),
        ],
        "academic": [
            ("Noam Chomsky", "Chomsky", {"person_type": "academic", "role": "professor", "org": "MIT"}),
            ("Steven Pinker", "Pinker", {"person_type": "academic", "role": "professor", "org": "Harvard"}),
            ("Paul Krugman", "Krugman", {"person_type": "academic", "role": "professor", "org": "Princeton"}),
            ("Joseph Stiglitz", "Stiglitz", {"person_type": "academic", "role": "professor", "org": "Columbia"}),
            ("Thomas Piketty", "Piketty", {"person_type": "academic", "role": "professor", "org": "Paris School of Economics"}),
            ("Yuval Noah Harari", "Harari", {"person_type": "academic", "role": "professor", "org": "Hebrew University"}),
            ("Niall Ferguson", "Ferguson", {"person_type": "academic", "role": "professor", "org": "Stanford"}),
            ("Lawrence Lessig", "Lessig", {"person_type": "academic", "role": "professor", "org": "Harvard Law School"}),
            ("Cornel West", "Cornel West", {"person_type": "academic", "role": "professor", "org": "Union Theological Seminary"}),
            ("Nassim Nicholas Taleb", "Taleb", {"person_type": "academic", "role": "professor", "org": "NYU"}),
            ("Jordan Peterson", "Peterson", {"person_type": "academic", "role": "professor", "org": "University of Toronto"}),
            ("Richard Dawkins", "Dawkins", {"person_type": "academic", "role": "professor", "org": "Oxford"}),
            ("Amy Cuddy", "Cuddy", {"person_type": "academic", "role": "professor", "org": "Harvard Business School"}),
            ("Brené Brown", "Brown", {"person_type": "academic", "role": "professor", "org": "University of Houston"}),
            ("Henry Kissinger", "Kissinger", {"person_type": "academic", "role": "professor", "org": "Georgetown"}),
            ("Daron Acemoglu", "Acemo", {"person_type": "academic", "role": "professor", "org": "MIT"}),
            ("Tyler Cowen", "Cowen", {"person_type": "academic", "role": "professor", "org": "George Mason University"}),
            ("Esther Duflo", "Duflo", {"person_type": "academic", "role": "professor", "org": "MIT"}),
            ("Abhijit Banerjee", "Banerjee", {"person_type": "academic", "role": "professor", "org": "MIT"}),
            ("Jeffrey Sachs", "Sachs", {"person_type": "academic", "role": "professor", "org": "Columbia"}),
        ],
        "artist": [
            ("Taylor Swift", "Taylor Swift", {"person_type": "artist"}),
            ("Beyoncé", "Beyonc", {"person_type": "artist"}),
            ("Ed Sheeran", "Ed Sheeran", {"person_type": "artist"}),
            ("Adele", "Adele", {"person_type": "artist"}),
            ("Drake", "Drake", {"person_type": "artist"}),
            ("Tom Hanks", "Tom Hanks", {"person_type": "artist"}),
            ("Meryl Streep", "Meryl Streep", {"person_type": "artist"}),
            ("Leonardo DiCaprio", "Leonardo DiCaprio", {"person_type": "artist"}),
            ("Cate Blanchett", "Cate Blanchett", {"person_type": "artist"}),
            ("Denzel Washington", "Denzel Washington", {"person_type": "artist"}),
            ("Christopher Nolan", "Christopher Nolan", {"person_type": "artist"}),
            ("Martin Scorsese", "Martin Scorsese", {"person_type": "artist"}),
            ("Steven Spielberg", "Spielberg", {"person_type": "artist"}),
            ("Banksy", "Banksy", {"person_type": "artist"}),
            ("Ai Weiwei", "Ai Weiwei", {"person_type": "artist"}),
            ("Damien Hirst", "Damien Hirst", {"person_type": "artist"}),
            ("J.K. Rowling", "Rowling", {"person_type": "artist"}),
            ("Stephen King", "Stephen King", {"person_type": "artist"}),
            ("Haruki Murakami", "Murakami", {"person_type": "artist"}),
            ("Bob Dylan", "Bob Dylan", {"person_type": "artist"}),
        ],
        "media": [
            ("PewDiePie", "PewDiePie", {"person_type": "media"}),
            ("MrBeast", "MrBeast", {"person_type": "media"}),
            ("Joe Rogan", "Joe Rogan", {"person_type": "media"}),
            ("Kim Kardashian", "Kim Kardashian", {"person_type": "media"}),
            ("Kylie Jenner", "Kylie Jenner", {"person_type": "media"}),
            ("Logan Paul", "Logan Paul", {"person_type": "media"}),
            ("Markiplier", "Markiplier", {"person_type": "media"}),
            ("Hasan Piker", "Hasan", {"person_type": "media"}),
            ("Marques Brownlee", "Marques Brownlee", {"person_type": "media"}),
            ("Emma Chamberlain", "Emma Chamberlain", {"person_type": "media"}),
            ("Casey Neistat", "Casey Neistat", {"person_type": "media"}),
            ("Lilly Singh", "Lilly Singh", {"person_type": "media"}),
            ("David Dobrik", "David Dobrik", {"person_type": "media"}),
            ("Charli D'Amelio", "Charli", {"person_type": "media"}),
            ("Addison Rae", "Addison Rae", {"person_type": "media"}),
            ("Ninja", "Ninja", {"person_type": "media"}),
            ("Pokimane", "Pokimane", {"person_type": "media"}),
            ("Linus Sebastian", "Linus", {"person_type": "media"}),
            ("Philip DeFranco", "Philip DeFranco", {"person_type": "media"}),
            ("Rhett McLaughlin", "Rhett", {"person_type": "media"}),
        ],
        "athlete": [
            ("LeBron James", "LeBron James", {"person_type": "athlete"}),
            ("Lionel Messi", "Lionel Messi", {"person_type": "athlete"}),
            ("Cristiano Ronaldo", "Cristiano Ronaldo", {"person_type": "athlete"}),
            ("Serena Williams", "Serena Williams", {"person_type": "athlete"}),
            ("Roger Federer", "Roger Federer", {"person_type": "athlete"}),
            ("Novak Djokovic", "Novak Djokovic", {"person_type": "athlete"}),
            ("Usain Bolt", "Usain Bolt", {"person_type": "athlete"}),
            ("Michael Phelps", "Michael Phelps", {"person_type": "athlete"}),
            ("Simone Biles", "Simone Biles", {"person_type": "athlete"}),
            ("Lewis Hamilton", "Lewis Hamilton", {"person_type": "athlete"}),
            ("Max Verstappen", "Max Verstappen", {"person_type": "athlete"}),
            ("Tom Brady", "Tom Brady", {"person_type": "athlete"}),
            ("Patrick Mahomes", "Mahomes", {"person_type": "athlete"}),
            ("Kylian Mbappé", "Mbapp", {"person_type": "athlete"}),
            ("Erling Haaland", "Haaland", {"person_type": "athlete"}),
            ("Stephen Curry", "Stephen Curry", {"person_type": "athlete"}),
            ("Naomi Osaka", "Naomi Osaka", {"person_type": "athlete"}),
            ("Katie Ledecky", "Ledecky", {"person_type": "athlete"}),
            ("Eliud Kipchoge", "Kipchoge", {"person_type": "athlete"}),
            ("Neymar", "Neymar", {"person_type": "athlete"}),
        ],
        "entrepreneur": [
            ("Jeff Bezos", "Jeff Bezos", {"person_type": "entrepreneur"}),
            ("Bill Gates", "Bill Gates", {"person_type": "entrepreneur"}),
            ("Larry Page", "Larry Page", {"person_type": "entrepreneur"}),
            ("Sergey Brin", "Sergey Brin", {"person_type": "entrepreneur"}),
            ("Jack Dorsey", "Jack Dorsey", {"person_type": "entrepreneur"}),
            ("Reid Hoffman", "Reid Hoffman", {"person_type": "entrepreneur"}),
            ("Peter Thiel", "Peter Thiel", {"person_type": "entrepreneur"}),
            ("Travis Kalanick", "Travis Kalanick", {"person_type": "entrepreneur"}),
            ("Brian Chesky", "Brian Chesky", {"person_type": "entrepreneur"}),
            ("Jack Ma", "Jack Ma", {"person_type": "entrepreneur"}),
            ("Richard Branson", "Richard Branson", {"person_type": "entrepreneur"}),
            ("Sam Altman", "Sam Altman", {"person_type": "entrepreneur"}),
            ("Evan Spiegel", "Evan Spiegel", {"person_type": "entrepreneur"}),
            ("Daniel Ek", "Daniel Ek", {"person_type": "entrepreneur"}),
            ("Patrick Collison", "Patrick Collison", {"person_type": "entrepreneur"}),
            ("Whitney Wolfe Herd", "Whitney Wolfe", {"person_type": "entrepreneur"}),
            ("Stewart Butterfield", "Butterfield", {"person_type": "entrepreneur"}),
            ("Drew Houston", "Drew Houston", {"person_type": "entrepreneur"}),
            ("Tony Hsieh", "Tony Hsieh", {"person_type": "entrepreneur"}),
            ("Steve Jobs", "Steve Jobs", {"person_type": "entrepreneur"}),
        ],
        "journalist": [
            ("Anderson Cooper", "Anderson Cooper", {"person_type": "journalist", "role": "anchor", "org": "CNN"}),
            ("Christiane Amanpour", "Amanpour", {"person_type": "journalist", "role": "journalist", "org": "CNN"}),
            ("Bob Woodward", "Bob Woodward", {"person_type": "journalist", "role": "journalist", "org": "Washington Post"}),
            ("Kara Swisher", "Kara Swisher", {"person_type": "journalist", "role": "tech journalist"}),
            ("Tucker Carlson", "Tucker Carlson", {"person_type": "journalist", "role": "host", "org": "Fox News"}),
            ("Rachel Maddow", "Rachel Maddow", {"person_type": "journalist", "role": "host", "org": "MSNBC"}),
            ("Lester Holt", "Lester Holt", {"person_type": "journalist", "role": "anchor", "org": "NBC Nightly News"}),
            ("David Muir", "David Muir", {"person_type": "journalist", "role": "anchor", "org": "ABC World News Tonight"}),
            ("Norah O'Donnell", "Norah O'Donnell", {"person_type": "journalist", "role": "anchor", "org": "CBS Evening News"}),
            ("Wolf Blitzer", "Wolf Blitzer", {"person_type": "journalist", "role": "anchor", "org": "CNN"}),
            ("Fareed Zakaria", "Fareed Zakaria", {"person_type": "journalist", "role": "journalist", "org": "CNN"}),
            ("Maggie Haberman", "Haberman", {"person_type": "journalist", "role": "journalist", "org": "New York Times"}),
            ("Glenn Greenwald", "Glenn Greenwald", {"person_type": "journalist", "role": "journalist", "org": "The Intercept"}),
            ("Ronan Farrow", "Ronan Farrow", {"person_type": "journalist", "role": "journalist", "org": "The New Yorker"}),
            ("Savannah Guthrie", "Savannah Guthrie", {"person_type": "journalist", "role": "anchor", "org": "Today Show"}),
            ("Jake Tapper", "Jake Tapper", {"person_type": "journalist", "role": "anchor", "org": "CNN"}),
            ("Jorge Ramos", "Jorge Ramos", {"person_type": "journalist", "role": "anchor", "org": "Univision"}),
            ("Lesley Stahl", "Lesley Stahl", {"person_type": "journalist", "role": "correspondent", "org": "60 Minutes"}),
            ("Scott Pelley", "Scott Pelley", {"person_type": "journalist", "role": "correspondent", "org": "60 Minutes"}),
            ("Gayle King", "Gayle King", {"person_type": "journalist", "role": "anchor", "org": "CBS Mornings"}),
        ],
        "activist": [
            ("Greta Thunberg", "Greta Thunberg", {"person_type": "activist"}),
            ("Malala Yousafzai", "Malala", {"person_type": "activist"}),
            ("Naomi Klein", "Naomi Klein", {"person_type": "activist"}),
            ("Ai Weiwei", "Ai Weiwei", {"person_type": "activist"}),
            ("Desmond Tutu", "Desmond Tutu", {"person_type": "activist"}),
            ("Gloria Steinem", "Gloria Steinem", {"person_type": "activist"}),
            ("Angela Davis", "Angela Davis", {"person_type": "activist"}),
            ("Wangari Maathai", "Maathai", {"person_type": "activist"}),
            ("Vandana Shiva", "Vandana Shiva", {"person_type": "activist"}),
            ("Bryan Stevenson", "Bryan Stevenson", {"person_type": "activist"}),
            ("Tarana Burke", "Tarana Burke", {"person_type": "activist"}),
            ("Patrisse Cullors", "Cullors", {"person_type": "activist"}),
            ("Luisa Neubauer", "Neubauer", {"person_type": "activist"}),
            ("Joshua Wong", "Joshua Wong", {"person_type": "activist"}),
            ("Alexei Navalny", "Navalny", {"person_type": "activist"}),
            ("Aung San Suu Kyi", "Suu Kyi", {"person_type": "activist"}),
            ("Nelson Mandela", "Nelson Mandela", {"person_type": "activist"}),
            ("Martin Luther King Jr", "Martin Luther King", {"person_type": "activist"}),
            ("Rosa Parks", "Rosa Parks", {"person_type": "activist"}),
            ("Cesar Chavez", "Cesar Chavez", {"person_type": "activist"}),
        ],
        "scientist": [
            ("Albert Einstein", "Einstein", {"person_type": "scientist", "role": "physicist", "org": "Princeton"}),
            ("Stephen Hawking", "Hawking", {"person_type": "scientist", "role": "physicist", "org": "University of Cambridge"}),
            ("Marie Curie", "Marie Curie", {"person_type": "scientist", "role": "physicist"}),
            ("Jennifer Doudna", "Doudna", {"person_type": "scientist", "role": "biochemist", "org": "UC Berkeley"}),
            ("Emmanuelle Charpentier", "Charpentier", {"person_type": "scientist", "role": "microbiologist"}),
            ("Katalin Karikó", "Karik", {"person_type": "scientist", "role": "biochemist", "org": "BioNTech"}),
            ("Demis Hassabis", "Demis Hassabis", {"person_type": "scientist", "role": "AI researcher", "org": "DeepMind"}),
            ("Geoffrey Hinton", "Geoffrey Hinton", {"person_type": "scientist", "role": "computer scientist"}),
            ("Yann LeCun", "Yann Le", {"person_type": "scientist", "role": "AI researcher", "org": "Meta"}),
            ("Yoshua Bengio", "Yoshua Bengio", {"person_type": "scientist", "role": "computer scientist", "org": "Mila"}),
            ("Andrew Ng", "Andrew Ng", {"person_type": "scientist", "role": "computer scientist", "org": "Stanford"}),
            ("Fei-Fei Li", "Fei-Fei Li", {"person_type": "scientist", "role": "computer scientist", "org": "Stanford"}),
            ("Neil deGrasse Tyson", "Neil deGrasse Tyson", {"person_type": "scientist", "role": "astrophysicist", "org": "Hayden Planetarium"}),
            ("Jane Goodall", "Jane Goodall", {"person_type": "scientist", "role": "primatologist"}),
            ("Francis Collins", "Francis Collins", {"person_type": "scientist", "role": "geneticist", "org": "NIH"}),
            ("Kip Thorne", "Thorne", {"person_type": "scientist", "role": "physicist", "org": "Caltech"}),
            ("Roger Penrose", "Penrose", {"person_type": "scientist", "role": "mathematician", "org": "Oxford"}),
            ("Tu Youyou", "Tu Youyou", {"person_type": "scientist", "role": "pharmacologist"}),
            ("James Watson", "Watson", {"person_type": "scientist", "role": "molecular biologist"}),
            ("Tim Berners-Lee", "Tim Berners-Lee", {"person_type": "scientist", "role": "computer scientist"}),
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

    mode = "hybrid" if hybrid else "embeddings-only"
    click.echo(
        f"Person search perf+accuracy test [{mode}] — "
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
    query_idx = 0
    # For --for-llm: collect every non-top1 result for review
    llm_issues: list[dict[str, Any]] = []

    for ptype, queries in test_queries_by_type.items():
        click.echo(f"\n{'=' * 80}", err=True)
        click.echo(f"  {ptype.upper()} ({len(queries)} queries)", err=True)
        click.echo(f"{'=' * 80}", err=True)

        hits_at_1 = 0
        hits_in_topk = 0
        type_timings: list[float] = []

        for i, (name, expected, query_kwargs) in enumerate(queries, 1):
            query_idx += 1
            expected_lower = expected.lower()

            query = format_person_query(name, **query_kwargs)
            t0 = _time.perf_counter()
            query_embedding = embedder.embed(query)
            embed_elapsed = _time.perf_counter() - t0

            t1 = _time.perf_counter()
            query_text = name if hybrid else None
            results = database.search(query_embedding, top_k=top_k, query_text=query_text)
            search_elapsed = _time.perf_counter() - t1

            total_elapsed = _time.perf_counter() - t0
            type_timings.append(total_elapsed)
            all_timings.append(total_elapsed)

            # Accuracy: check if expected name appears in results
            top1_match = False
            topk_match = False
            topk_rank = -1
            if results:
                if expected_lower in results[0][0].name.lower():
                    top1_match = True
                    topk_match = True
                    topk_rank = 1
                else:
                    for rank, (rec, _score) in enumerate(results, 1):
                        if expected_lower in rec.name.lower():
                            topk_match = True
                            topk_rank = rank
                            break

            if top1_match:
                hits_at_1 += 1
            if topk_match:
                hits_in_topk += 1

            # Collect for --for-llm output (all non-top1 hits are worth reviewing)
            if for_llm and not top1_match:
                top_results = [
                    {"rank": r, "name": rec.name, "score": round(sc, 4),
                     "person_type": rec.person_type.value, "role": rec.known_for_role, "org": rec.known_for_org_name}
                    for r, (rec, sc) in enumerate(results, 1)
                ]
                llm_issues.append({
                    "type": ptype,
                    "query": query,
                    "expected": expected,
                    "status": "wrong_rank" if topk_match else "missing",
                    "found_at_rank": topk_rank if topk_match else None,
                    "top_results": top_results,
                })

            # Display
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

        click.echo(
            f"  → {ptype}: acc@1={acc1:.0f}%  acc@{top_k}={acck:.0f}%  "
            f"mean={mean_ms:.1f}ms",
            err=True,
        )

    # Summary
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
    global_acc1 = global_hits_at_1 / global_total * 100 if global_total else 0
    global_acck = global_hits_in_topk / global_total * 100 if global_total else 0
    global_mean = sum(all_timings) / len(all_timings) * 1000 if all_timings else 0
    click.echo(
        f"  {'TOTAL':<16s} {global_total:4d} "
        f"{global_acc1:6.1f}% {global_acck:6.1f}% "
        f"{global_mean:7.1f}ms",
        err=True,
    )
    click.echo(f"\n  Total time: {sum(all_timings):.2f}s  |  "
               f"Min: {min(all_timings) * 1000:.1f}ms  |  "
               f"Max: {max(all_timings) * 1000:.1f}ms", err=True)

    # --for-llm: structured output to stdout for LLM review
    if for_llm:
        import json as _json
        n_missing = sum(1 for i in llm_issues if i["status"] == "missing")
        n_wrong_rank = sum(1 for i in llm_issues if i["status"] == "wrong_rank")
        llm_output = {
            "summary": {
                "total_queries": global_total,
                "acc_at_1": round(global_acc1, 1),
                "acc_at_k": round(global_acck, 1),
                "top_k": top_k,
                "mode": mode,
                "failures": n_missing,
                "wrong_rank": n_wrong_rank,
            },
            "instructions": (
                "Review each issue below. For 'missing' items the expected person was not found "
                "in the top-k results at all — check if the expected name substring is wrong (typo, "
                "accent, alternate spelling) or if the person is genuinely not in the database. "
                "For 'wrong_rank' items the person was found but not at rank 1 — check if the "
                "query could be improved or if the expected value is too ambiguous. "
                "Propose fixes to the test_queries_by_type dict in cli.py if the expected value "
                "is incorrect. Do NOT change the test if the search result is genuinely wrong."
            ),
            "issues": llm_issues,
        }
        click.echo(_json.dumps(llm_output, indent=2, ensure_ascii=False))

    database.close()


@click.command("search")
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(), help="Database path")
@click.option("--top-k", type=int, default=10, help="Number of results")
@click.option("--source", type=click.Choice(["gleif", "sec_edgar", "companies_house", "wikipedia"]), help="Filter by source")
@click.option("--hybrid", is_flag=True, help="Use hybrid text + embeddings search (default is embeddings-only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def db_search(query: str, db_path: Optional[str], top_k: int, source: Optional[str], hybrid: bool, verbose: bool):
    """
    Search for an organization in the database.

    \b
    Examples:
        corp-entity-db search "Apple Inc"
        corp-entity-db search "Microsoft" --source sec_edgar
        corp-entity-db search "Apple" --hybrid
    """
    _configure_logging(verbose)

    from corp_entity_db import OrganizationDatabase, CompanyEmbedder

    db_path_obj = _resolve_db_path(db_path)
    embedder = CompanyEmbedder()
    database = OrganizationDatabase(db_path=db_path_obj)

    mode = "hybrid (text + embeddings)" if hybrid else "embeddings-only"
    click.echo(f"Searching for '{query}' [{mode}]...", err=True)

    # Embed query
    query_embedding = embedder.embed(query)

    # Search
    query_text = query if hybrid else None
    results = database.search(query_embedding, top_k=top_k, source_filter=source, query_text=query_text)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop {len(results)} matches:")
    click.echo("-" * 60)

    for i, (record, similarity) in enumerate(results, 1):
        click.echo(f"{i}. {record.legal_name}")
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
