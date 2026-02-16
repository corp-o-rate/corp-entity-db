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
        org_str = f" at {record.known_for_org}" if record.known_for_org else ""
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

    from corp_entity_db.store import get_person_database
    from corp_entity_db.embeddings import CompanyEmbedder

    # Each entry: (query_string, expected_name_substring)
    # The expected_name_substring is matched case-insensitively against result names.
    test_queries_by_type: dict[str, list[tuple[str, str]]] = {
        "executive": [
            ("Tim Cook, CEO, Apple", "Tim Cook"),
            ("Satya Nadella, CEO, Microsoft", "Satya Nadella"),
            ("Andy Jassy, CEO, Amazon", "Andy Jassy"),
            ("Sundar Pichai, CEO, Google", "Sundar Pichai"),
            ("Jensen Huang, CEO, NVIDIA", "Jensen Huang"),
            ("Mark Zuckerberg, CEO, Meta", "Mark Zuckerberg"),
            ("Jamie Dimon, CEO, JPMorgan Chase", "Jamie Dimon"),
            ("Warren Buffett, CEO, Berkshire Hathaway", "Warren Buffett"),
            ("Elon Musk, CEO, Tesla", "Elon Musk"),
            ("Lisa Su, CEO, AMD", "Lisa Su"),
            ("Mary Barra, CEO, General Motors", "Mary Barra"),
            ("David Solomon, CEO, Goldman Sachs", "Solomon"),
            ("Brian Moynihan, CEO, Bank of America", "Brian Moynihan"),
            ("Arvind Krishna, CEO, IBM", "Arvind Krishna"),
            ("Pat Gelsinger, CEO, Intel", "Pat Gelsinger"),
            ("Chuck Robbins, CEO, Cisco", "Chuck Robbins"),
            ("Safra Catz, CEO, Oracle", "Safra Catz"),
            ("Shantanu Narayen, CEO, Adobe", "Shantanu Narayen"),
            ("Marc Benioff, CEO, Salesforce", "Marc Benioff"),
            ("Darius Adamczyk, CEO, Honeywell", "Darius Adamczyk"),
        ],
        "politician": [
            ("Joe Biden, President, United States", "Joe Biden"),
            ("Donald Trump, President, United States", "Donald Trump"),
            ("Emmanuel Macron, President, France", "Emmanuel Macron"),
            ("Rishi Sunak, Prime Minister, United Kingdom", "Rishi Sunak"),
            ("Olaf Scholz, Chancellor, Germany", "Olaf Scholz"),
            ("Justin Trudeau, Prime Minister, Canada", "Justin Trudeau"),
            ("Narendra Modi, Prime Minister, India", "Narendra Modi"),
            ("Fumio Kishida, Prime Minister, Japan", "Fumio Kishida"),
            ("Volodymyr Zelenskyy, President, Ukraine", "Zelenskyy"),
            ("Pedro Sánchez, Prime Minister, Spain", "Pedro S"),
            ("Anthony Albanese, Prime Minister, Australia", "Albanese"),
            ("Giorgia Meloni, Prime Minister, Italy", "Meloni"),
            ("Lula da Silva, President, Brazil", "Lula"),
            ("Jacinda Ardern, Prime Minister, New Zealand", "Ardern"),
            ("Xi Jinping, President, China", "Xi Jinping"),
            ("Recep Tayyip Erdogan, President, Turkey", "Erdo"),
            ("Benjamin Netanyahu, Prime Minister, Israel", "Netanyahu"),
            ("Yoon Suk-yeol, President, South Korea", "Yoon"),
            ("Mark Rutte, Prime Minister, Netherlands", "Rutte"),
            ("Keir Starmer, Prime Minister, United Kingdom", "Starmer"),
        ],
        "government": [
            ("Janet Yellen, Secretary of the Treasury, United States", "Yellen"),
            ("Antony Blinken, Secretary of State, United States", "Blinken"),
            ("Jerome Powell, Chair, Federal Reserve", "Powell"),
            ("Christine Lagarde, President, European Central Bank", "Lagarde"),
            ("Ursula von der Leyen, President, European Commission", "von der Leyen"),
            ("António Guterres, Secretary-General, United Nations", "Guterres"),
            ("Tedros Adhanom, Director-General, WHO", "Tedros"),
            ("Kristalina Georgieva, Managing Director, IMF", "Georgieva"),
            ("Gary Gensler, Chair, SEC", "Gensler"),
            ("Merrick Garland, Attorney General, United States", "Garland"),
            ("Andrew Bailey, Governor, Bank of England", "Bailey"),
            ("Haruhiko Kuroda, Governor, Bank of Japan", "Kuroda"),
            ("Lloyd Austin, Secretary of Defense, United States", "Austin"),
            ("Avril Haines, Director of National Intelligence, United States", "Haines"),
            ("Alejandro Mayorkas, Secretary, DHS", "Mayorkas"),
            ("Gina Raimondo, Secretary of Commerce, United States", "Raimondo"),
            ("Janet Woodcock, Commissioner, FDA", "Woodcock"),
            ("Ajay Banga, President, World Bank", "Banga"),
            ("Ngozi Okonjo-Iweala, Director-General, WTO", "Okonjo"),
            ("Charles Michel, President, European Council", "Michel"),
        ],
        "military": [
            ("Mark Milley, Chairman Joint Chiefs of Staff, United States", "Milley"),
            ("Valerii Zaluzhnyi, Commander-in-Chief, Ukraine Armed Forces", "Zaluzhnyi"),
            ("Tony Radakin, Chief of Defence Staff, United Kingdom", "Radakin"),
            ("Thierry Burkhard, Chief of Defence Staff, France", "Burkhard"),
            ("Rob Bauer, Chair, NATO Military Committee", "Bauer"),
            ("Christopher Cavoli, SACEUR, NATO", "Cavoli"),
            ("Michael Kurilla, Commander, CENTCOM", "Kurilla"),
            ("Charles Brown, Chairman Joint Chiefs of Staff, United States", "Charles E. Brown"),
            ("Eberhard Zorn, Inspector General, Germany Bundeswehr", "Zorn"),
            ("Koji Yamazaki, Chief of Staff, Japan Self-Defense Forces", "Yamazaki"),
            ("Angus Campbell, Chief of Defence Force, Australia", "Campbell"),
            ("Bipin Rawat, Chief of Defence Staff, India", "Rawat"),
            ("Wayne Eyre, Chief of Defence Staff, Canada", "Eyre"),
            ("Sergei Shoigu, Minister of Defence, Russia", "Shoigu"),
            ("James Hecker, Commander, US Air Forces in Europe", "Hecker"),
            ("Samuel Paparo, Commander, US Indo-Pacific Command", "Paparo"),
            ("Laura Richardson, Commander, US Southern Command", "Richardson"),
            ("Mauro Del Vecchio, Commander, NATO Joint Force Command", "Del Vecchio"),
            ("Stuart Peach, Chair, NATO Military Committee", "Peach"),
            ("Eirik Kristoffersen, Chief of Defence, Norway", "Kristoffersen"),
        ],
        "legal": [
            ("John Roberts, Chief Justice, Supreme Court of the United States", "Roberts"),
            ("Sonia Sotomayor, Justice, Supreme Court of the United States", "Sotomayor"),
            ("Ketanji Brown Jackson, Justice, Supreme Court of the United States", "Ketanji"),
            ("Clarence Thomas, Justice, Supreme Court of the United States", "Clarence Thomas"),
            ("Elena Kagan, Justice, Supreme Court of the United States", "Kagan"),
            ("Neil Gorsuch, Justice, Supreme Court of the United States", "Gorsuch"),
            ("Brett Kavanaugh, Justice, Supreme Court of the United States", "Kavanaugh"),
            ("Amy Coney Barrett, Justice, Supreme Court of the United States", "Barrett"),
            ("Samuel Alito, Justice, Supreme Court of the United States", "Alito"),
            ("Ruth Bader Ginsburg, Justice, Supreme Court of the United States", "Ginsburg"),
            ("Brenda Hale, President, Supreme Court, United Kingdom", "Hale"),
            ("Robert Reed, President, UK Supreme Court", "Reed"),
            ("Karim Khan, Prosecutor, International Criminal Court", "Khan"),
            ("Joan Donoghue, President, International Court of Justice", "Donoghue"),
            ("Didier Reynders, Commissioner for Justice, European Commission", "Reynders"),
            ("Fatou Bensouda, Prosecutor, International Criminal Court", "Bensouda"),
            ("Loretta Lynch, Attorney General, United States", "Lynch"),
            ("Eric Holder, Attorney General, United States", "Holder"),
            ("Robert Mueller, Special Counsel, Department of Justice", "Mueller"),
            ("Jack Smith, Special Counsel, Department of Justice", "Jack Smith"),
        ],
        "professional": [
            ("Atul Gawande, surgeon, Brigham and Women's Hospital", "Gawande"),
            ("Sanjay Gupta, neurosurgeon, Emory University Hospital", "Sanjay Gupta"),
            ("Anthony Fauci, immunologist, NIAID", "Fauci"),
            ("Devi Shetty, cardiac surgeon, Narayana Health", "Shetty"),
            ("Norman Foster, architect, Foster + Partners", "Norman Foster"),
            ("Bjarke Ingels, architect, BIG", "Bjarke Ingels"),
            ("Zaha Hadid, architect, Zaha Hadid Architects", "Zaha Hadid"),
            ("Renzo Piano, architect, Renzo Piano Building Workshop", "Renzo Piano"),
            ("Frank Gehry, architect, Gehry Partners", "Gehry"),
            ("Tadao Ando, architect, Tadao Ando Architect", "Tadao Ando"),
            ("I. M. Pei, architect, Pei Cobb Freed", "Pei"),
            ("Santiago Calatrava, architect, Calatrava", "Calatrava"),
            ("Rem Koolhaas, architect, OMA", "Koolhaas"),
            ("Thomas Heatherwick, designer, Heatherwick Studio", "Heatherwick"),
            ("Jony Ive, designer, Apple", "Ive"),
            ("Dieter Rams, designer, Braun", "Dieter Rams"),
            ("Philippe Starck, designer", "Starck"),
            ("Toyo Ito, architect", "Toyo Ito"),
            ("Daniel Libeskind, architect", "Libeskind"),
            ("Peter Zumthor, architect", "Zumthor"),
        ],
        "academic": [
            ("Noam Chomsky, professor, MIT", "Chomsky"),
            ("Steven Pinker, professor, Harvard", "Pinker"),
            ("Paul Krugman, professor, Princeton", "Krugman"),
            ("Joseph Stiglitz, professor, Columbia", "Stiglitz"),
            ("Thomas Piketty, professor, Paris School of Economics", "Piketty"),
            ("Yuval Noah Harari, professor, Hebrew University", "Harari"),
            ("Niall Ferguson, professor, Stanford", "Ferguson"),
            ("Lawrence Lessig, professor, Harvard Law School", "Lessig"),
            ("Cornel West, professor, Union Theological Seminary", "Cornel West"),
            ("Nassim Nicholas Taleb, professor, NYU", "Taleb"),
            ("Jordan Peterson, professor, University of Toronto", "Peterson"),
            ("Richard Dawkins, professor, Oxford", "Dawkins"),
            ("Amy Cuddy, professor, Harvard Business School", "Cuddy"),
            ("Brené Brown, professor, University of Houston", "Brown"),
            ("Henry Kissinger, professor, Georgetown", "Kissinger"),
            ("Daron Acemoglu, professor, MIT", "Acemo"),
            ("Tyler Cowen, professor, George Mason University", "Cowen"),
            ("Esther Duflo, professor, MIT", "Duflo"),
            ("Abhijit Banerjee, professor, MIT", "Banerjee"),
            ("Jeffrey Sachs, professor, Columbia", "Sachs"),
        ],
        "artist": [
            ("Taylor Swift, singer, musician", "Taylor Swift"),
            ("Beyoncé, singer, musician", "Beyonc"),
            ("Ed Sheeran, singer, musician", "Ed Sheeran"),
            ("Adele, singer, musician", "Adele"),
            ("Drake, rapper, musician", "Drake"),
            ("Tom Hanks, actor, Hollywood", "Tom Hanks"),
            ("Meryl Streep, actress, Hollywood", "Meryl Streep"),
            ("Leonardo DiCaprio, actor, Hollywood", "Leonardo DiCaprio"),
            ("Cate Blanchett, actress", "Cate Blanchett"),
            ("Denzel Washington, actor, Hollywood", "Denzel Washington"),
            ("Christopher Nolan, director, filmmaker", "Christopher Nolan"),
            ("Martin Scorsese, director, filmmaker", "Martin Scorsese"),
            ("Steven Spielberg, director, filmmaker", "Spielberg"),
            ("Banksy, street artist", "Banksy"),
            ("Ai Weiwei, artist, sculptor", "Ai Weiwei"),
            ("Damien Hirst, artist", "Damien Hirst"),
            ("J.K. Rowling, author, Harry Potter", "Rowling"),
            ("Stephen King, author, writer", "Stephen King"),
            ("Haruki Murakami, author, writer", "Murakami"),
            ("Bob Dylan, musician, singer-songwriter", "Bob Dylan"),
        ],
        "media": [
            ("PewDiePie, YouTuber", "PewDiePie"),
            ("MrBeast, YouTuber", "MrBeast"),
            ("Joe Rogan, podcaster, The Joe Rogan Experience", "Joe Rogan"),
            ("Kim Kardashian, social media personality", "Kim Kardashian"),
            ("Kylie Jenner, social media personality", "Kylie Jenner"),
            ("Logan Paul, YouTuber, social media personality", "Logan Paul"),
            ("Markiplier, YouTuber, gaming creator", "Markiplier"),
            ("Hasan Piker, Twitch streamer", "Hasan"),
            ("Marques Brownlee, YouTuber, tech reviewer", "Marques Brownlee"),
            ("Emma Chamberlain, YouTuber, social media personality", "Emma Chamberlain"),
            ("Casey Neistat, YouTuber, filmmaker", "Casey Neistat"),
            ("Lilly Singh, YouTuber, television host", "Lilly Singh"),
            ("David Dobrik, YouTuber, social media personality", "David Dobrik"),
            ("Charli D'Amelio, TikTok personality", "Charli"),
            ("Addison Rae, TikTok personality", "Addison Rae"),
            ("Ninja, Twitch streamer, gamer", "Ninja"),
            ("Pokimane, Twitch streamer", "Pokimane"),
            ("Linus Sebastian, YouTuber, Linus Tech Tips", "Linus"),
            ("Philip DeFranco, YouTuber, news commentator", "Philip DeFranco"),
            ("Rhett McLaughlin, YouTuber, Good Mythical Morning", "Rhett"),
        ],
        "athlete": [
            ("LeBron James, basketball player, Los Angeles Lakers", "LeBron James"),
            ("Lionel Messi, football player, Inter Miami", "Lionel Messi"),
            ("Cristiano Ronaldo, football player, Al Nassr", "Cristiano Ronaldo"),
            ("Serena Williams, tennis player", "Serena Williams"),
            ("Roger Federer, tennis player", "Roger Federer"),
            ("Novak Djokovic, tennis player", "Novak Djokovic"),
            ("Usain Bolt, sprinter, Jamaica", "Usain Bolt"),
            ("Michael Phelps, swimmer, United States", "Michael Phelps"),
            ("Simone Biles, gymnast, United States", "Simone Biles"),
            ("Lewis Hamilton, Formula 1 driver, Mercedes", "Lewis Hamilton"),
            ("Max Verstappen, Formula 1 driver, Red Bull", "Max Verstappen"),
            ("Tom Brady, football player, NFL", "Tom Brady"),
            ("Patrick Mahomes, football player, Kansas City Chiefs", "Mahomes"),
            ("Kylian Mbappé, football player, Real Madrid", "Mbapp"),
            ("Erling Haaland, football player, Manchester City", "Haaland"),
            ("Stephen Curry, basketball player, Golden State Warriors", "Stephen Curry"),
            ("Naomi Osaka, tennis player", "Naomi Osaka"),
            ("Katie Ledecky, swimmer, United States", "Ledecky"),
            ("Eliud Kipchoge, marathon runner, Kenya", "Kipchoge"),
            ("Neymar, football player, Brazil", "Neymar"),
        ],
        "entrepreneur": [
            ("Jeff Bezos, founder, Amazon", "Jeff Bezos"),
            ("Bill Gates, founder, Microsoft", "Bill Gates"),
            ("Larry Page, co-founder, Google", "Larry Page"),
            ("Sergey Brin, co-founder, Google", "Sergey Brin"),
            ("Jack Dorsey, founder, Twitter", "Jack Dorsey"),
            ("Reid Hoffman, co-founder, LinkedIn", "Reid Hoffman"),
            ("Peter Thiel, co-founder, PayPal", "Peter Thiel"),
            ("Travis Kalanick, founder, Uber", "Travis Kalanick"),
            ("Brian Chesky, co-founder, Airbnb", "Brian Chesky"),
            ("Jack Ma, founder, Alibaba", "Jack Ma"),
            ("Richard Branson, founder, Virgin Group", "Richard Branson"),
            ("Sam Altman, CEO, OpenAI", "Sam Altman"),
            ("Evan Spiegel, co-founder, Snap", "Evan Spiegel"),
            ("Daniel Ek, co-founder, Spotify", "Daniel Ek"),
            ("Patrick Collison, co-founder, Stripe", "Patrick Collison"),
            ("Whitney Wolfe Herd, founder, Bumble", "Whitney Wolfe"),
            ("Stewart Butterfield, co-founder, Slack", "Butterfield"),
            ("Drew Houston, co-founder, Dropbox", "Drew Houston"),
            ("Tony Hsieh, CEO, Zappos", "Tony Hsieh"),
            ("Steve Jobs, founder, Apple", "Steve Jobs"),
        ],
        "journalist": [
            ("Anderson Cooper, anchor, CNN", "Anderson Cooper"),
            ("Christiane Amanpour, journalist, CNN", "Amanpour"),
            ("Bob Woodward, journalist, Washington Post", "Bob Woodward"),
            ("Kara Swisher, tech journalist, podcaster", "Kara Swisher"),
            ("Tucker Carlson, host, Fox News", "Tucker Carlson"),
            ("Rachel Maddow, host, MSNBC", "Rachel Maddow"),
            ("Lester Holt, anchor, NBC Nightly News", "Lester Holt"),
            ("David Muir, anchor, ABC World News Tonight", "David Muir"),
            ("Norah O'Donnell, anchor, CBS Evening News", "Norah O'Donnell"),
            ("Wolf Blitzer, anchor, CNN", "Wolf Blitzer"),
            ("Fareed Zakaria, journalist, CNN", "Fareed Zakaria"),
            ("Maggie Haberman, journalist, New York Times", "Haberman"),
            ("Glenn Greenwald, journalist, The Intercept", "Glenn Greenwald"),
            ("Ronan Farrow, journalist, The New Yorker", "Ronan Farrow"),
            ("Savannah Guthrie, anchor, Today Show", "Savannah Guthrie"),
            ("Jake Tapper, anchor, CNN", "Jake Tapper"),
            ("Jorge Ramos, anchor, Univision", "Jorge Ramos"),
            ("Lesley Stahl, correspondent, 60 Minutes", "Lesley Stahl"),
            ("Scott Pelley, correspondent, 60 Minutes", "Scott Pelley"),
            ("Gayle King, anchor, CBS Mornings", "Gayle King"),
        ],
        "activist": [
            ("Greta Thunberg, climate activist", "Greta Thunberg"),
            ("Malala Yousafzai, education activist", "Malala"),
            ("Naomi Klein, author, activist", "Naomi Klein"),
            ("Ai Weiwei, artist, human rights activist", "Ai Weiwei"),
            ("Desmond Tutu, anti-apartheid activist, South Africa", "Desmond Tutu"),
            ("Gloria Steinem, feminist activist", "Gloria Steinem"),
            ("Angela Davis, civil rights activist", "Angela Davis"),
            ("Wangari Maathai, environmental activist, Kenya", "Maathai"),
            ("Vandana Shiva, environmental activist, India", "Vandana Shiva"),
            ("Bryan Stevenson, civil rights lawyer, Equal Justice Initiative", "Bryan Stevenson"),
            ("Tarana Burke, founder, MeToo movement", "Tarana Burke"),
            ("Patrisse Cullors, co-founder, Black Lives Matter", "Cullors"),
            ("Luisa Neubauer, climate activist, Fridays for Future", "Neubauer"),
            ("Joshua Wong, pro-democracy activist, Hong Kong", "Joshua Wong"),
            ("Alexei Navalny, opposition leader, Russia", "Navalny"),
            ("Aung San Suu Kyi, pro-democracy leader, Myanmar", "Suu Kyi"),
            ("Nelson Mandela, anti-apartheid leader, South Africa", "Nelson Mandela"),
            ("Martin Luther King Jr, civil rights leader, United States", "Martin Luther King"),
            ("Rosa Parks, civil rights activist, United States", "Rosa Parks"),
            ("Cesar Chavez, labor rights activist, United States", "Cesar Chavez"),
        ],
        "scientist": [
            ("Albert Einstein, physicist, Princeton", "Einstein"),
            ("Stephen Hawking, physicist, University of Cambridge", "Hawking"),
            ("Marie Curie, physicist, chemist", "Marie Curie"),
            ("Jennifer Doudna, biochemist, CRISPR, UC Berkeley", "Doudna"),
            ("Emmanuelle Charpentier, microbiologist, CRISPR", "Charpentier"),
            ("Katalin Karikó, biochemist, mRNA, BioNTech", "Karik"),
            ("Demis Hassabis, AI researcher, DeepMind", "Demis Hassabis"),
            ("Geoffrey Hinton, computer scientist, AI pioneer", "Geoffrey Hinton"),
            ("Yann LeCun, AI researcher, Meta", "Yann Le"),
            ("Yoshua Bengio, computer scientist, Mila", "Yoshua Bengio"),
            ("Andrew Ng, computer scientist, Stanford", "Andrew Ng"),
            ("Fei-Fei Li, computer scientist, Stanford", "Fei-Fei Li"),
            ("Neil deGrasse Tyson, astrophysicist, Hayden Planetarium", "Neil deGrasse Tyson"),
            ("Jane Goodall, primatologist, Gombe", "Jane Goodall"),
            ("Francis Collins, geneticist, NIH", "Francis Collins"),
            ("Kip Thorne, physicist, Caltech", "Thorne"),
            ("Roger Penrose, mathematician, Oxford", "Penrose"),
            ("Tu Youyou, pharmacologist, Nobel Prize", "Tu Youyou"),
            ("James Watson, molecular biologist, DNA", "Watson"),
            ("Tim Berners-Lee, computer scientist, inventor of the Web", "Tim Berners-Lee"),
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

        for i, (query, expected) in enumerate(queries, 1):
            query_idx += 1
            expected_lower = expected.lower()

            t0 = _time.perf_counter()
            query_embedding = embedder.embed(query)
            embed_elapsed = _time.perf_counter() - t0

            t1 = _time.perf_counter()
            query_text = query if hybrid else None
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
                     "person_type": rec.person_type.value, "role": rec.known_for_role, "org": rec.known_for_org}
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
