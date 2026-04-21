"""External alias importers for organizations.

Imports alias records from:
- wiki-entity-similarity (Wikipedia anchor text → org mapping)
- ParaNames (multilingual entity names from Wikidata)
"""

import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

from corp_names import normalize_company

from ..seed_data import SOURCE_NAME_TO_ID

logger = logging.getLogger(__name__)

# Dataset URLs
WIKI_ANCHOR_URL = (
    "https://huggingface.co/datasets/Exr0n/wiki-entity-similarity/resolve/"
    "refs%2Fconvert%2Fparquet/2018thresh5corpus/train/0000.parquet"
)
PARANAMES_URL = (
    "https://github.com/bltlab/paranames/releases/download/"
    "v2024.05.07.0/paranames.tsv.gz"
)


def download_wiki_anchor(output_dir: Path | None = None, force: bool = False) -> Path:
    """Download the wiki-entity-similarity Parquet file from HuggingFace."""
    if output_dir is None:
        output_dir = Path.home() / ".cache" / "corp-extractor" / "aliases"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "wiki_entity_similarity.parquet"
    if output_path.exists() and not force:
        logger.info(f"Using cached wiki-anchor data: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    logger.info(f"Downloading wiki-entity-similarity (~85 MB)...")
    urllib.request.urlretrieve(WIKI_ANCHOR_URL, output_path)
    logger.info(f"Downloaded to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def download_paranames(output_dir: Path | None = None, force: bool = False) -> Path:
    """Download and decompress the ParaNames TSV from GitHub releases."""
    if output_dir is None:
        output_dir = Path.home() / ".cache" / "corp-extractor" / "aliases"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "paranames.tsv"
    if output_path.exists() and not force:
        logger.info(f"Using cached ParaNames data: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    gz_path = output_dir / "paranames.tsv.gz"
    logger.info(f"Downloading ParaNames (~1 GB compressed)...")
    urllib.request.urlretrieve(PARANAMES_URL, gz_path)
    logger.info(f"Decompressing {gz_path}...")

    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()
    logger.info(f"Decompressed to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def import_wiki_entity_similarity(conn, file_path: str | Path) -> int:
    """
    Import Wikipedia anchor text aliases from Exr0n/wiki-entity-similarity dataset.

    Matches article titles to existing orgs by name_normalized and creates alias
    records with alias_source_id=8 (wiki_anchor).

    Args:
        conn: SQLite connection (writable)
        file_path: Path to the dataset file (Parquet or CSV)

    Returns:
        Number of alias records inserted
    """
    import pandas as pd

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.info(f"Loading wiki-entity-similarity from {file_path}")

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    logger.info(f"Loaded {len(df):,} rows")

    # Build name_normalized → org lookup from existing primary records
    cursor = conn.execute("""
        SELECT id, name_normalized, source_id, source_identifier, qid,
               region_id, entity_type_id, from_date, to_date, canon_id
        FROM organizations
        WHERE alias_source_id IS NULL
    """)
    name_to_orgs: dict[str, list[dict]] = {}
    for row in cursor:
        key = row[1]  # name_normalized
        name_to_orgs.setdefault(key, []).append({
            "id": row[0], "source_id": row[2], "source_identifier": row[3],
            "qid": row[4], "region_id": row[5], "entity_type_id": row[6],
            "from_date": row[7], "to_date": row[8], "canon_id": row[9],
        })

    wiki_anchor_id = SOURCE_NAME_TO_ID["wiki_anchor"]  # 8
    inserted = 0

    # Expected columns: 'anchor_text' and 'title' (or similar)
    anchor_col = "anchor_text" if "anchor_text" in df.columns else df.columns[0]
    title_col = "title" if "title" in df.columns else df.columns[1]

    for _, row in df.iterrows():
        anchor = str(row[anchor_col]).strip()
        title = str(row[title_col]).strip()
        if not anchor or not title:
            continue

        # Normalize title to find matching org
        title_norm = normalize_company(title).normalized
        orgs = name_to_orgs.get(title_norm)
        if not orgs:
            continue

        anchor_norm = normalize_company(anchor).normalized
        if anchor_norm == title_norm:
            continue  # Skip if alias normalizes same as the org

        # Create alias for first matching org
        org = orgs[0]
        canon_id = org["canon_id"] or org["id"]
        try:
            conn.execute("""
                INSERT OR IGNORE INTO organizations
                (name, name_normalized, source_id, source_identifier, qid,
                 region_id, entity_type_id, from_date, to_date, record,
                 canon_id, canon_size, alias_source_id, alias_source_identifier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, 1, ?, ?)
            """, (
                anchor, anchor_norm,
                org["source_id"], org["source_identifier"], org["qid"],
                org["region_id"], org["entity_type_id"],
                org["from_date"], org["to_date"],
                canon_id, wiki_anchor_id, title,
            ))
            inserted += 1
        except Exception as e:
            logger.debug(f"Skipped wiki anchor '{anchor}' for '{title}': {e}")

        if inserted % 10_000 == 0 and inserted > 0:
            conn.commit()
            logger.info(f"Inserted {inserted:,} wiki anchor aliases...")

    conn.commit()
    logger.info(f"Wiki anchor import complete: {inserted:,} aliases inserted")
    return inserted


def import_paranames(conn, file_path: str | Path) -> int:
    """
    Import ParaNames multilingual entity aliases from bltlab/ParaNames dataset.

    Filters for type=ORG entries, matches by QID to existing orgs, and creates
    alias records with alias_source_id=9 (paranames).

    Args:
        conn: SQLite connection (writable)
        file_path: Path to the ParaNames TSV file

    Returns:
        Number of alias records inserted
    """
    import csv

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ParaNames file not found: {file_path}")

    logger.info(f"Loading ParaNames from {file_path}")

    # Build QID → org lookup from existing primary records
    cursor = conn.execute("""
        SELECT id, qid, name_normalized, source_id, source_identifier,
               region_id, entity_type_id, from_date, to_date, canon_id
        FROM organizations
        WHERE alias_source_id IS NULL AND qid IS NOT NULL
    """)
    qid_to_org: dict[int, dict] = {}
    for row in cursor:
        qid_to_org[row[1]] = {
            "id": row[0], "name_normalized": row[2],
            "source_id": row[3], "source_identifier": row[4],
            "region_id": row[5], "entity_type_id": row[6],
            "from_date": row[7], "to_date": row[8], "canon_id": row[9],
        }

    paranames_id = SOURCE_NAME_TO_ID["paranames"]  # 9
    inserted = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_type = row.get("type", "")
            if entity_type != "ORG":
                continue

            qid_str = row.get("wikidata_id", "")
            if not qid_str.startswith("Q"):
                continue
            qid_int = int(qid_str[1:])

            org = qid_to_org.get(qid_int)
            if not org:
                continue

            alias = row.get("name", "").strip()
            if not alias:
                continue

            lang = row.get("language", "")
            if lang != "en":
                continue  # Only import English aliases

            alias_norm = normalize_company(alias).normalized
            if alias_norm == org["name_normalized"]:
                continue

            canon_id = org["canon_id"] or org["id"]
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO organizations
                    (name, name_normalized, source_id, source_identifier, qid,
                     region_id, entity_type_id, from_date, to_date, record,
                     canon_id, canon_size, alias_source_id, alias_source_identifier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, 1, ?, ?)
                """, (
                    alias, alias_norm,
                    org["source_id"], org["source_identifier"], qid_int,
                    org["region_id"], org["entity_type_id"],
                    org["from_date"], org["to_date"],
                    canon_id, paranames_id, qid_str,
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Skipped paraname '{alias}' for {qid_str}: {e}")

            if inserted % 10_000 == 0 and inserted > 0:
                conn.commit()
                logger.info(f"Inserted {inserted:,} ParaNames aliases...")

    conn.commit()
    logger.info(f"ParaNames import complete: {inserted:,} aliases inserted")
    return inserted
