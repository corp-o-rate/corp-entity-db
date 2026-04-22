"""Download DB + USearch indexes at Docker build time.

Configures logging so shard/manifest downloads are visible in build output,
and prints a final summary of everything fetched into the HuggingFace snapshot dir.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from corp_entity_db.hub import download_database

db = download_database()
snap = db.parent
print(f"Database: {db}")
print(f"Snapshot dir: {snap}")
print("Files downloaded:")
total = 0
for f in sorted(snap.iterdir()):
    size = f.resolve().stat().st_size
    total += size
    print(f"  {f.name}  {size / 1024 ** 2:,.0f} MB")
print(f"Total: {total / 1024 ** 3:,.1f} GB")
