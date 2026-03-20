#!/usr/bin/env python3
"""Simple and direct title cleaning."""

import sqlite3
import re
from tqdm import tqdm

def clean_title(title):
    if not title:
        return ""
    t = title
    # Remove brackets
    t = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', t)
    # Remove | and after
    t = t.split('|')[0]
    # Remove "with" and after
    t = re.sub(r'\s+with\s+.*', '', t, flags=re.IGNORECASE)
    # Handle hyphens
    if t.count(' - ') >= 2:
        parts = t.split(' - ')
        t = parts[1] if len(parts) >= 3 else t
    elif t.count(' - ') == 1:
        parts = t.split(' - ')
        if len(parts) == 2 and any(parts[1].strip().lower().startswith(x) for x in ['team', 'category', 'department', 'remote', 'hybrid']):
            t = parts[0]
    # Remove hashtags
    t = re.sub(r'#\w+', '', t)
    # Clean spaces
    t = re.sub(r'\s+', ' ', t).strip(' -.,;:')
    return t or title

print("Cleaning job_ads...")
conn = sqlite3.connect("jobs_database.db")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA journal_mode=MEMORY")

# Process in smaller batches without loading all into memory
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM job_ads")
total = cursor.fetchone()[0]

batch_size = 5000
for offset in tqdm(range(0, total, batch_size), desc="job_ads"):
    cursor.execute(f"SELECT id, title FROM job_ads LIMIT {batch_size} OFFSET {offset}")
    batch = cursor.fetchall()
    updates = [(clean_title(title), id) for id, title in batch]
    cursor.executemany("UPDATE job_ads SET job_title_clean=? WHERE id=?", updates)
    conn.commit()

print("\nCleaning job_title_embeddings...")
cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
total = cursor.fetchone()[0]

for offset in tqdm(range(0, total, batch_size), desc="embeddings"):
    cursor.execute(f"SELECT id, job_title FROM job_title_embeddings LIMIT {batch_size} OFFSET {offset}")
    batch = cursor.fetchall()
    updates = [(clean_title(title), id) for id, title in batch]
    cursor.executemany("UPDATE job_title_embeddings SET job_title_clean=? WHERE id=?", updates)
    conn.commit()

print("\nCreating indexes...")
try:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_ads_clean ON job_ads(job_title_clean)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_clean ON job_title_embeddings(job_title_clean)")
except: pass

conn.close()
print("\n✓ Done!")
