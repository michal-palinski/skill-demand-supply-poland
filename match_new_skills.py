#!/usr/bin/env python3
"""Match skills for new items that don't have matches yet."""

import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

REQ_RESP_DB = "req_resp_embeddings.db"
DIMENSIONS = 1024
TOP_K = 3

# Load skills
print("Loading skills...")
df = pd.read_parquet("df_skills_en_emb.parquet")
skill_embs = np.array([e[:DIMENSIONS] for e in df['emb_voyage-3-large'].values], dtype=np.float32)
norms = np.linalg.norm(skill_embs, axis=1, keepdims=True)
norms[norms == 0] = 1
skill_embs = skill_embs / norms
skill_labels = df['preferredLabel'].values.tolist()
skill_uris = df['conceptUri'].values.tolist()
skill_types = df['skillType'].values.tolist()
skills_T = skill_embs.T
print(f"  {len(skill_labels)} skills")

# Find items without matches
conn = sqlite3.connect(REQ_RESP_DB)
c = conn.cursor()

c.execute("""
    SELECT i.id, i.job_id, i.type, i.text_clean, e.embedding
    FROM items i
    JOIN embeddings e ON e.item_id = i.id
    WHERE i.id NOT IN (SELECT DISTINCT item_id FROM skill_matches)
    ORDER BY i.id
""")
rows = c.fetchall()
print(f"  {len(rows):,} items need matching")

if not rows:
    print("Nothing to do!")
    conn.close()
    exit(0)

# Parse
ids, job_ids, types, texts = [], [], [], []
embs = []
for item_id, job_id, typ, text, emb_blob in rows:
    ids.append(item_id)
    job_ids.append(job_id)
    types.append(typ)
    texts.append(text)
    embs.append(np.frombuffer(emb_blob, dtype=np.float32))

embs = np.array(embs, dtype=np.float32)
norms = np.linalg.norm(embs, axis=1, keepdims=True)
norms[norms == 0] = 1
embs = embs / norms

# Compute matches
batch_size = 500
match_rows = []

for start in tqdm(range(0, len(embs), batch_size), desc="Matching"):
    end = min(start + batch_size, len(embs))
    batch = embs[start:end]
    sims = batch @ skills_T

    top_idx = np.argpartition(sims, -TOP_K, axis=1)[:, -TOP_K:]
    for i in range(len(batch)):
        sorted_local = np.argsort(sims[i, top_idx[i]])[::-1]
        idx = start + i
        for rank, j in enumerate(sorted_local):
            skill_idx = top_idx[i][j]
            match_rows.append((
                ids[idx], job_ids[idx], types[idx], texts[idx],
                rank + 1,
                skill_labels[skill_idx],
                skill_uris[skill_idx],
                skill_types[skill_idx],
                round(float(sims[i, top_idx[i][j]]), 4)
            ))

print(f"Inserting {len(match_rows):,} matches...")
c.executemany("""
    INSERT INTO skill_matches
    (item_id, job_id, item_type, item_text, rank, skill_label, skill_uri, skill_type, similarity)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", match_rows)
conn.commit()

# Summary
c.execute("SELECT COUNT(*) FROM sample_offers")
n_offers = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM skill_matches")
n_matches = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT skill_label) FROM skill_matches")
n_skills = c.fetchone()[0]

print(f"\nTotal offers: {n_offers:,}")
print(f"Total matches: {n_matches:,}")
print(f"Unique skills: {n_skills:,}")
conn.close()
print("Done!")
