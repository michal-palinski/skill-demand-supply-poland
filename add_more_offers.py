#!/usr/bin/env python3
"""
Add 3000 more offers to req_resp_embeddings.db.
Reuses logic from embed_req_resp.py but appends to existing DB.
"""

import sqlite3
import re
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm
import voyageai

load_dotenv()

MAIN_DB = "jobs_database.db"
REQ_RESP_DB = "req_resp_embeddings.db"
MODEL = "voyage-3-large"
DIMENSIONS = 1024
BATCH_SIZE = 128
N_NEW = 3000


def clean_item(text):
    text = text.strip()
    text = re.sub(r'^[\s]*[•\-\*>●○■□▪▸‣◦►➤➢✓✔☑→⇒]+[\s]*', '', text)
    text = re.sub(r'^[\s]*[\d]+[.\)]\s*', '', text)
    text = re.sub(r'^[\s]*[a-zA-Z][.\)]\s*', '', text)
    text = text.rstrip(';').rstrip(',').strip()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_and_clean(raw_text):
    if not raw_text:
        return []
    lines = raw_text.split('\n')
    if len(lines) == 1:
        lines = re.split(r'[•●○■□▪]', raw_text)
    if len(lines) == 1:
        lines = raw_text.split(';')
    items = []
    for line in lines:
        cleaned = clean_item(line)
        if len(cleaned) >= 5:
            items.append(cleaned)
    return items


def get_existing_job_ids():
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()
    c.execute("SELECT job_id FROM sample_offers")
    ids = {row[0] for row in c.fetchall()}
    conn.close()
    return ids


def get_new_offers(existing_ids, n=N_NEW):
    conn = sqlite3.connect(MAIN_DB)
    c = conn.cursor()
    # Get random offers excluding existing ones
    placeholders = ",".join("?" * len(existing_ids))
    c.execute(f"""
        SELECT id, title, requirements, responsibilities
        FROM job_ads
        WHERE requirements IS NOT NULL AND requirements != ''
          AND responsibilities IS NOT NULL AND responsibilities != ''
          AND id NOT IN ({placeholders})
        ORDER BY RANDOM()
        LIMIT ?
    """, list(existing_ids) + [n])
    rows = c.fetchall()
    conn.close()
    print(f"Fetched {len(rows)} new offers (excluding {len(existing_ids)} existing)")
    return rows


def main():
    print(f"Adding {N_NEW} more offers to {REQ_RESP_DB}")
    print(f"Model: {MODEL}, Dimensions: {DIMENSIONS}\n")

    # 1. Get existing IDs
    existing_ids = get_existing_job_ids()
    print(f"Existing offers: {len(existing_ids)}")

    # 2. Fetch new offers
    offers = get_new_offers(existing_ids, N_NEW)

    # 3. Connect to DB and insert
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()

    total_req = 0
    total_resp = 0
    all_items = []

    for job_id, title, req_raw, resp_raw in tqdm(offers, desc="Parsing items"):
        c.execute(
            "INSERT INTO sample_offers VALUES (?, ?, ?, ?)",
            (job_id, title, req_raw, resp_raw)
        )

        for item in split_and_clean(req_raw):
            c.execute(
                "INSERT INTO items (job_id, type, text_clean) VALUES (?, 'requirement', ?)",
                (job_id, item)
            )
            all_items.append((c.lastrowid, item))
            total_req += 1

        for item in split_and_clean(resp_raw):
            c.execute(
                "INSERT INTO items (job_id, type, text_clean) VALUES (?, 'responsibility', ?)",
                (job_id, item)
            )
            all_items.append((c.lastrowid, item))
            total_resp += 1

    conn.commit()

    print(f"\nNew requirements:       {total_req:,}")
    print(f"New responsibilities:   {total_resp:,}")
    print(f"New items total:        {total_req + total_resp:,}")

    # 4. Generate embeddings
    api_key = os.getenv("VOYAGE_API_KEY")
    vo = voyageai.Client(api_key=api_key)

    total = len(all_items)
    print(f"\nGenerating embeddings for {total:,} new items...")

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
        batch = all_items[i:i + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        result = vo.embed(
            texts=texts,
            model=MODEL,
            input_type="document",
            output_dimension=DIMENSIONS
        )

        rows = []
        for item_id, emb in zip(ids, result.embeddings):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            rows.append((item_id, emb_blob))

        c.executemany("INSERT INTO embeddings (item_id, embedding) VALUES (?, ?)", rows)
        conn.commit()

    print("Embeddings saved!")

    # 5. Now match skills
    print("\nMatching skills...")
    import pandas as pd

    # Load skills
    df = pd.read_parquet("df_skills_en_emb.parquet")
    skill_embs = np.array([e[:DIMENSIONS] for e in df['emb_voyage-3-large'].values], dtype=np.float32)
    norms = np.linalg.norm(skill_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    skill_embs = skill_embs / norms
    skill_labels = df['preferredLabel'].values.tolist()
    skill_uris = df['conceptUri'].values.tolist()
    skill_types = df['skillType'].values.tolist()
    skills_T = skill_embs.T

    # Load new item embeddings
    new_ids = [item[0] for item in all_items]
    placeholders = ",".join("?" * len(new_ids))
    c.execute(f"SELECT item_id, embedding FROM embeddings WHERE item_id IN ({placeholders})", new_ids)
    emb_rows = c.fetchall()

    item_embs = np.array([np.frombuffer(row[1], dtype=np.float32) for row in emb_rows], dtype=np.float32)
    item_emb_ids = [row[0] for row in emb_rows]
    norms = np.linalg.norm(item_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    item_embs = item_embs / norms

    # Get item metadata
    id_to_meta = {}
    for item_id in new_ids:
        c.execute("SELECT job_id, type, text_clean FROM items WHERE id = ?", (item_id,))
        row = c.fetchone()
        if row:
            id_to_meta[item_id] = row

    TOP_K = 3
    batch_size = 500
    match_rows = []

    for start in tqdm(range(0, len(item_embs), batch_size), desc="Computing similarity"):
        end = min(start + batch_size, len(item_embs))
        batch = item_embs[start:end]
        sims = batch @ skills_T

        top_idx = np.argpartition(sims, -TOP_K, axis=1)[:, -TOP_K:]
        for i in range(len(batch)):
            sorted_local = np.argsort(sims[i, top_idx[i]])[::-1]
            item_id = item_emb_ids[start + i]
            meta = id_to_meta.get(item_id)
            if not meta:
                continue
            job_id, typ, text = meta
            for rank, j in enumerate(sorted_local):
                skill_idx = top_idx[i][j]
                match_rows.append((
                    item_id, job_id, typ, text,
                    rank + 1,
                    skill_labels[skill_idx],
                    skill_uris[skill_idx],
                    skill_types[skill_idx],
                    round(float(sims[i, top_idx[i][j]]), 4)
                ))

    c.executemany("""
        INSERT INTO skill_matches
        (item_id, job_id, item_type, item_text, rank, skill_label, skill_uri, skill_type, similarity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, match_rows)
    conn.commit()

    # Summary
    c.execute("SELECT COUNT(*) FROM sample_offers")
    total_offers = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM items")
    total_items = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM skill_matches")
    total_matches = c.fetchone()[0]

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Total offers now:     {total_offers:,}")
    print(f"Total items now:      {total_items:,}")
    print(f"Total skill matches:  {total_matches:,}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
