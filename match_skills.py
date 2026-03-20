#!/usr/bin/env python3
"""
Match req/resp embeddings to ESCO skills using cosine similarity.
Skills use voyage-3-large at 2048d → truncate to 1024d (Matryoshka).
Req/resp already at 1024d voyage-3-large.
"""

import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

REQ_RESP_DB = "req_resp_embeddings.db"
SKILLS_PARQUET = "df_skills_en_emb.parquet"
EMB_COL = "emb_voyage-3-large"
TARGET_DIM = 1024
TOP_K = 3


def load_skills():
    """Load ESCO skills with truncated & normalized embeddings."""
    print("Loading skills parquet...")
    df = pd.read_parquet(SKILLS_PARQUET)
    print(f"  {len(df):,} skills loaded")

    # Extract embeddings, truncate to 1024d, normalize
    embs = np.array([e[:TARGET_DIM] for e in df[EMB_COL].values], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    labels = df['preferredLabel'].values.tolist()
    uris = df['conceptUri'].values.tolist()
    skill_types = df['skillType'].values.tolist()

    print(f"  Embeddings shape: {embs.shape}")
    return labels, uris, skill_types, embs


def load_req_resp_embeddings():
    """Load req/resp items and their embeddings."""
    print("Loading req/resp embeddings...")
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()

    c.execute("""
        SELECT i.id, i.job_id, i.type, i.text_clean, e.embedding
        FROM items i
        JOIN embeddings e ON e.item_id = i.id
        ORDER BY i.id
    """)
    rows = c.fetchall()
    conn.close()

    ids = []
    job_ids = []
    types = []
    texts = []
    embs = []

    for item_id, job_id, typ, text, emb_blob in rows:
        ids.append(item_id)
        job_ids.append(job_id)
        types.append(typ)
        texts.append(text)
        embs.append(np.frombuffer(emb_blob, dtype=np.float32))

    embs = np.array(embs, dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms

    print(f"  {len(ids):,} items loaded, embeddings shape: {embs.shape}")
    return ids, job_ids, types, texts, embs


def compute_matches(item_embs, skill_embs, top_k=TOP_K, batch_size=500):
    """Compute top-K skill matches for each item using batched matmul."""
    n_items = item_embs.shape[0]
    all_top_indices = np.zeros((n_items, top_k), dtype=np.int32)
    all_top_scores = np.zeros((n_items, top_k), dtype=np.float32)

    # skills_embs is (n_skills, dim), transpose for matmul
    skills_T = skill_embs.T  # (dim, n_skills)

    for start in tqdm(range(0, n_items, batch_size), desc="Computing similarity"):
        end = min(start + batch_size, n_items)
        batch = item_embs[start:end]  # (batch, dim)

        # Cosine similarity (both normalized)
        sims = batch @ skills_T  # (batch, n_skills)

        # Top-K per row
        top_idx = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        # Sort within top-K
        for i in range(len(batch)):
            sorted_local = np.argsort(sims[i, top_idx[i]])[::-1]
            all_top_indices[start + i] = top_idx[i][sorted_local]
            all_top_scores[start + i] = sims[i, top_idx[i][sorted_local]]

    return all_top_indices, all_top_scores


def save_results(item_ids, job_ids, types, texts,
                 top_indices, top_scores,
                 skill_labels, skill_uris, skill_types):
    """Save skill matches to the req_resp database."""
    print("Saving results...")
    conn = sqlite3.connect(REQ_RESP_DB)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS skill_matches")
    c.execute("""
        CREATE TABLE skill_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            item_type TEXT NOT NULL,
            item_text TEXT NOT NULL,
            rank INTEGER NOT NULL,
            skill_label TEXT NOT NULL,
            skill_uri TEXT,
            skill_type TEXT,
            similarity REAL NOT NULL,
            FOREIGN KEY (item_id) REFERENCES items(id)
        )
    """)

    rows = []
    for i in tqdm(range(len(item_ids)), desc="Building rows"):
        for rank in range(TOP_K):
            skill_idx = top_indices[i, rank]
            rows.append((
                item_ids[i],
                job_ids[i],
                types[i],
                texts[i],
                rank + 1,
                skill_labels[skill_idx],
                skill_uris[skill_idx],
                skill_types[skill_idx],
                round(float(top_scores[i, rank]), 4)
            ))

    c.executemany("""
        INSERT INTO skill_matches
        (item_id, job_id, item_type, item_text, rank, skill_label, skill_uri, skill_type, similarity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    c.execute("CREATE INDEX idx_sm_job_id ON skill_matches(job_id)")
    c.execute("CREATE INDEX idx_sm_item_id ON skill_matches(item_id)")
    c.execute("CREATE INDEX idx_sm_skill ON skill_matches(skill_label)")

    conn.commit()

    # Summary
    c.execute("SELECT COUNT(*) FROM skill_matches")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT skill_label) FROM skill_matches")
    unique_skills = c.fetchone()[0]
    c.execute("SELECT AVG(similarity) FROM skill_matches WHERE rank = 1")
    avg_top1 = c.fetchone()[0]

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Total matches:          {total:,}")
    print(f"Unique skills matched:  {unique_skills:,}")
    print(f"Avg top-1 similarity:   {avg_top1:.4f}")

    # Examples
    c.execute("""
        SELECT item_type, item_text, skill_label, similarity
        FROM skill_matches
        WHERE rank = 1
        ORDER BY similarity DESC
        LIMIT 10
    """)
    print(f"\nTop matches:")
    for typ, text, skill, sim in c.fetchall():
        tag = "REQ " if typ == "requirement" else "RESP"
        print(f"  [{tag}] {text[:50]:50s} → {skill[:40]:40s} ({sim:.3f})")

    conn.close()


def main():
    # 1. Load data
    skill_labels, skill_uris, skill_types, skill_embs = load_skills()
    item_ids, job_ids, types, texts, item_embs = load_req_resp_embeddings()

    # 2. Compute similarity
    top_indices, top_scores = compute_matches(item_embs, skill_embs)

    # 3. Save
    save_results(
        item_ids, job_ids, types, texts,
        top_indices, top_scores,
        skill_labels, skill_uris, skill_types
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
