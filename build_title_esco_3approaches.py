#!/usr/bin/env python3
"""
Build a single DB with unique title_clean matched to ESCO occupations via 3 approaches:
  1. label   — title_clean embedding  vs  ESCO label embedding
  2. combined — title+resp embedding  vs  ESCO label+desc embedding  (best per title)
  3. context  — contextual embedding  vs  ESCO contextual embedding  (best per title)

Schema:
  title_clean, esco_label, esco_combined, esco_contextual,
  score_label, score_combined, score_contextual
"""

import os, pickle, sqlite3
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict

IDX_DIR = "faiss_indexes"
OUT_DB  = "title_esco_3approaches.db"
CHUNK   = 10_000


def load(name):
    idx = faiss.read_index(os.path.join(IDX_DIR, f"{name}.index"))
    with open(os.path.join(IDX_DIR, f"{name}_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return idx, meta


def search_unique(job_idx, job_labels, esco_idx, esco_labels, desc=""):
    """Search each unique-title vector against ESCO. Returns {title: (esco_label, score)}."""
    n = len(job_labels)
    result = {}
    for start in tqdm(range(0, n, CHUNK), desc=desc, unit="chunk"):
        end = min(start + CHUNK, n)
        vecs = job_idx.reconstruct_n(start, end - start)
        dists, idxs = esco_idx.search(vecs, 1)
        for i in range(end - start):
            title = job_labels[start + i]
            esco_i = int(idxs[i, 0])
            if esco_i < len(esco_labels):
                result[title] = (esco_labels[esco_i], round(float(dists[i, 0]), 2))
    return result


def search_rows_aggregated(job_idx, job_labels, esco_idx, esco_labels, desc=""):
    """
    Search all per-row vectors against ESCO (top-1 per row).
    Aggregate by title_clean: keep the ESCO match with the highest score.
    Returns {title: (esco_label, score)}.
    """
    n = len(job_labels)
    best = {}
    for start in tqdm(range(0, n, CHUNK), desc=desc, unit="chunk"):
        end = min(start + CHUNK, n)
        vecs = job_idx.reconstruct_n(start, end - start)
        dists, idxs = esco_idx.search(vecs, 1)
        for i in range(end - start):
            title = job_labels[start + i]
            score = round(float(dists[i, 0]), 2)
            esco_i = int(idxs[i, 0])
            if esco_i < len(esco_labels):
                esco_lbl = esco_labels[esco_i]
                if title not in best or score > best[title][1]:
                    best[title] = (esco_lbl, score)
    return best


def main():
    # ── Load ESCO indexes ────────────────────────────────────────
    print("Loading ESCO indexes …")
    esco_l_idx, esco_l_meta = load("esco_occupations_label")
    esco_d_idx, esco_d_meta = load("esco_occupations_label_desc")
    esco_c_idx, esco_c_meta = load("esco_occupations_contextual_2048")

    esco_l_labels = esco_l_meta["labels"]
    esco_d_labels = esco_d_meta["labels"]
    esco_c_labels = esco_c_meta["labels"]

    # ── Load job indexes ─────────────────────────────────────────
    print("Loading job indexes …")
    job_u_idx, job_u_meta = load("job_title_clean")
    job_r_idx, job_r_meta = load("job_title_clean_resp")
    job_ctx_idx, job_ctx_meta = load("job_title_resp_contextual_2048")

    job_u_labels = job_u_meta["labels"]
    job_r_labels = job_r_meta["labels"]
    job_ctx_labels = job_ctx_meta["labels"]

    # ── Approach 1: label ────────────────────────────────────────
    print("\n[1/3] Approach: label (unique title vs ESCO label)")
    a1 = search_unique(job_u_idx, job_u_labels, esco_l_idx, esco_l_labels,
                       desc="label")

    # ── Approach 2: combined (title+resp vs ESCO label+desc) ─────
    print("\n[2/3] Approach: combined (title+resp vs ESCO label+desc)")
    a2 = search_rows_aggregated(job_r_idx, job_r_labels, esco_d_idx, esco_d_labels,
                                desc="combined")

    # ── Approach 3: contextual ───────────────────────────────────
    print("\n[3/3] Approach: contextual (ctx title+resp vs ctx ESCO)")
    a3 = search_rows_aggregated(job_ctx_idx, job_ctx_labels, esco_c_idx, esco_c_labels,
                                desc="contextual")

    # ── Merge & save ─────────────────────────────────────────────
    all_titles = sorted(set(a1.keys()) | set(a2.keys()) | set(a3.keys()))
    print(f"\nUnique titles: {len(all_titles):,}")

    conn = sqlite3.connect(OUT_DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("DROP TABLE IF EXISTS matches")
    conn.execute("""
        CREATE TABLE matches (
            title_clean       TEXT PRIMARY KEY,
            esco_label        TEXT,
            esco_combined     TEXT,
            esco_contextual   TEXT,
            score_label       REAL,
            score_combined    REAL,
            score_contextual  REAL
        )
    """)

    rows = []
    for t in all_titles:
        e1, s1 = a1.get(t, (None, None))
        e2, s2 = a2.get(t, (None, None))
        e3, s3 = a3.get(t, (None, None))
        rows.append((t, e1, e2, e3, s1, s2, s3))

    conn.execute("BEGIN")
    conn.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?)", rows)
    conn.execute("COMMIT")
    conn.execute("VACUUM")
    conn.close()

    size_mb = os.path.getsize(OUT_DB) / 1_048_576
    print(f"\nSaved: {OUT_DB}  ({size_mb:.1f} MB, {len(rows):,} rows)")

    # sample
    conn = sqlite3.connect(OUT_DB)
    for row in conn.execute("SELECT * FROM matches ORDER BY RANDOM() LIMIT 5"):
        print(f"  {row[0]}")
        print(f"    label:  {row[1]}  ({row[4]})")
        print(f"    comb:   {row[2]}  ({row[5]})")
        print(f"    ctx:    {row[3]}  ({row[6]})")
    conn.close()


if __name__ == "__main__":
    main()
