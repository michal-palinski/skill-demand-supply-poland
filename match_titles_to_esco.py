#!/usr/bin/env python3
"""
Match every unique title_clean to the best ESCO occupation via FAISS.

Sources
-------
  job_title_clean.index            — 168 264 unique title_clean vectors
  esco_occupations_label.index     — 3 043 ESCO vectors (label only)
  esco_occupations_label_desc.index— 3 043 ESCO vectors (label + description)

Output
------
  title_esco_matches.db
    table: matches
      title_clean       TEXT  — unique job title (cleaned)
      job_count         INT   — how many job_ads rows share this title
      job_ids           TEXT  — JSON array of job_ads.id
      esco_label        TEXT  — best ESCO label  (matched via label index)
      esco_uri_label    TEXT
      esco_code_label   TEXT
      sim_label         REAL  — cosine similarity (label index)
      esco_label_desc   TEXT  — best ESCO label  (matched via label+desc index)
      esco_uri_desc     TEXT
      esco_code_desc    TEXT
      sim_label_desc    REAL  — cosine similarity (label+desc index)
"""

import json
import pickle
import sqlite3
import time

import faiss
import numpy as np
from tqdm import tqdm

INDEX_DIR = "faiss_indexes"
OUT_DB    = "title_esco_matches.db"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_index_and_meta(name: str):
    idx  = faiss.read_index(f"{INDEX_DIR}/{name}.index")
    with open(f"{INDEX_DIR}/{name}_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    print(f"  loaded {name}: {idx.ntotal:,} vectors")
    return idx, meta


def reconstruct_all(index: faiss.Index) -> np.ndarray:
    """Pull every vector out of a FlatIP index as a (n, d) array."""
    n, d = index.ntotal, index.d
    vecs = np.empty((n, d), dtype=np.float32)
    # reconstruct_n is much faster than a loop
    index.reconstruct_n(0, n, vecs)
    return vecs


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── load indexes ──────────────────────────────────────────────────────────
    print("Loading indexes…")
    job_idx,   job_meta   = load_index_and_meta("job_title_clean")
    esco_l_idx, esco_l_meta = load_index_and_meta("esco_occupations_label")
    esco_d_idx, esco_d_meta = load_index_and_meta("esco_occupations_label_desc")

    # ── reconstruct job title vectors ─────────────────────────────────────────
    print("\nReconstructing job title vectors…")
    job_vecs = reconstruct_all(job_idx)
    print(f"  shape: {job_vecs.shape}")

    # ── search: title vs ESCO label ───────────────────────────────────────────
    print("\nSearching title_clean vs ESCO label index…")
    t1 = time.time()
    D_label, I_label = esco_l_idx.search(job_vecs, 1)   # top-1
    print(f"  done in {time.time()-t1:.1f}s")

    # ── search: title vs ESCO label+desc ─────────────────────────────────────
    print("Searching title_clean vs ESCO label+desc index…")
    t2 = time.time()
    D_desc, I_desc = esco_d_idx.search(job_vecs, 1)
    print(f"  done in {time.time()-t2:.1f}s")

    # ── build output DB ───────────────────────────────────────────────────────
    print(f"\nWriting results to {OUT_DB}…")
    conn = sqlite3.connect(OUT_DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            title_clean     TEXT PRIMARY KEY,
            job_count       INTEGER,
            job_ids         TEXT,       -- JSON array of job_ads.id
            esco_label      TEXT,       -- best match via label index
            esco_uri_label  TEXT,
            esco_code_label TEXT,
            sim_label       REAL,
            esco_label_desc TEXT,       -- best match via label+desc index
            esco_uri_desc   TEXT,
            esco_code_desc  TEXT,
            sim_label_desc  REAL
        )
    """)
    conn.execute("DELETE FROM matches")

    rows = []
    for i, title in enumerate(tqdm(job_meta["labels"], unit="title")):
        ids = job_meta["ids"][i]

        li   = int(I_label[i, 0])
        di   = int(I_desc[i, 0])

        rows.append((
            title,
            len(ids),
            json.dumps(ids),
            esco_l_meta["labels"][li],
            esco_l_meta["uris"][li],
            esco_l_meta["codes"][li],
            float(D_label[i, 0]),
            esco_d_meta["labels"][di],
            esco_d_meta["uris"][di],
            esco_d_meta["codes"][di],
            float(D_desc[i, 0]),
        ))

    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute("COMMIT")
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s → {OUT_DB}")
    print(f"  {len(rows):,} rows written")

    # ── quick sanity check ────────────────────────────────────────────────────
    conn = sqlite3.connect(OUT_DB)
    sample = conn.execute(
        "SELECT title_clean, esco_label, sim_label, esco_label_desc, sim_label_desc "
        "FROM matches ORDER BY RANDOM() LIMIT 10"
    ).fetchall()
    conn.close()

    print("\nSample matches:")
    print(f"{'title_clean':45s}  {'esco_label':30s}  sim_l  {'esco_label_desc':30s}  sim_d")
    print("-" * 130)
    for r in sample:
        print(f"{r[0][:45]:45s}  {r[1][:30]:30s}  {r[2]:.3f}  {r[3][:30]:30s}  {r[4]:.3f}")


if __name__ == "__main__":
    main()
