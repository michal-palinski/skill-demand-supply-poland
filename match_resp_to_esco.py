#!/usr/bin/env python3
"""
Match title_clean + responsibilities embeddings against ESCO indexes.

Naming convention for similarity columns:
  sim_<query_side>_vs_<esco_side>

  query sides:
    title      = job title_clean only          (job_title_clean.index)
    title_resp = job title_clean + resp        (job_title_clean_resp.index)

  esco sides:
    esco_label = ESCO preferredLabel only      (esco_occupations_label.index)
    esco_desc  = ESCO preferredLabel + descr   (esco_occupations_label_desc.index)

Output table: job_esco_matches_resp
  job_id                       INT
  title_clean                  TEXT
  -- title+resp  vs  esco label only
  match_title_resp_esco_label  TEXT
  uri_title_resp_esco_label    TEXT
  code_title_resp_esco_label   TEXT
  sim_title_resp_esco_label    REAL
  -- title+resp  vs  esco label+desc
  match_title_resp_esco_desc   TEXT
  uri_title_resp_esco_desc     TEXT
  code_title_resp_esco_desc    TEXT
  sim_title_resp_esco_desc     REAL
  -- title only  vs  esco label only  (reference from matches table)
  match_title_esco_label       TEXT
  sim_title_esco_label         REAL
"""

import json
import pickle
import sqlite3
import time

import faiss
import numpy as np
from tqdm import tqdm

INDEX_DIR  = "faiss_indexes"
OUT_DB     = "title_esco_matches.db"
CHUNK      = 10_000   # vectors reconstructed at once (keeps RAM ~320 MB per chunk)


def load_meta(name: str) -> dict:
    with open(f"{INDEX_DIR}/{name}_metadata.pkl", "rb") as f:
        return pickle.load(f)


def load_index(name: str) -> faiss.Index:
    idx = faiss.read_index(f"{INDEX_DIR}/{name}.index")
    print(f"  {name}: {idx.ntotal:,} vectors, d={idx.d}")
    return idx


def main() -> None:
    t0 = time.time()

    # ── load indexes & metadata ───────────────────────────────────────────────
    print("Loading indexes…")
    resp_idx   = load_index("job_title_clean_resp")
    esco_l_idx = load_index("esco_occupations_label")
    esco_d_idx = load_index("esco_occupations_label_desc")

    resp_meta   = load_meta("job_title_clean_resp")
    esco_l_meta = load_meta("esco_occupations_label")
    esco_d_meta = load_meta("esco_occupations_label_desc")

    # title-only matches for comparison (already computed)
    title_matches: dict[str, tuple] = {}  # title_clean → (esco_label, sim)
    conn_r = sqlite3.connect(OUT_DB)
    rows = conn_r.execute(
        "SELECT title_clean, esco_label, sim_label FROM matches"
    ).fetchall()
    conn_r.close()
    for title, lbl, sim in rows:
        title_matches[title] = (lbl, sim)
    print(f"  loaded {len(title_matches):,} title-only reference matches")

    n_total = resp_idx.ntotal

    # ── prepare output DB ─────────────────────────────────────────────────────
    conn = sqlite3.connect(OUT_DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_esco_matches_resp (
            job_id                      INTEGER PRIMARY KEY,
            title_clean                 TEXT,
            match_title_resp_esco_label TEXT,
            uri_title_resp_esco_label   TEXT,
            code_title_resp_esco_label  TEXT,
            sim_title_resp_esco_label   REAL,
            match_title_resp_esco_desc  TEXT,
            uri_title_resp_esco_desc    TEXT,
            code_title_resp_esco_desc   TEXT,
            sim_title_resp_esco_desc    REAL,
            match_title_esco_label      TEXT,
            sim_title_esco_label        REAL
        )
    """)

    # find where we left off (resume support)
    done_count = conn.execute(
        "SELECT COUNT(*) FROM job_esco_matches_resp"
    ).fetchone()[0]
    start = done_count
    if start > 0:
        print(f"  resuming from row {start:,}")

    # ── chunk through resp index ──────────────────────────────────────────────
    print(f"\nMatching {n_total:,} job vectors against ESCO…")

    buf = np.empty((CHUNK, resp_idx.d), dtype=np.float32)

    with tqdm(total=n_total - start, initial=0, unit="job") as bar:
        for chunk_start in range(start, n_total, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n_total)
            size      = chunk_end - chunk_start

            resp_idx.reconstruct_n(chunk_start, size, buf[:size])

            D_l, I_l = esco_l_idx.search(buf[:size], 1)
            D_d, I_d = esco_d_idx.search(buf[:size], 1)

            rows_out = []
            for i in range(size):
                global_i   = chunk_start + i
                job_id     = resp_meta["ids"][global_i]
                title      = resp_meta["labels"][global_i]

                li = int(I_l[i, 0])
                di = int(I_d[i, 0])

                ref = title_matches.get(title, (None, None))

                rows_out.append((
                    job_id,
                    title,
                    # title+resp vs esco label
                    esco_l_meta["labels"][li],
                    esco_l_meta["uris"][li],
                    esco_l_meta["codes"][li],
                    float(D_l[i, 0]),
                    # title+resp vs esco label+desc
                    esco_d_meta["labels"][di],
                    esco_d_meta["uris"][di],
                    esco_d_meta["codes"][di],
                    float(D_d[i, 0]),
                    # title only vs esco label (reference)
                    ref[0],
                    ref[1],
                ))

            conn.execute("BEGIN")
            conn.executemany(
                "INSERT OR REPLACE INTO job_esco_matches_resp VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?)",
                rows_out,
            )
            conn.execute("COMMIT")
            bar.update(size)

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s → {OUT_DB} :: job_esco_matches_resp")

    # ── quick sample ──────────────────────────────────────────────────────────
    conn = sqlite3.connect(OUT_DB)
    sample = conn.execute("""
        SELECT title_clean,
               match_title_resp_esco_label, sim_title_resp_esco_label,
               match_title_resp_esco_desc,  sim_title_resp_esco_desc,
               match_title_esco_label,      sim_title_esco_label
        FROM job_esco_matches_resp
        ORDER BY RANDOM() LIMIT 8
    """).fetchall()
    conn.close()

    h = f"{'title_clean':38}  {'title+resp→esco_label':26}  s_rl  {'title+resp→esco_desc':26}  s_rd  {'title→esco_label':26}  s_tl"
    print(f"\n{h}")
    print("-" * len(h))
    for r in sample:
        print(
            f"{str(r[0])[:38]:38}  {str(r[1] or '')[:26]:26}  {r[2]:.3f}"
            f"  {str(r[3] or '')[:26]:26}  {r[4]:.3f}"
            f"  {str(r[5] or '')[:26]:26}  {r[6] or 0:.3f}"
        )


if __name__ == "__main__":
    main()
