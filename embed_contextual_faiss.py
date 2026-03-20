#!/usr/bin/env python3
"""
Contextual chunk embeddings using voyage-context-3 (2048d) → FAISS indexes.

Each "document" = list of chunks fed together so every chunk embedding
carries context from its siblings.

INDEX 1 — ESCO occupations (3 043 items)
  Input:  [preferredLabel, description]
  Keep:   embedding of chunk 0 (label — enriched with desc context)
  Output: faiss_indexes/esco_occupations_contextual.index
          faiss_indexes/esco_occupations_contextual_metadata.pkl

INDEX 2 — Job ads title_clean + responsibilities (749 569 rows)
  Input:  [title_clean, responsibilities_clean]
  Keep:   embedding of chunk 0 (title — enriched with resp context)
  Output: faiss_indexes/job_title_resp_contextual.index
          faiss_indexes/job_title_resp_contextual_metadata.pkl
  Streaming + checkpoints every CKPT_EVERY batches.
"""

import os
import pickle
import re
import sqlite3
import time

import faiss
import numpy as np
import pandas as pd
import voyageai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH   = "jobs_database.db"
ESCO_CSV  = "ESCO dataset - v1.2.1 - classification - pl - csv/occupations_pl.csv"
INDEX_DIR = "faiss_indexes"
import argparse

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, default=2048, choices=[256, 512, 1024, 2048])
    return p.parse_args()

_ARGS = _parse_args()

MODEL     = "voyage-context-3"
DIMS      = _ARGS.dims
BATCH     = 100     # documents per API call (API limit: 1 000)
CKPT_EVERY = 300    # save FAISS checkpoint every N batches (~30 000 rows)
MAX_RESP_CHARS = 6000  # truncate very long responsibilities

ESCO_INDEX = os.path.join(INDEX_DIR, f"esco_occupations_contextual_{DIMS}.index")
ESCO_META  = os.path.join(INDEX_DIR, f"esco_occupations_contextual_{DIMS}_metadata.pkl")
JOB_INDEX  = os.path.join(INDEX_DIR, f"job_title_resp_contextual_{DIMS}.index")
JOB_META   = os.path.join(INDEX_DIR, f"job_title_resp_contextual_{DIMS}_metadata.pkl")
# ──────────────────────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(
    r"(?m)^[\s]*(?:[•◦▪▸►‣\-–—*·○●■□▶]|(?:\d{1,2}|[a-zA-Z])[.)]\s)\s*"
)

def clean_resp(text: str) -> str:
    if not text:
        return ""
    t = _BULLET_RE.sub("", text)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def init_voyage() -> voyageai.Client:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    return voyageai.Client(api_key=api_key)


def save_index(index: faiss.Index, meta: dict, idx_path: str, meta_path: str) -> None:
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  {index.ntotal:,} vectors → {idx_path}")


def embed_batch_contextual(
    vo: voyageai.Client,
    batch: list[list[str]],     # list of chunk-lists
    retries: int = 4,
) -> list[list[float]]:
    """Call contextualized_embed, return list of label (chunk-0) embeddings."""
    for attempt in range(retries):
        try:
            result = vo.contextualized_embed(
                inputs=batch,
                model=MODEL,
                input_type="document",
                output_dimension=DIMS,
            )
            return [r.embeddings[0] for r in result.results]
        except Exception as e:
            wait = 3 * (attempt + 1)
            if attempt < retries - 1:
                print(f"\n  ⚠ {e} — retry in {wait}s…")
                time.sleep(wait)
            else:
                raise


# ── INDEX 1: ESCO occupations ─────────────────────────────────────────────────

def build_esco_index(vo: voyageai.Client) -> None:
    print("\n[1/2] ESCO occupations contextual embedding…")

    df = pd.read_csv(ESCO_CSV)
    df["description"] = df["description"].fillna("")
    df = df[df["preferredLabel"].notna() & (df["preferredLabel"] != "")].reset_index(drop=True)
    print(f"  {len(df):,} occupations loaded")

    labels = df["preferredLabel"].tolist()
    descs  = df["description"].tolist()
    uris   = df["conceptUri"].tolist()
    codes  = df["code"].tolist()

    # resume if index already exists
    index = faiss.IndexFlatIP(DIMS)
    start_batch = 0
    if os.path.exists(ESCO_INDEX):
        index = faiss.read_index(ESCO_INDEX)
        start_batch = index.ntotal // BATCH
        print(f"  resuming from batch {start_batch} ({index.ntotal:,} already done)")

    # build chunk-list inputs: [label, description] (or just [label] if no desc)
    inputs = [
        [lbl, desc] if desc and len(desc) > 10 else [lbl]
        for lbl, desc in zip(labels, descs)
    ]

    total_batches = (len(inputs) + BATCH - 1) // BATCH

    with tqdm(total=total_batches, initial=start_batch, desc="  batches", unit="batch") as bar:
        for b in range(start_batch, total_batches):
            lo, hi  = b * BATCH, min((b + 1) * BATCH, len(inputs))
            batch   = inputs[lo:hi]
            embeds  = embed_batch_contextual(vo, batch)
            arr     = np.array(embeds, dtype=np.float32)
            faiss.normalize_L2(arr)
            index.add(arr)
            bar.update(1)
            time.sleep(0.05)

    save_index(index, {
        "labels":     labels,
        "uris":       uris,
        "codes":      codes,
        "count":      len(labels),
        "model":      MODEL,
        "dimensions": DIMS,
        "type":       "esco_occupations_contextual",
    }, ESCO_INDEX, ESCO_META)
    print("  ESCO index done.")


# ── INDEX 2: job title_clean + responsibilities ───────────────────────────────

def build_job_index(vo: voyageai.Client) -> None:
    print("\n[2/2] Job title_clean + responsibilities contextual embedding…")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title_clean, responsibilities "
        "FROM job_ads WHERE title_clean IS NOT NULL AND title_clean != '' "
        "ORDER BY id"
    ).fetchall()
    conn.close()
    print(f"  {len(rows):,} rows loaded from DB")

    all_ids    = [r[0] for r in rows]
    all_titles = [r[1] for r in rows]
    all_resp   = [clean_resp(r[2] or "")[:MAX_RESP_CHARS] for r in rows]

    # chunk-list: [title, resp] or just [title] if resp empty
    inputs = [
        [t, r] if r else [t]
        for t, r in zip(all_titles, all_resp)
    ]

    # resume
    index = faiss.IndexFlatIP(DIMS)
    start_batch = 0
    if os.path.exists(JOB_INDEX):
        index = faiss.read_index(JOB_INDEX)
        start_batch = index.ntotal // BATCH
        print(f"  resuming from batch {start_batch} ({index.ntotal:,} already done)")

    total_batches = (len(inputs) + BATCH - 1) // BATCH

    with tqdm(total=total_batches, initial=start_batch, desc="  batches", unit="batch") as bar:
        for b in range(start_batch, total_batches):
            lo, hi = b * BATCH, min((b + 1) * BATCH, len(inputs))
            batch  = inputs[lo:hi]
            embeds = embed_batch_contextual(vo, batch)
            arr    = np.array(embeds, dtype=np.float32)
            faiss.normalize_L2(arr)
            index.add(arr)
            bar.update(1)

            # checkpoint
            if (b + 1) % CKPT_EVERY == 0:
                n_done = index.ntotal
                faiss.write_index(index, JOB_INDEX)
                with open(JOB_META, "wb") as f:
                    pickle.dump({
                        "ids":        all_ids[:n_done],
                        "labels":     all_titles[:n_done],
                        "count":      n_done,
                        "model":      MODEL,
                        "dimensions": DIMS,
                        "type":       "job_title_resp_contextual",
                    }, f)

            time.sleep(0.05)

    save_index(index, {
        "ids":        all_ids,
        "labels":     all_titles,
        "count":      len(all_ids),
        "model":      MODEL,
        "dimensions": DIMS,
        "type":       "job_title_resp_contextual",
    }, JOB_INDEX, JOB_META)
    print("  Job index done.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    vo = init_voyage()

    t0 = time.time()
    build_esco_index(vo)
    build_job_index(vo)
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
