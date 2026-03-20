#!/usr/bin/env python3
"""
Create Voyage AI voyage-4 embeddings (2048 dims) for job_ads titles.

Two FAISS indexes:
  faiss_indexes/job_title_clean.index          – unique title_clean values
  faiss_indexes/job_title_clean_resp.index     – title_clean + responsibilities (all rows)

Metadata pickles carry 'ids' so every vector maps back to job_ads.id.
"""

import os
import pickle
import re
import sqlite3
import time

import faiss
import numpy as np
import voyageai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH   = "jobs_database.db"
INDEX_DIR = "faiss_indexes"
MODEL     = "voyage-4"
DIMS      = 2048
BATCH     = 128          # voyage-4 max per request
MAX_CHARS = 8000         # safety truncation for very long combined text

LABEL_INDEX = os.path.join(INDEX_DIR, "job_title_clean.index")
LABEL_META  = os.path.join(INDEX_DIR, "job_title_clean_metadata.pkl")
RESP_INDEX  = os.path.join(INDEX_DIR, "job_title_clean_resp.index")
RESP_META   = os.path.join(INDEX_DIR, "job_title_clean_resp_metadata.pkl")
# ──────────────────────────────────────────────────────────────────────────────

# bullet chars + numbering patterns at start of line
_BULLET_RE = re.compile(
    r"(?m)^[\s]*"                             # optional leading whitespace
    r"(?:[•◦▪▸►‣\-–—*·○●■□▶]"              # bullet symbols
    r"|(?:\d{1,2}|[a-zA-Z])[.)]\s"           # 1. 2. a. b)
    r")\s*"
)

def clean_resp(text: str) -> str:
    """Strip bullet markers; join lines with space; collapse whitespace."""
    if not text:
        return ""
    t = _BULLET_RE.sub("", text)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


CHECKPOINT_EVERY = 500   # save index to disk every N API batches


def embed_into_index(
    vo: voyageai.Client,
    texts: list[str],
    index: faiss.Index,
    *,
    start_batch: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_meta_fn=None,
) -> None:
    """
    Embed texts in API batches and add each batch directly to `index`.
    Never accumulates more than one batch of Python floats in memory.

    Args:
        start_batch: resume from this batch index (skip already-embedded texts).
        checkpoint_path: if set, save index + metadata every CHECKPOINT_EVERY batches.
        checkpoint_meta_fn: callable() → dict  called just before each checkpoint save.
    """
    total_batches = (len(texts) + BATCH - 1) // BATCH

    with tqdm(total=total_batches, initial=start_batch,
              desc="  batches", unit="batch") as bar:
        for b in range(start_batch, total_batches):
            lo, hi = b * BATCH, min((b + 1) * BATCH, len(texts))
            batch   = texts[lo:hi]

            result  = vo.embed(
                texts=batch,
                model=MODEL,
                input_type="document",
                output_dimension=DIMS,
            )

            # convert + normalise immediately — never keep Python float lists
            arr = np.array(result.embeddings, dtype=np.float32)
            faiss.normalize_L2(arr)
            index.add(arr)
            bar.update(1)

            # periodic checkpoint
            if checkpoint_path and (b + 1) % CHECKPOINT_EVERY == 0:
                faiss.write_index(index, checkpoint_path)
                if checkpoint_meta_fn:
                    meta = checkpoint_meta_fn(processed_batches=b + 1)
                    with open(checkpoint_path.replace(".index", "_metadata.pkl"), "wb") as f:
                        pickle.dump(meta, f)

            if hi < len(texts):
                time.sleep(0.05)


def save_index(index: faiss.Index, meta: dict, idx_path: str, meta_path: str) -> None:
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  {index.ntotal:,} vectors → {idx_path}")
    print(f"  metadata     → {meta_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set")

    vo = voyageai.Client(api_key=api_key)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data from DB…")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title_clean, responsibilities "
        "FROM job_ads "
        "WHERE title_clean IS NOT NULL AND title_clean != '' "
        "ORDER BY id"
    ).fetchall()
    conn.close()
    print(f"  {len(rows):,} rows loaded")

    # ────────────────────────────────────────────────────────────────────────
    # INDEX 1 – unique title_clean
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1/2] Building unique title_clean index…")

    # Group ids by title_clean
    label_to_ids: dict[str, list[int]] = {}
    for job_id, title, _ in rows:
        label_to_ids.setdefault(title, []).append(job_id)

    unique_labels  = list(label_to_ids.keys())          # preserve insertion order
    unique_ids     = [label_to_ids[lbl] for lbl in unique_labels]

    print(f"  {len(unique_labels):,} unique titles")
    idx1 = faiss.IndexFlatIP(DIMS)

    # resume if checkpoint exists
    start_b1 = 0
    if os.path.exists(LABEL_INDEX):
        idx1 = faiss.read_index(LABEL_INDEX)
        start_b1 = idx1.ntotal // BATCH
        print(f"  resuming from batch {start_b1} ({idx1.ntotal:,} already embedded)")

    embed_into_index(
        vo, unique_labels, idx1,
        start_batch=start_b1,
        checkpoint_path=LABEL_INDEX,
        checkpoint_meta_fn=lambda processed_batches: {
            "labels": unique_labels, "ids": unique_ids,
            "count": len(unique_labels), "model": MODEL,
            "dimensions": DIMS, "type": "title_clean_unique",
        },
    )

    save_index(
        idx1,
        {
            "labels":     unique_labels,
            "ids":        unique_ids,
            "count":      len(unique_labels),
            "model":      MODEL,
            "dimensions": DIMS,
            "type":       "title_clean_unique",
        },
        LABEL_INDEX,
        LABEL_META,
    )

    # ────────────────────────────────────────────────────────────────────────
    # INDEX 2 – title_clean + responsibilities  (every row)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[2/2] Building title_clean + responsibilities index…")

    all_ids:    list[int] = []
    all_titles: list[str] = []
    all_texts:  list[str] = []

    for job_id, title, resp in rows:
        resp_clean = clean_resp(resp or "")
        combined   = (title + " " + resp_clean).strip()[:MAX_CHARS]
        all_ids.append(job_id)
        all_titles.append(title)
        all_texts.append(combined)

    print(f"  {len(all_texts):,} texts (title + responsibilities)")
    idx2 = faiss.IndexFlatIP(DIMS)

    # resume if checkpoint exists
    start_b2 = 0
    if os.path.exists(RESP_INDEX):
        idx2 = faiss.read_index(RESP_INDEX)
        start_b2 = idx2.ntotal // BATCH
        print(f"  resuming from batch {start_b2} ({idx2.ntotal:,} already embedded)")

    embed_into_index(
        vo, all_texts, idx2,
        start_batch=start_b2,
        checkpoint_path=RESP_INDEX,
        checkpoint_meta_fn=lambda processed_batches: {
            "ids": all_ids[:processed_batches * BATCH],
            "labels": all_titles[:processed_batches * BATCH],
            "count": processed_batches * BATCH,
            "model": MODEL, "dimensions": DIMS,
            "type": "title_clean_responsibilities",
        },
    )

    save_index(
        idx2,
        {
            "ids":        all_ids,
            "labels":     all_titles,
            "count":      len(all_ids),
            "model":      MODEL,
            "dimensions": DIMS,
            "type":       "title_clean_responsibilities",
        },
        RESP_INDEX,
        RESP_META,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
