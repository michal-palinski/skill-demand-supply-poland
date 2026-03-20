#!/usr/bin/env python3
"""
Async version of embed_contextual_faiss.py.

Uses aiohttp with CONCURRENCY concurrent requests to the Voyage API,
giving ~CONCURRENCY× throughput vs the synchronous version.

INDEX 1 — ESCO occupations
  faiss_indexes/esco_occupations_contextual_{DIMS}.index

INDEX 2 — Job ads title_clean + responsibilities
  faiss_indexes/job_title_resp_contextual_{DIMS}.index

Usage:
    python embed_contextual_faiss_async.py --dims 2048
    python embed_contextual_faiss_async.py --dims 1024
"""

import argparse
import asyncio
import os
import pickle
import re
import sqlite3
import time

import aiohttp
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dims",        type=int, default=2048, choices=[256, 512, 1024, 2048])
    p.add_argument("--concurrency", type=int, default=8,
                   help="concurrent API requests (default 8)")
    p.add_argument("--batch",       type=int, default=100,
                   help="documents per API request (default 100)")
    p.add_argument("--esco-only",   action="store_true")
    p.add_argument("--jobs-only",   action="store_true")
    return p.parse_args()

ARGS = _parse()

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = "jobs_database.db"
ESCO_CSV       = "ESCO dataset - v1.2.1 - classification - pl - csv/occupations_pl.csv"
INDEX_DIR      = "faiss_indexes"
MODEL          = "voyage-context-3"
API_URL        = "https://api.voyageai.com/v1/contextualizedembeddings"
DIMS           = ARGS.dims
BATCH          = ARGS.batch
CONCURRENCY    = ARGS.concurrency
MAX_RESP_CHARS = 6000
# checkpoint every N "super-batches" (1 super-batch = CONCURRENCY × BATCH docs)
CKPT_EVERY     = 3

os.makedirs(INDEX_DIR, exist_ok=True)

ESCO_INDEX = os.path.join(INDEX_DIR, f"esco_occupations_contextual_{DIMS}.index")
ESCO_META  = os.path.join(INDEX_DIR, f"esco_occupations_contextual_{DIMS}_metadata.pkl")
JOB_INDEX  = os.path.join(INDEX_DIR, f"job_title_resp_contextual_{DIMS}.index")
JOB_META   = os.path.join(INDEX_DIR, f"job_title_resp_contextual_{DIMS}_metadata.pkl")

API_KEY = os.getenv("VOYAGE_API_KEY")
if not API_KEY:
    raise RuntimeError("VOYAGE_API_KEY not set")

# ── helpers ───────────────────────────────────────────────────────────────────
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


def save_index(index: faiss.Index, meta: dict, idx_path: str, meta_path: str) -> None:
    faiss.write_index(index, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  saved {index.ntotal:,} vectors → {idx_path}")


# ── async API call ─────────────────────────────────────────────────────────────
async def embed_one_batch(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
    batch:     list[list[str]],
    retries:   int = 4,
) -> tuple[int, list[list[float]]]:
    """
    POST one batch to the contextualizedembeddings endpoint.
    Returns (batch_idx, list_of_label_embeddings) to preserve order.
    """
    payload = {
        "inputs":           batch,
        "model":            MODEL,
        "input_type":       "document",
        "output_dimension": DIMS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(API_URL, headers=headers, json=payload) as resp:
                    if resp.status == 429:          # rate-limited
                        wait = 5 * (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    # REST API: data["data"] = list of docs,
                    # each doc["data"] = list of chunk embeddings
                    label_embs = [doc["data"][0]["embedding"] for doc in data["data"]]
                    return batch_idx, label_embs
            except aiohttp.ClientError as e:
                if attempt < retries - 1:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    raise RuntimeError(f"Batch {batch_idx} failed after {retries} retries: {e}")

    raise RuntimeError(f"Batch {batch_idx} exhausted retries")


async def embed_all_async(
    inputs:      list[list[str]],
    start_batch: int = 0,
    index:       faiss.Index | None = None,
    meta_fn=None,
    ckpt_index_path: str | None = None,
    ckpt_meta_path:  str | None = None,
    desc: str = "batches",
) -> faiss.Index:
    """
    Embed all inputs using CONCURRENCY concurrent requests.
    Processes in 'super-batches' of CONCURRENCY×BATCH docs to keep RAM low.
    """
    if index is None:
        index = faiss.IndexFlatIP(DIMS)

    total_batches   = (len(inputs) + BATCH - 1) // BATCH
    super_batch_size = CONCURRENCY   # batches per gather call

    semaphore = asyncio.Semaphore(CONCURRENCY)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=connector) as session:

        with tqdm(total=total_batches, initial=start_batch,
                  desc=f"  {desc}", unit="batch") as bar:

            sb = 0   # super-batch counter (for checkpointing)
            b  = start_batch

            while b < total_batches:
                # slice of up to super_batch_size batches
                sb_end   = min(b + super_batch_size, total_batches)
                tasks    = []
                for bi in range(b, sb_end):
                    lo, hi  = bi * BATCH, min((bi + 1) * BATCH, len(inputs))
                    tasks.append(
                        embed_one_batch(session, semaphore, bi, inputs[lo:hi])
                    )

                results = await asyncio.gather(*tasks)
                # results are (batch_idx, embs) — sort by batch_idx to preserve order
                results.sort(key=lambda x: x[0])

                for _, embs in results:
                    arr = np.array(embs, dtype=np.float32)
                    faiss.normalize_L2(arr)
                    index.add(arr)

                bar.update(sb_end - b)
                b  = sb_end
                sb += 1

                # checkpoint
                if ckpt_index_path and sb % CKPT_EVERY == 0:
                    faiss.write_index(index, ckpt_index_path)
                    if meta_fn and ckpt_meta_path:
                        with open(ckpt_meta_path, "wb") as f:
                            pickle.dump(meta_fn(index.ntotal), f)

    return index


# ── INDEX 1: ESCO occupations ─────────────────────────────────────────────────
async def build_esco_index() -> None:
    print(f"\n[1/2] ESCO occupations — voyage-context-3 {DIMS}d async …")

    df = pd.read_csv(ESCO_CSV)
    df["description"] = df["description"].fillna("")
    df = df[df["preferredLabel"].notna() & (df["preferredLabel"] != "")].reset_index(drop=True)
    print(f"  {len(df):,} occupations")

    labels = df["preferredLabel"].tolist()
    descs  = df["description"].tolist()
    uris   = df["conceptUri"].tolist()
    codes  = df["code"].tolist()

    inputs = [
        [lbl, desc] if desc and len(desc) > 10 else [lbl]
        for lbl, desc in zip(labels, descs)
    ]

    index       = faiss.IndexFlatIP(DIMS)
    start_batch = 0
    if os.path.exists(ESCO_INDEX):
        index       = faiss.read_index(ESCO_INDEX)
        start_batch = index.ntotal // BATCH
        print(f"  resuming from batch {start_batch} ({index.ntotal:,} done)")

    index = await embed_all_async(
        inputs, start_batch=start_batch, index=index, desc="ESCO",
    )

    save_index(index, {
        "labels": labels, "uris": uris, "codes": codes,
        "count": len(labels), "model": MODEL, "dimensions": DIMS,
        "type": "esco_occupations_contextual",
    }, ESCO_INDEX, ESCO_META)


# ── INDEX 2: job ads ───────────────────────────────────────────────────────────
async def build_job_index() -> None:
    print(f"\n[2/2] Job title+resp — voyage-context-3 {DIMS}d async …")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title_clean, responsibilities "
        "FROM job_ads WHERE title_clean IS NOT NULL AND title_clean != '' "
        "ORDER BY id"
    ).fetchall()
    conn.close()
    print(f"  {len(rows):,} rows")

    all_ids    = [r[0] for r in rows]
    all_titles = [r[1] for r in rows]
    all_resp   = [clean_resp(r[2] or "")[:MAX_RESP_CHARS] for r in rows]

    inputs = [
        [t, r] if r else [t]
        for t, r in zip(all_titles, all_resp)
    ]

    index       = faiss.IndexFlatIP(DIMS)
    start_batch = 0
    if os.path.exists(JOB_INDEX):
        index       = faiss.read_index(JOB_INDEX)
        start_batch = index.ntotal // BATCH
        print(f"  resuming from batch {start_batch} ({index.ntotal:,} done)")

    def meta_fn(n_done: int) -> dict:
        return {
            "ids": all_ids[:n_done], "labels": all_titles[:n_done],
            "count": n_done, "model": MODEL, "dimensions": DIMS,
            "type": "job_title_resp_contextual",
        }

    index = await embed_all_async(
        inputs,
        start_batch=start_batch,
        index=index,
        meta_fn=meta_fn,
        ckpt_index_path=JOB_INDEX,
        ckpt_meta_path=JOB_META,
        desc="Jobs",
    )

    save_index(index, {
        "ids": all_ids, "labels": all_titles,
        "count": len(all_ids), "model": MODEL, "dimensions": DIMS,
        "type": "job_title_resp_contextual",
    }, JOB_INDEX, JOB_META)


# ── main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    t0 = time.time()
    print(f"voyage-context-3 | dims={DIMS} | concurrency={CONCURRENCY} | batch={BATCH}")
    print(f"effective throughput: ~{CONCURRENCY * BATCH} docs/request-cycle")

    if not ARGS.jobs_only:
        await build_esco_index()
    if not ARGS.esco_only:
        await build_job_index()

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    asyncio.run(main())
