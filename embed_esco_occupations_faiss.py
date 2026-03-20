#!/usr/bin/env python3
"""
Create Voyage AI voyage-4 embeddings (2048 dims) for ESCO occupations_pl.csv.

Two FAISS indexes are produced:
  faiss_indexes/esco_occupations_label.index          — preferredLabel only
  faiss_indexes/esco_occupations_label_desc.index     — preferredLabel + description
"""

import os
import pickle
import time

import faiss
import numpy as np
import pandas as pd
import voyageai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "ESCO dataset - v1.2.1 - classification - pl - csv/occupations_pl.csv"
INDEX_DIR = "faiss_indexes"
MODEL = "voyage-4"
DIMENSIONS = 2048
BATCH_SIZE = 128   # voyage-4 max input per request

LABEL_INDEX_FILE    = os.path.join(INDEX_DIR, "esco_occupations_label.index")
LABEL_META_FILE     = os.path.join(INDEX_DIR, "esco_occupations_label_metadata.pkl")
LABELDESC_INDEX_FILE = os.path.join(INDEX_DIR, "esco_occupations_label_desc.index")
LABELDESC_META_FILE  = os.path.join(INDEX_DIR, "esco_occupations_label_desc_metadata.pkl")
# ──────────────────────────────────────────────────────────────────────────────


def embed_texts(vo: voyageai.Client, texts: list[str], input_type: str = "document") -> np.ndarray:
    """Embed all texts in batches, return float32 array (n, DIMENSIONS)."""
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  batches", unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        result = vo.embed(
            texts=batch,
            model=MODEL,
            input_type=input_type,
            output_dimension=DIMENSIONS,
        )
        all_embeddings.extend(result.embeddings)
        # gentle rate-limit buffer
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.1)

    arr = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(arr)          # cosine similarity via inner-product
    return arr


def build_and_save_index(
    embeddings: np.ndarray,
    metadata: dict,
    index_path: str,
    meta_path: str,
) -> None:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)     # inner-product on normalized vecs = cosine
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  saved {index.ntotal:,} vectors → {index_path}")
    print(f"  metadata → {meta_path}")


def main() -> None:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set in environment / .env")

    vo = voyageai.Client(api_key=api_key)
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading CSV…")
    df = pd.read_csv(CSV_PATH)
    df["description"] = df["description"].fillna("")
    print(f"  {len(df):,} rows")

    labels = df["preferredLabel"].tolist()
    labels_with_desc = (df["preferredLabel"] + " " + df["description"]).str.strip().tolist()

    metadata_base = {
        "labels": labels,
        "uris": df["conceptUri"].tolist(),
        "codes": df["code"].tolist(),
        "count": len(df),
        "model": MODEL,
        "dimensions": DIMENSIONS,
    }

    # ── Index 1: preferredLabel ───────────────────────────────────────────────
    print(f"\n[1/2] Embedding preferredLabel ({len(labels):,} texts)…")
    emb_label = embed_texts(vo, labels)
    print(f"  shape: {emb_label.shape}")
    build_and_save_index(
        emb_label,
        {**metadata_base, "type": "label_only"},
        LABEL_INDEX_FILE,
        LABEL_META_FILE,
    )

    # ── Index 2: preferredLabel + description ─────────────────────────────────
    print(f"\n[2/2] Embedding preferredLabel + description ({len(labels_with_desc):,} texts)…")
    emb_labeldesc = embed_texts(vo, labels_with_desc)
    print(f"  shape: {emb_labeldesc.shape}")
    build_and_save_index(
        emb_labeldesc,
        {**metadata_base, "type": "label_and_description", "texts": labels_with_desc},
        LABELDESC_INDEX_FILE,
        LABELDESC_META_FILE,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
