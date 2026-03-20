#!/usr/bin/env python3
"""
Embed ESCO skills using KaLM (tencent/KaLM-Embedding-Gemma3-12B-2511).
Uses HF token from hf_token_write in .env.
Embeds each skill individually and saves to FAISS.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
SKILLS_CSV = BASE / "ESCO dataset - v1.2.1 - classification - pl - csv" / "skills_pl.csv"
INDEX_DIR = BASE / "faiss_indexes"
INDEX_PATH = INDEX_DIR / "skills_kalm.index"
META_PATH = INDEX_DIR / "skills_kalm_metadata.pkl"

MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
DIMS = 3840  # KaLM embedding dimension
# ──────────────────────────────────────────────────────────────────────────────


def load_skills() -> tuple[list[str], list[str], list[str]]:
    """Returns (labels, uris, texts). texts = preferredLabel for embedding."""
    df = pd.read_csv(SKILLS_CSV)
    labels = df["preferredLabel"].fillna("").astype(str).tolist()
    uris = df["conceptUri"].fillna("").astype(str).tolist()
    return labels, uris, labels


def main() -> None:
    token = os.getenv("hf_token_write") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("hf_token_write or HF_TOKEN not set in .env")

    os.environ["HF_TOKEN"] = token

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError(
            "pip install sentence-transformers torch"
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ESCO skills…")
    labels, uris, texts = load_skills()
    n = len(texts)
    print(f"  {n:,} skills")

    print(f"Loading model {MODEL_ID}…")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        token=token,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = 512

    # encode_document for documents (empty prompt); encode_query for queries
    encode_fn = getattr(model, "encode_document", None) or model.encode

    print("Embedding each skill individually…")
    embeddings_list: list[np.ndarray] = []
    for text in tqdm(texts, desc="skills", unit="skill"):
        txt = text.strip() or " "
        emb = encode_fn(
            [txt],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        vec = np.asarray(emb[0], dtype=np.float32)
        embeddings_list.append(vec)

    arr = np.stack(embeddings_list, axis=0)
    faiss.normalize_L2(arr)

    print(f"Building FAISS index ({arr.shape[0]:,} × {arr.shape[1]})…")
    index = faiss.IndexFlatIP(DIMS)
    index.add(arr)

    faiss.write_index(index, str(INDEX_PATH))
    meta = {
        "labels": labels,
        "uris": uris,
        "count": n,
        "model": MODEL_ID,
        "dimensions": DIMS,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"  → {INDEX_PATH}")
    print(f"  → {META_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
