#!/usr/bin/env python3
"""
Embed unique BUR competencies (from bur_competencies_2025) using KaLM.
Zapis do FAISS z metadanymi umożliwiającymi identyfikację (label → bur_ids).
"""

import json
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
BUR_PARQUET = BASE / "trainings" / "data" / "bur_competencies_2025.parquet"
INDEX_DIR = BASE / "faiss_indexes"
INDEX_PATH = INDEX_DIR / "bur_competencies_kalm.index"
META_PATH = INDEX_DIR / "bur_competencies_kalm_metadata.pkl"

MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
DIMS = 3840
# ──────────────────────────────────────────────────────────────────────────────


def load_unique_bur_competencies() -> tuple[list[str], dict[str, list[int]]]:
    """Returns (unique_labels, bur_ids_by_label)."""
    df = pd.read_parquet(BUR_PARQUET)
    if "competencies" not in df.columns:
        raise ValueError("Kolumna 'competencies' nie istnieje. Sprawdź parquet.")

    bur_ids_by_label: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        bur_id = int(row["id"])
        comps = row.get("competencies")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except json.JSONDecodeError:
                comps = []
        if not isinstance(comps, list):
            comps = []
        for c in comps:
            label = str(c).strip() if c else ""
            if not label:
                continue
            if label not in bur_ids_by_label:
                bur_ids_by_label[label] = []
            if bur_id not in bur_ids_by_label[label]:
                bur_ids_by_label[label].append(bur_id)

    labels = sorted(bur_ids_by_label.keys())
    return labels, bur_ids_by_label


def main() -> None:
    token = os.getenv("hf_token_write") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("hf_token_write lub HF_TOKEN w .env")

    os.environ["HF_TOKEN"] = token

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError("pip install sentence-transformers torch")

    if not BUR_PARQUET.exists():
        raise FileNotFoundError(f"Brak pliku: {BUR_PARQUET}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Ładowanie unikalnych kompetencji BUR…")
    labels, bur_ids_by_label = load_unique_bur_competencies()
    n = len(labels)
    print(f"  {n:,} unikalnych kompetencji")

    print(f"Ładowanie modelu {MODEL_ID}…")
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
    encode_fn = getattr(model, "encode_document", None) or model.encode

    BATCH_SIZE = 64
    print(f"Embedding {n:,} kompetencji (batch={BATCH_SIZE})…")
    embeddings_list: list[np.ndarray] = []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="batche", unit="batch"):
        batch_texts = [(labels[j] or " ").strip() for j in range(i, min(i + BATCH_SIZE, n))]
        emb = encode_fn(
            batch_texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
        for k in range(len(batch_texts)):
            embeddings_list.append(np.asarray(emb[k], dtype=np.float32))

    arr = np.stack(embeddings_list, axis=0)
    faiss.normalize_L2(arr)

    print(f"Budowanie indeksu FAISS ({arr.shape[0]:,} × {arr.shape[1]})…")
    index = faiss.IndexFlatIP(DIMS)
    index.add(arr)

    faiss.write_index(index, str(INDEX_PATH))
    meta = {
        "labels": labels,
        "bur_ids_by_label": bur_ids_by_label,
        "count": n,
        "model": MODEL_ID,
        "dimensions": DIMS,
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"  → {INDEX_PATH}")
    print(f"  → {META_PATH}")
    print("Gotowe.")


if __name__ == "__main__":
    main()
