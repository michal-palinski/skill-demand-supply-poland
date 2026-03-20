#!/usr/bin/env python3
"""
Embed BUR competencies (~410k) z KaLM na wielu GPU (4×A100).
Użycie: CUDA_VISIBLE_DEVICES=0,1,2,3 python embed_bur_competencies_kalm_multigpu.py
Lub: N_GPUS=4 python embed_bur_competencies_kalm_multigpu.py

Optymalizacje:
- Data parallelism: podział na N chunków, każdy GPU obsługuje swój chunk
- Większy batch (128–256) na A100 80GB
- Multiprocessing z przypisaniem GPU
"""

from __future__ import annotations

import json
import multiprocessing as mp
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
TEMP_DIR = INDEX_DIR / "_bur_multigpu_temp"
INDEX_PATH = INDEX_DIR / "bur_competencies_kalm.index"
META_PATH = INDEX_DIR / "bur_competencies_kalm_metadata.pkl"

MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
DIMS = 3840

# Na A100 80GB: batch 128–256 jest zazwyczaj OK dla KaLM 12B (bf16)
BATCH_SIZE = int(os.getenv("BUR_BATCH_SIZE", "128"))
N_GPUS = int(os.getenv("N_GPUS", "4"))
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


def _worker_embed(
    gpu_id: int,
    chunk_labels: list[str],
    out_path: Path,
    token: str,
    batch_size: int,
) -> None:
    """Pojedynczy worker: ładuje model na GPU gpu_id, embeduje chunk, zapisuje .npy."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_TOKEN"] = token

    from sentence_transformers import SentenceTransformer
    import torch

    if not chunk_labels:
        np.save(out_path, np.zeros((0, DIMS), dtype=np.float32))
        return

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

    embeddings_list: list[np.ndarray] = []
    texts = [(t or " ").strip() for t in chunk_labels]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = encode_fn(
            batch,
            normalize_embeddings=True,
            batch_size=len(batch),
            show_progress_bar=False,
        )
        for k in range(len(batch)):
            embeddings_list.append(np.asarray(emb[k], dtype=np.float32))

    arr = np.stack(embeddings_list, axis=0)
    np.save(out_path, arr)


def main() -> None:
    token = os.getenv("hf_token_write") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("hf_token_write lub HF_TOKEN w .env")

    try:
        import torch
    except ImportError:
        raise ImportError("pip install torch")

    n_gpus = min(N_GPUS, torch.cuda.device_count())
    if n_gpus < 1:
        raise RuntimeError("Brak GPU. Uruchom na maszynie z CUDA.")

    if not BUR_PARQUET.exists():
        raise FileNotFoundError(f"Brak pliku: {BUR_PARQUET}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Ładowanie unikalnych kompetencji BUR…")
    labels, bur_ids_by_label = load_unique_bur_competencies()
    n = len(labels)
    print(f"  {n:,} unikalnych kompetencji")
    print(f"  GPU: {n_gpus} × A100, batch={BATCH_SIZE}")

    # Podział na chunki (każdy GPU dostaje ~równy slice)
    chunk_size = (n + n_gpus - 1) // n_gpus
    chunks: list[tuple[int, int, int]] = []
    for g in range(n_gpus):
        start = g * chunk_size
        end = min(start + chunk_size, n)
        if start < end:
            chunks.append((g, start, end))

    print(f"  Chunki: {[(c[1], c[2]) for c in chunks]} (start, end)")

    # Uruchom workerów równolegle
    procs: list[mp.Process] = []
    out_paths: list[Path] = []
    for gpu_id, start, end in chunks:
        chunk_labels = labels[start:end]
        out_p = TEMP_DIR / f"emb_gpu{gpu_id}.npy"
        out_paths.append(out_p)
        p = mp.Process(
            target=_worker_embed,
            args=(gpu_id, chunk_labels, out_p, token, BATCH_SIZE),
        )
        p.start()
        procs.append(p)

    for p in tqdm(procs, desc="Oczekiwanie na GPU", unit="gpu"):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker zakończył się z kodem {p.exitcode}")

    # Merge embeddings w kolejności chunków
    print("Scalanie embeddingów…")
    parts = [np.load(p) for p in out_paths]
    arr = np.concatenate(parts, axis=0).astype(np.float32)
    assert arr.shape[0] == n and arr.shape[1] == DIMS, (arr.shape, n, DIMS)

    # Usuń pliki tymczasowe
    for p in out_paths:
        p.unlink(missing_ok=True)
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

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
