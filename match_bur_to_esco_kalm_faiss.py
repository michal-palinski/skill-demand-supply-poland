#!/usr/bin/env python3
"""
Top-1 dopasowanie kompetencji BUR do ESCO skills (KaLM + FAISS IndexFlatIP).
Wymaga: faiss_indexes/skills_kalm.index + skills_kalm_metadata.pkl
         faiss_indexes/bur_competencies_kalm.index + bur_competencies_kalm_metadata.pkl
Oba indeksy muszą być z tego samego modelu (3840d, znormalizowane wektory).

Zapis: PyArrow (parquet) — bez pandas (unika konfliktu NumPy 2.x z numexpr/bottleneck).

Uruchomienie:
  python match_bur_to_esco_kalm_faiss.py
  python match_bur_to_esco_kalm_faiss.py --out trainings/data/bur_to_esco_kalm_top1.parquet
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    pa = None  # type: ignore
    pq = None  # type: ignore
    _PYARROW_ERR = e
else:
    _PYARROW_ERR = None

BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "faiss_indexes"
ESCO_INDEX = INDEX_DIR / "skills_kalm.index"
ESCO_META = INDEX_DIR / "skills_kalm_metadata.pkl"
BUR_INDEX = INDEX_DIR / "bur_competencies_kalm.index"
BUR_META = INDEX_DIR / "bur_competencies_kalm_metadata.pkl"
DEFAULT_OUT = BASE / "trainings" / "data" / "bur_to_esco_kalm_top1.parquet"

SEARCH_BATCH = 2048


def _write_parquet(
    out: Path,
    bur_comp: list[str],
    bur_ids_json: list[str],
    bur_n: list[int],
    esco_lab: list[str],
    esco_uri: list[str],
    sim: list[float],
) -> None:
    if pa is None or pq is None:
        raise ImportError(
            "Zainstaluj pyarrow: pip install pyarrow\n"
            f"(import pyarrow nie powiódł się: {_PYARROW_ERR})"
        )
    table = pa.table(
        {
            "bur_competency": bur_comp,
            "bur_bur_ids_json": bur_ids_json,
            "bur_n_trainings": bur_n,
            "esco_preferredLabel": esco_lab,
            "esco_conceptUri": esco_uri,
            "similarity": sim,
        }
    )
    pq.write_table(table, str(out), compression="zstd")


def _write_csv(
    out: Path,
    bur_comp: list[str],
    bur_ids_json: list[str],
    bur_n: list[int],
    esco_lab: list[str],
    esco_uri: list[str],
    sim: list[float],
) -> None:
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bur_competency",
                "bur_bur_ids_json",
                "bur_n_trainings",
                "esco_preferredLabel",
                "esco_conceptUri",
                "similarity",
            ]
        )
        for row in zip(bur_comp, bur_ids_json, bur_n, esco_lab, esco_uri, sim):
            w.writerow(row)


def _describe_similarity(scores: np.ndarray) -> None:
    x = np.asarray(scores, dtype=np.float64)
    print("count   ", len(x))
    print("mean    ", float(np.mean(x)))
    print("std     ", float(np.std(x)))
    print("min     ", float(np.min(x)))
    print("25%     ", float(np.percentile(x, 25)))
    print("50%     ", float(np.percentile(x, 50)))
    print("75%     ", float(np.percentile(x, 75)))
    print("max     ", float(np.max(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ścieżka wyjścia (.parquet)")
    ap.add_argument("--csv", type=Path, default=None, help="Opcjonalnie zapisz też CSV")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Tylko pierwsze N wierszy (test)",
    )
    args = ap.parse_args()

    for p in (ESCO_INDEX, ESCO_META, BUR_INDEX, BUR_META):
        if not p.exists():
            raise FileNotFoundError(f"Brak pliku: {p}")

    print("Ładowanie indeksu ESCO…")
    esco_idx = faiss.read_index(str(ESCO_INDEX))
    with open(ESCO_META, "rb") as f:
        esco_meta = pickle.load(f)
    esco_labels = esco_meta["labels"]
    esco_uris = esco_meta["uris"]
    assert len(esco_labels) == esco_idx.ntotal == len(esco_uris), (
        len(esco_labels),
        esco_idx.ntotal,
        len(esco_uris),
    )

    print("Ładowanie indeksu BUR…")
    bur_idx = faiss.read_index(str(BUR_INDEX))
    with open(BUR_META, "rb") as f:
        bur_meta = pickle.load(f)
    bur_labels = bur_meta["labels"]
    bur_ids_by_label: dict = bur_meta.get("bur_ids_by_label", {})
    n_bur = bur_idx.ntotal
    if len(bur_labels) != n_bur:
        raise ValueError(
            f"Niespójność: len(bur_labels)={len(bur_labels)} vs bur_idx.ntotal={n_bur}"
        )
    if args.limit is not None:
        n_bur = min(n_bur, args.limit)
        bur_labels = bur_labels[:n_bur]
    if esco_idx.d != bur_idx.d:
        raise ValueError(f"Różne wymiary: ESCO d={esco_idx.d}, BUR d={bur_idx.d}")

    print(f"  ESCO: {esco_idx.ntotal:,} skills, d={esco_idx.d}")
    print(f"  BUR:  {n_bur:,} kompetencji")

    top_esco_idx = np.empty(n_bur, dtype=np.int64)
    scores = np.empty(n_bur, dtype=np.float32)

    with tqdm(total=n_bur, desc="BUR→ESCO top-1 (FAISS)", unit="vec") as pbar:
        for start in range(0, n_bur, SEARCH_BATCH):
            end = min(start + SEARCH_BATCH, n_bur)
            vecs = np.stack([bur_idx.reconstruct(int(i)) for i in range(start, end)]).astype(
                np.float32
            )
            faiss.normalize_L2(vecs)
            D, I = esco_idx.search(vecs, 1)
            top_esco_idx[start:end] = I[:, 0]
            scores[start:end] = D[:, 0]
            pbar.update(end - start)

    bur_comp: list[str] = []
    bur_ids_json: list[str] = []
    bur_n: list[int] = []
    esco_lab: list[str] = []
    esco_uri: list[str] = []
    sim: list[float] = []

    for i in tqdm(range(n_bur), desc="Budowa tabeli", unit="row"):
        bl = bur_labels[i]
        j = int(top_esco_idx[i])
        ids = bur_ids_by_label.get(bl, [])
        bur_comp.append(bl)
        bur_ids_json.append(json.dumps(ids, ensure_ascii=False))
        bur_n.append(len(ids))
        esco_lab.append(esco_labels[j])
        esco_uri.append(esco_uris[j])
        sim.append(float(scores[i]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suf = args.out.suffix.lower()
    if suf == ".parquet":
        _write_parquet(args.out, bur_comp, bur_ids_json, bur_n, esco_lab, esco_uri, sim)
    elif suf == ".csv":
        _write_csv(args.out, bur_comp, bur_ids_json, bur_n, esco_lab, esco_uri, sim)
    else:
        raise ValueError(f"Nieobsługiwane rozszerzenie {suf!r}; użyj .parquet lub .csv")

    print(f"Zapisano: {args.out} ({n_bur:,} wierszy)")

    if args.csv:
        _write_csv(args.csv, bur_comp, bur_ids_json, bur_n, esco_lab, esco_uri, sim)
        print(f"Zapisano CSV: {args.csv}")

    print("\nPodsumowanie similarity (top-1):")
    _describe_similarity(scores)


if __name__ == "__main__":
    main()
