#!/usr/bin/env python3
"""
Verify skills_kalm.index and skills_kalm_metadata.pkl downloaded from HF Space.
Checks: FAISS structure, metadata consistency, alignment labels↔index, sample search.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INDEX_PATH = BASE / "faiss_indexes" / "skills_kalm.index"
META_PATH = BASE / "faiss_indexes" / "skills_kalm_metadata.pkl"
SKILLS_CSV = BASE / "ESCO dataset - v1.2.1 - classification - pl - csv" / "skills_pl.csv"

EXPECTED_DIMS = 3840
EXPECTED_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"


def main():
    print("=" * 60)
    print("Weryfikacja skills_kalm.index z HF Space")
    print("=" * 60)

    # 1. Pliki istnieją
    if not INDEX_PATH.exists():
        print(f"❌ Brak pliku: {INDEX_PATH}")
        return 1
    if not META_PATH.exists():
        print(f"❌ Brak pliku: {META_PATH}")
        return 1
    print(f"✓ Pliki: {INDEX_PATH.name}, {META_PATH.name}")

    # 2. Wczytaj FAISS
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        print(f"❌ Błąd wczytywania FAISS: {e}")
        return 1
    ntotal = index.ntotal
    d = index.d
    print(f"✓ FAISS: ntotal={ntotal:,}, d={d}")

    if d != EXPECTED_DIMS:
        print(f"❌ Oczekiwano dimensions={EXPECTED_DIMS}, jest {d}")
    else:
        print(f"✓ Wymiary zgodne z KaLM ({EXPECTED_DIMS})")

    # 3. Wczytaj metadane
    try:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        print(f"❌ Błąd wczytywania metadanych: {e}")
        return 1

    required_keys = {"labels", "uris", "count", "model", "dimensions"}
    missing = required_keys - set(meta.keys())
    if missing:
        print(f"❌ Brakujące klucze w meta: {missing}")
    else:
        print(f"✓ Meta: {list(meta.keys())}")

    labels = meta["labels"]
    uris = meta["uris"]
    count = meta["count"]
    model = meta["model"]
    dims = meta["dimensions"]

    print(f"  model: {model}")
    print(f"  dimensions: {dims}")
    print(f"  count: {count:,}")

    # 4. Zgodność count ↔ ntotal ↔ len(labels)
    if count != ntotal:
        print(f"❌ Niespójność: meta.count={count} vs index.ntotal={ntotal}")
    elif len(labels) != ntotal:
        print(f"❌ Niespójność: len(labels)={len(labels)} vs ntotal={ntotal}")
    else:
        print(f"✓ Liczby zgodne: count={count:,}, labels={len(labels):,}")

    # 5. Porównanie z skills_pl.csv
    if SKILLS_CSV.exists():
        df = pd.read_csv(SKILLS_CSV)
        csv_rows = len(df)
        if csv_rows != ntotal:
            print(f"⚠ CSV ma {csv_rows:,} wierszy, indeks {ntotal:,} (może być OK jeśli CSV się zmienił)")
        else:
            print(f"✓ Zgodność z skills_pl.csv: {csv_rows:,} wierszy")

    # 6. Próbne wyszukiwanie (query = wektor z indeksu, powinien znaleźć siebie na 1. miejscu)
    vec = index.reconstruct(0)
    vec = np.array([vec], dtype=np.float32)
    faiss.normalize_L2(vec)
    D, I = index.search(vec, 3)
    top_idx = int(I[0][0])
    top_score = float(D[0][0])
    top_label = labels[top_idx] if top_idx < len(labels) else "?"
    print(f"\n✓ Test search (reconstruct[0] → search):")
    print(f"  Top1: idx={top_idx}, score={top_score:.4f}, label=\"{top_label[:50]}...\"" if len(str(top_label)) > 50 else f"  Top1: idx={top_idx}, score={top_score:.4f}, label=\"{top_label}\"")
    if top_idx != 0:
        print(f"  ⚠ Oczekiwano idx=0 (ten sam wektor), dostałem {top_idx}")
    else:
        print(f"  ✓ Wynik spójny (ten sam wektor na 1. miejscu)")

    # 7. Losowa próbka etykiet
    print(f"\n✓ Przykładowe etykiety (indeks 0, 100, 1000):")
    for i in [0, 100, min(1000, len(labels) - 1)]:
        if i < len(labels):
            lab = labels[i][:60] + "…" if len(labels[i]) > 60 else labels[i]
            print(f"  [{i}] {lab}")

    print("\n" + "=" * 60)
    print("Weryfikacja zakończona.")
    return 0


if __name__ == "__main__":
    exit(main())
