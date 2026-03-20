"""
Embed Polish ESCO skills using voyage-context-3 contextualized chunk embeddings.

Each ESCO skill = document with [preferredLabel_pl, description_pl].
The label embedding captures semantic context from its description.

Outputs:
    faiss_index/esco_pl_contextual_1024.npy       — (N, 1024) float32 embeddings
    faiss_index/esco_pl_contextual_metadata.json   — labels + metadata

Requires:
    pip install voyageai python-dotenv tqdm pandas numpy
"""
import os
import json
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import voyageai

BASE = os.path.dirname(__file__)
ESCO_CSV = os.path.join(BASE, "pl", "ESCO dataset - v1.2.1 - classification - pl - csv", "skills_pl.csv")
FAISS_DIR = os.path.join(BASE, "faiss_index")
os.makedirs(FAISS_DIR, exist_ok=True)

BATCH_SIZE = 100   # documents per API request
DIM = 1024


def init_voyage():
    load_dotenv()
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key or api_key == "your-voyage-key-here":
        raise ValueError("VOYAGE_API_KEY not set in .env")
    return voyageai.Client(api_key=api_key)


def load_esco_pl():
    """Load Polish ESCO skills with descriptions."""
    print("Loading Polish ESCO skills from CSV...")
    df = pd.read_csv(ESCO_CSV)
    df["description"] = df["description"].fillna("")
    df["preferredLabel"] = df["preferredLabel"].fillna("")

    df = df[df["preferredLabel"].str.len() > 0].reset_index(drop=True)

    labels = df["preferredLabel"].tolist()
    descriptions = df["description"].tolist()

    print(f"  Loaded {len(labels):,} ESCO skills (PL)")
    print(f"  With descriptions: {sum(1 for d in descriptions if d):,}")
    print(f"  Sample: '{labels[0]}' — {descriptions[0][:80]}…")

    return labels, descriptions, df


def embed_esco_contextual(vo, labels, descriptions):
    """Embed Polish ESCO skills using voyage-context-3.

    Each skill is a "document" with chunks: [label, description].
    We keep only the label embedding (chunk 0) — it carries context
    from the description.
    """
    cache_path = os.path.join(FAISS_DIR, "esco_pl_contextual_1024.npy")
    meta_path = os.path.join(FAISS_DIR, "esco_pl_contextual_metadata.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        embs = np.load(cache_path)
        if embs.shape[0] == len(labels) and embs.shape[1] == DIM:
            print(f"\n✅ Cached embeddings found — {embs.shape[0]:,} x {DIM}")
            return embs
        print("   Cache mismatch, re-embedding...")

    print(f"\n🔄 Embedding {len(labels):,} Polish ESCO skills with voyage-context-3 …")
    print(f"   Each skill = [label_pl, description_pl] document")
    print(f"   Batch size: {BATCH_SIZE}, output dim: {DIM}")

    # Build inputs
    all_inputs = []
    for label, desc in zip(labels, descriptions):
        if desc and len(desc) > 10:
            all_inputs.append([label, desc])
        else:
            all_inputs.append([label])

    all_label_embeddings = []

    with tqdm(total=len(all_inputs), desc="Embedding ESCO PL") as pbar:
        for i in range(0, len(all_inputs), BATCH_SIZE):
            batch = all_inputs[i : i + BATCH_SIZE]

            retries = 3
            while retries > 0:
                try:
                    result = vo.contextualized_embed(
                        inputs=batch,
                        model="voyage-context-3",
                        input_type="document",
                        output_dimension=DIM,
                    )
                    for r in result.results:
                        all_label_embeddings.append(r.embeddings[0])

                    pbar.update(len(batch))
                    break
                except Exception as e:
                    retries -= 1
                    if retries > 0:
                        print(f"\n   ⚠️ {e} — retrying in 3s…")
                        time.sleep(3)
                    else:
                        raise

    embeddings = np.array(all_label_embeddings, dtype=np.float32)

    # Cache
    np.save(cache_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": "pl",
                "model": "voyage-context-3",
                "dimensions": DIM,
                "total": len(labels),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "labels": labels,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n   💾 Saved: {cache_path}")
    print(f"   💾 Saved: {meta_path}")
    return embeddings


def main():
    print("=" * 60)
    print("Polish ESCO Contextual Embedding")
    print("=" * 60)

    vo = init_voyage()
    labels, descriptions, df = load_esco_pl()
    embeddings = embed_esco_contextual(vo, labels, descriptions)

    print(f"\n{'=' * 60}")
    print(f"✅ Done!  {embeddings.shape[0]:,} embeddings, dim={embeddings.shape[1]}")
    print(f"   Cache: faiss_index/esco_pl_contextual_1024.npy")
    print(f"   Meta:  faiss_index/esco_pl_contextual_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
