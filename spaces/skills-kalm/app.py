"""
ESCO Skills Search – KaLM embeddings + FAISS.
Uruchom na Hugging Face Space z GPU (A10G).
"""

import os
from pathlib import Path

import faiss
import streamlit as st

BASE = Path(__file__).resolve().parent
INDEX_PATH = BASE / "faiss_indexes" / "skills_kalm.index"
META_PATH = BASE / "faiss_indexes" / "skills_kalm_metadata.pkl"
MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
TOP_K = 10


@st.cache_resource
def load_model():
    token = os.environ.get("HF_TOKEN")
    if not token:
        st.error("Ustaw HF_TOKEN w Space Secrets (Settings → Variables and secrets)")
        return None
    from sentence_transformers import SentenceTransformer
    import torch
    model_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    return SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        token=token,
        model_kwargs=model_kwargs,
    )


@st.cache_resource
def load_index():
    """Load FAISS from repo or HF Hub (INDEX_HF_REPO env)."""
    import pickle
    index_path, meta_path = INDEX_PATH, META_PATH
    hf_repo = os.environ.get("INDEX_HF_REPO")
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            index_path = hf_hub_download(repo_id=hf_repo, filename="skills_kalm.index", repo_type="dataset")
            meta_path = hf_hub_download(repo_id=hf_repo, filename="skills_kalm_metadata.pkl", repo_type="dataset")
        except Exception as e:
            st.error(f"Nie udało się pobrać indeksu z {hf_repo}: {e}")
            return None, None
    elif not index_path.exists() or not meta_path.exists():
        st.error(
            "Brak plików FAISS. Opcje: 1) Dodaj faiss_indexes/ do repo "
            "2) Ustaw INDEX_HF_REPO (repo z datasetem) w Secrets."
        )
        return None, None
    else:
        index_path, meta_path = str(index_path), str(meta_path)
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query: str, index, meta, model, k: int = TOP_K):
    encode_fn = getattr(model, "encode_query", None) or model.encode
    qv = encode_fn(
        [query.strip() or " "],
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False,
    )
    qv = qv[0].astype("float32").reshape(1, -1)
    faiss.normalize_L2(qv)
    scores, idxs = index.search(qv, min(k, index.ntotal))
    labels = meta.get("labels", [])
    uris = meta.get("uris", [])
    return [
        {"label": labels[i], "uri": uris[i] if i < len(uris) else "", "score": float(scores[0][j])}
        for j, i in enumerate(idxs[0])
        if i >= 0 and i < len(labels)
    ]


def main():
    st.set_page_config(page_title="ESCO Skills Search", page_icon="🔍", layout="centered")
    st.title("🔍 ESCO Skills Search")
    st.caption("Wyszukiwanie umiejętności ESCO przez KaLM embeddings (3840d) + FAISS")

    index, meta = load_index()
    if index is None:
        return

    model = load_model()
    if model is None:
        return

    st.write(f"**Index:** {index.ntotal:,} skills | **Model:** {MODEL_ID}")

    q = st.text_input("Wpisz kompetencję lub frazę (np. analiza danych sprzedażowych)", placeholder="analiza danych sprzedażowych")
    k = st.slider("Liczba wyników", 1, 50, TOP_K)

    if not q.strip():
        st.info("Wprowadź zapytanie, aby wyszukać podobne umiejętności ESCO.")
        return

    if st.button("Szukaj"):
        with st.spinner("Wyszukiwanie…"):
            results = search(q, index, meta, model, k)
        if not results:
            st.warning("Brak wyników.")
        else:
            for j, r in enumerate(results[:k], 1):
                with st.expander(f"**{j}. {r['label']}** (sim={r['score']:.3f})", expanded=(j <= 3)):
                    st.write(r["label"])
                    if r.get("uri"):
                        st.caption(r["uri"])


if __name__ == "__main__":
    main()
