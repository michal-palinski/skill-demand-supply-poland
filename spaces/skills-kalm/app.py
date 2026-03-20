"""
Space: Embed ESCO skills + BUR competencies (KaLM) → FAISS.
Budowanie indeksów i udostępnienie do pobrania.
"""

import json
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
SKILLS_CSV = BASE / "skills_pl.csv"
BUR_PARQUET = BASE / "bur_competencies_2025.parquet"
INDEX_DIR = BASE / "faiss_indexes"

# ESCO
ESCO_INDEX = INDEX_DIR / "skills_kalm.index"
ESCO_META = INDEX_DIR / "skills_kalm_metadata.pkl"

# BUR
BUR_INDEX = INDEX_DIR / "bur_competencies_kalm.index"
BUR_META = INDEX_DIR / "bur_competencies_kalm_metadata.pkl"

MODEL_ID = "tencent/KaLM-Embedding-Gemma3-12B-2511"
DIMS = 3840
BATCH_SIZE = 64


def _get_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGING_FACE_HUB_TOKEN")
    except Exception:
        return None


@st.cache_resource
def _load_model(token):
    from sentence_transformers import SentenceTransformer
    import torch
    model_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True, token=token, model_kwargs=model_kwargs)
    model.max_seq_length = 512
    return model


def _run_esco_embedding():
    token = _get_token()
    if not token:
        st.error("Ustaw HF_TOKEN w Space Secrets.")
        return False
    if not SKILLS_CSV.exists():
        st.error("Brak skills_pl.csv")
        return False

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SKILLS_CSV)
    labels = df["preferredLabel"].fillna("").astype(str).tolist()
    uris = df["conceptUri"].fillna("").astype(str).tolist()
    n = len(labels)

    st.info("Ładowanie modelu KaLM…")
    model = _load_model(token)
    encode_fn = getattr(model, "encode_document", None) or model.encode

    st.info(f"Embedding {n:,} ESCO skills…")
    embeddings_list = []
    pbar = st.progress(0.0, text="")
    for i in range(0, n, BATCH_SIZE):
        batch = [(labels[j] or " ").strip() for j in range(i, min(i + BATCH_SIZE, n))]
        emb = encode_fn(batch, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False)
        for k in range(len(batch)):
            embeddings_list.append(np.asarray(emb[k], dtype=np.float32))
        pbar.progress(min(1.0, (i + len(batch)) / n), text=f"{min(i + BATCH_SIZE, n):,} / {n:,}")
    pbar.progress(1.0, text=f"Gotowe: {n:,}")

    arr = np.stack(embeddings_list, axis=0)
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(DIMS)
    idx.add(arr)
    faiss.write_index(idx, str(ESCO_INDEX))
    meta = {"labels": labels, "uris": uris, "count": n, "model": MODEL_ID, "dimensions": DIMS}
    with open(ESCO_META, "wb") as f:
        pickle.dump(meta, f)
    st.success("Zapisano skills_kalm.index")
    return True


@st.cache_data
def _load_bur_unique():
    df = pd.read_parquet(BUR_PARQUET)
    bur_ids_by_label = {}
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


def _run_bur_embedding():
    token = _get_token()
    if not token:
        st.error("Ustaw HF_TOKEN w Space Secrets.")
        return False
    if not BUR_PARQUET.exists():
        st.error("Brak bur_competencies_2025.parquet")
        return False

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    st.info("Ładowanie unikalnych kompetencji BUR…")
    labels, bur_ids_by_label = _load_bur_unique()
    n = len(labels)
    st.write(f"**{n:,}** unikalnych kompetencji")

    st.info("Ładowanie modelu KaLM…")
    model = _load_model(token)
    encode_fn = getattr(model, "encode_document", None) or model.encode

    st.info(f"Embedding {n:,} kompetencji (batch={BATCH_SIZE})…")
    embeddings_list = []
    status_placeholder = st.empty()
    pbar = st.progress(0.0, text=f"0 / {n:,} (0%)")
    for i in range(0, n, BATCH_SIZE):
        batch = [(labels[j] or " ").strip() for j in range(i, min(i + BATCH_SIZE, n))]
        emb = encode_fn(batch, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False)
        for k in range(len(batch)):
            embeddings_list.append(np.asarray(emb[k], dtype=np.float32))
        done = min(i + len(batch), n)
        pct = 100 * done / n
        status_placeholder.caption(f"📊 {done:,} / {n:,} ({pct:.1f}%)")
        pbar.progress(done / n, text=f"{done:,} / {n:,} ({pct:.1f}%)")
        time.sleep(0.08)  # pozwala Streamlit wysłać aktualizację na HF (wolne WebSocket)
    status_placeholder.empty()
    pbar.progress(1.0, text=f"Gotowe: {n:,} (100%)")

    arr = np.stack(embeddings_list, axis=0)
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(DIMS)
    idx.add(arr)
    faiss.write_index(idx, str(BUR_INDEX))
    meta = {"labels": labels, "bur_ids_by_label": bur_ids_by_label, "count": n, "model": MODEL_ID, "dimensions": DIMS}
    with open(BUR_META, "wb") as f:
        pickle.dump(meta, f)
    st.success("Zapisano bur_competencies_kalm.index")
    return True


def main():
    st.set_page_config(page_title="KaLM Embed", page_icon="🔧", layout="centered")
    st.title("🔧 KaLM embeddings → FAISS")
    st.caption("ESCO skills + BUR competencies. Buduj indeksy i pobierz pliki.")

    token = _get_token()
    if token:
        st.success("Token HF: OK")
    else:
        st.error("Brak HF_TOKEN. Ustaw w Settings → Repository secrets.")

    tab1, tab2 = st.tabs(["ESCO skills", "BUR competencies"])

    with tab1:
        st.subheader("ESCO skills")
        if ESCO_INDEX.exists():
            st.success("Indeks istnieje.")
            st.download_button("⬇️ skills_kalm.index", data=ESCO_INDEX.read_bytes(), file_name="skills_kalm.index", mime="application/octet-stream", key="esco_idx")
            st.download_button("⬇️ skills_kalm_metadata.pkl", data=ESCO_META.read_bytes(), file_name="skills_kalm_metadata.pkl", mime="application/octet-stream", key="esco_meta")
        else:
            if st.button("Uruchom embedding ESCO", type="primary", key="esco_btn"):
                with st.spinner("Trwa embedding…"):
                    _run_esco_embedding()
                st.rerun()

    with tab2:
        st.subheader("BUR competencies (unikalne)")
        if BUR_PARQUET.exists():
            try:
                labels, _ = _load_bur_unique()
                st.markdown(f"Parquet: **{len(labels):,}** unikalnych kompetencji")
            except Exception as e:
                st.warning(f"Nie udało się wczytać parquet: {e}")
        if BUR_INDEX.exists():
            st.success("Indeks istnieje.")
            st.download_button("⬇️ bur_competencies_kalm.index", data=BUR_INDEX.read_bytes(), file_name="bur_competencies_kalm.index", mime="application/octet-stream", key="bur_idx")
            st.download_button("⬇️ bur_competencies_kalm_metadata.pkl", data=BUR_META.read_bytes(), file_name="bur_competencies_kalm_metadata.pkl", mime="application/octet-stream", key="bur_meta")
        else:
            if st.button("Uruchom embedding BUR", type="primary", key="bur_btn"):
                _run_bur_embedding()
                st.rerun()


if __name__ == "__main__":
    main()
