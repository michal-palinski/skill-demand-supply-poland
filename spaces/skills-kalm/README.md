---
title: ESCO Skills Embed
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
suggested_hardware: a100-large
models:
  - tencent/KaLM-Embedding-Gemma3-12B-2511
tags:
  - esco
  - embeddings
  - faiss
---

# ESCO Skills Embed

Buduje embeddings ESCO skills (KaLM 12B) i zapisuje indeks FAISS do `faiss_indexes/`. Pobierz pliki lokalnie.

## Konfiguracja

1. **Secrets:** Ustaw `HF_TOKEN` w Space → Settings → Variables and secrets.
2. **Hardware:** A100 (KaLM 12B potrzebuje ~24GB VRAM).

## Użycie

1. Kliknij **Uruchom embedding** – Space wykona embedding ~36k skills (ok. 1–2 h na A100).
2. Po zakończeniu pobierz `skills_kalm.index` i `skills_kalm_metadata.pkl`.
3. Skopiuj do lokalnego `faiss_indexes/`.
