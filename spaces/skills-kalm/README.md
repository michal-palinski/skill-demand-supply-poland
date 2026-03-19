---
title: ESCO Skills Search
emoji: 🔍
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
  - retrieval
---

# ESCO Skills Search

Wyszukiwanie umiejętności ESCO przez embeddings KaLM + FAISS.

## Uruchomienie na Space

1. **Secrets:** W Space → Settings → Variables and secrets dodaj `HF_TOKEN` (token Hugging Face).
2. **FAISS index:** Uruchom lokalnie skrypt z głównego repo:
   ```bash
   python embed_skills_kalm_faiss.py
   ```
   Skopiuj do tego repo:
   - `faiss_indexes/skills_kalm.index`
   - `faiss_indexes/skills_kalm_metadata.pkl`
3. **GPU:** Space wymaga GPU (A10G) dla modelu KaLM 12B. Ustaw `a10g-small` w Settings → Hardware.
4. **skills_pl.csv:** Skrypt embedowania wymaga ESCO `skills_pl.csv`. Możesz go pobrać z ESCO lub skopiować z głównego projektu.

## Użycie

Wpisz kompetencję w języku polskim (np. „analiza danych sprzedażowych”) – Space zwróci najbardziej podobne umiejętności z klasyfikacji ESCO.
