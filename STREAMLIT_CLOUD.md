# Wdrożenie na Streamlit Community Cloud

Nie da się wdrożyć „za Ciebie” bez dostępu do Twojego konta GitHub + Streamlit. Po jednorazowej konfiguracji każdy **push** na branch aktualizuje aplikację.

## Wymagania w repozytorium

1. Folder **`deploy/`** z plikami z `prepare_deploy.py` (duże `*.db` przez **Git LFS** — masz `.gitattributes`).
2. W repozytorium muszą być **`app_deploy.py`**, **`streamlit_app.py`**, **`requirements-streamlit.txt`**.

## Kroki w [share.streamlit.io](https://share.streamlit.io)

1. **New app** → wybierz repo i branch (np. `main`).
2. **Main file path:** `streamlit_app.py`
3. **App URL** (opcjonalnie) — własna nazwa.
4. **Advanced settings** → **Python version** → `3.11` (lub 3.10).
5. **Requirements file:** `requirements-streamlit.txt`  
   (nie używaj głównego `requirements.txt` — zawiera FAISS / `sentence-transformers`, których Cloud app nie potrzebuje.)

6. **Deploy.**

## Limit rozmiaru

`deploy/app_data.db` + `req_resp_slim.db` itd. muszą zmieścić się w limitach Streamlit (repo + LFS). Jeśli build się wykrzacza, rozważ mniejszy eksport lub hosting dużych plików poza repo.

## Lokalnie przed pushem

```bash
python export_app_data.py    # jeśli masz pełne źródła
python prepare_deploy.py
git add deploy/ app_deploy.py streamlit_app.py requirements-streamlit.txt
git commit -m "Deploy data + Streamlit entry"
git push
```

Powiązana aplikacja na Streamlit odświeży się sama po pushu.
