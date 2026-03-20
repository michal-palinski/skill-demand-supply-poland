#!/usr/bin/env python3
"""
Entry point for Streamlit Community Cloud.

W ustawieniach aplikacji ustaw **Main file**: `streamlit_app.py`
(oraz **Requirements file**: `requirements-streamlit.txt`).

Uruchamia `app_deploy.py` (dane z folderu `deploy/`, bez FAISS / VoyageAI).
"""
from __future__ import annotations

import runpy
from pathlib import Path

_root = Path(__file__).resolve().parent
runpy.run_path(str(_root / "app_deploy.py"), run_name="__main__")
