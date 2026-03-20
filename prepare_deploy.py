#!/usr/bin/env python3
"""
Prepare the deploy/ folder for Streamlit Cloud.

Run once locally before committing:
    python prepare_deploy.py

What it does:
  1. Copies required files from app_deploy/ to deploy/ (see REQUIRED_FILES).
     Optionally copies trainings_regional_cache.json if present (BUR Trainings tab).
  2. Adds an FTS5 full-text search index to deploy/app_data.db so that
     app_deploy.py can search job titles without FAISS or VoyageAI.
"""

import os
import shutil
import sqlite3

SRC_DIR = os.path.join(os.path.dirname(__file__), "app_deploy")
DST_DIR = os.path.join(os.path.dirname(__file__), "deploy")

REQUIRED_FILES = [
    "app_data.db",
    "req_resp_slim.db",
    "skill_trends.db",
    "skills_stats_cache.json",
]

OPTIONAL_FILES = [
    "trainings_regional_cache.json",  # precompute_trainings_regional_cache.py
    "ai_tab_cache.json",
]


def copy_files():
    os.makedirs(DST_DIR, exist_ok=True)
    for fname in REQUIRED_FILES:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Source file not found: {src}\nRun `python export_app_data.py` first."
            )
        print(f"  Copying {fname}  ({os.path.getsize(src) / 1024 / 1024:.1f} MB)...")
        shutil.copy2(src, dst)
    for fname in OPTIONAL_FILES:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)
        if os.path.exists(src):
            print(f"  Copying {fname}  ({os.path.getsize(src) / 1024 / 1024:.1f} MB)...")
            shutil.copy2(src, dst)
        else:
            print(f"  SKIP (optional): {fname}")


def build_fts5(db_path: str):
    """Create (or rebuild) the FTS5 virtual table for job title search."""
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS job_titles_fts")

    # unicode61 with remove_diacritics=1 lets users search without Polish
    # diacritics (e.g. "pieleg" matches "pielęgniarz").
    conn.execute("""
        CREATE VIRTUAL TABLE job_titles_fts
        USING fts5(
            title,
            content='job_title_counts',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 1'
        )
    """)
    conn.execute("INSERT INTO job_titles_fts(job_titles_fts) VALUES('rebuild')")
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM job_titles_fts").fetchone()[0]
    conn.close()
    print(f"  FTS5 index built: {n:,} job titles indexed.")


def summarise():
    print("\nDeploy folder:")
    for fname in os.listdir(DST_DIR):
        size = os.path.getsize(os.path.join(DST_DIR, fname))
        print(f"  {fname:<35}  {size / 1024 / 1024:>7.1f} MB")


def main():
    print("=== prepare_deploy.py ===\n")

    print("[1/2] Copying data files...")
    copy_files()

    print("\n[2/2] Building FTS5 search index...")
    build_fts5(os.path.join(DST_DIR, "app_data.db"))

    summarise()
    print("\nDone. Commit the deploy/ folder (use Git LFS for *.db files).")


if __name__ == "__main__":
    main()
