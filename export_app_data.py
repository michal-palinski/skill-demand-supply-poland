#!/usr/bin/env python3
"""
Export only the data needed by app_search.py into a slim deployment package.

Output directory: app_deploy/
  app_data.db            — job_title_counts + job_kzis_matches  (replaces 3.4 GB jobs_database.db)
  req_resp_slim.db       — sample_offers + skill_matches         (replaces 295 MB req_resp_embeddings.db)
  skill_trends.db        — copied as-is (16 MB, already small)
  skills_stats_cache.json— copied as-is (2 MB)
  faiss_indexes/         — job_titles + kzis (727 MB + 12 MB + pkl files)

Trainings tab (BUR): wygeneruj `app_deploy/trainings_regional_cache.json` przez
`python precompute_trainings_regional_cache.py --parquet trainings/data/bur_trainings_0126.parquet`
(nie jest tworzony przez ten skrypt eksportu).
"""

import os
import shutil
import sqlite3
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC / "app_deploy"

JOBS_DB   = SRC / "jobs_database.db"
RR_DB     = SRC / "req_resp_embeddings.db"
TRENDS_DB = SRC / "skill_trends.db"
CACHE_JSON = SRC / "skills_stats_cache.json"

FAISS_SRC = SRC / "faiss_indexes"
FAISS_FILES = [
    "job_titles.index",
    "job_titles_metadata.pkl",
    "kzis_occupations.index",
    "kzis_occupations_metadata.pkl",
]


def mb(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1_073_741_824:
        return f"{size / 1_073_741_824:.1f} GB"
    return f"{size / 1_048_576:.1f} MB"


def export_app_data_db():
    """
    From jobs_database.db extract:
      - job_title_counts  (title, count)   — pre-aggregated COUNT per title
      - job_kzis_matches  (job_title, rank, kzis_occupation_name, similarity_score)
    """
    out_path = OUT / "app_data.db"
    print(f"\n[1/4] Exporting app_data.db …")

    src = sqlite3.connect(JOBS_DB)
    dst = sqlite3.connect(out_path)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=OFF")

    # job_title_counts
    print("  Reading job_title_counts …")
    rows = src.execute(
        "SELECT title, COUNT(*) AS count FROM job_ads WHERE title IS NOT NULL GROUP BY title"
    ).fetchall()
    dst.execute("CREATE TABLE IF NOT EXISTS job_title_counts (title TEXT PRIMARY KEY, count INTEGER)")
    dst.executemany("INSERT OR REPLACE INTO job_title_counts VALUES (?,?)", rows)
    dst.commit()
    print(f"  job_title_counts: {len(rows):,} rows")

    # job_kzis_matches (only columns used by the app)
    print("  Reading job_kzis_matches …")
    rows = src.execute(
        "SELECT job_title, rank, kzis_occupation_name, CAST(similarity_score AS REAL) "
        "FROM job_kzis_matches"
    ).fetchall()
    dst.execute("""
        CREATE TABLE IF NOT EXISTS job_kzis_matches (
            job_title TEXT,
            rank INTEGER,
            kzis_occupation_name TEXT,
            similarity_score REAL
        )
    """)
    dst.execute("CREATE INDEX IF NOT EXISTS idx_kzis_title ON job_kzis_matches(job_title)")
    dst.executemany("INSERT INTO job_kzis_matches VALUES (?,?,?,?)", rows)
    dst.commit()
    print(f"  job_kzis_matches: {len(rows):,} rows")

    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"  → {out_path.name}  {mb(out_path)}")


def export_req_resp_slim():
    """
    From req_resp_embeddings.db extract:
      - sample_offers  (job_id, title)
      - skill_matches  (job_id, item_type, item_text, skill_label, skill_type, similarity, rank)
    The 'embeddings' table (~295 MB worth) is NOT copied.
    """
    out_path = OUT / "req_resp_slim.db"
    print(f"\n[2/4] Exporting req_resp_slim.db …")

    src = sqlite3.connect(RR_DB)
    dst = sqlite3.connect(out_path)
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=OFF")

    # sample_offers
    rows = src.execute("SELECT job_id, title FROM sample_offers ORDER BY title").fetchall()
    dst.execute("CREATE TABLE IF NOT EXISTS sample_offers (job_id INTEGER PRIMARY KEY, title TEXT)")
    dst.executemany("INSERT OR REPLACE INTO sample_offers VALUES (?,?)", rows)
    dst.commit()
    print(f"  sample_offers: {len(rows):,} rows")

    # skill_matches
    rows = src.execute("""
        SELECT job_id, item_type, item_text, skill_label, skill_type, similarity, rank
        FROM skill_matches
    """).fetchall()
    dst.execute("""
        CREATE TABLE IF NOT EXISTS skill_matches (
            job_id INTEGER,
            item_type TEXT,
            item_text TEXT,
            skill_label TEXT,
            skill_type TEXT,
            similarity REAL,
            rank INTEGER
        )
    """)
    dst.execute("CREATE INDEX IF NOT EXISTS idx_sm_job ON skill_matches(job_id)")
    dst.executemany("INSERT INTO skill_matches VALUES (?,?,?,?,?,?,?)", rows)
    dst.commit()
    print(f"  skill_matches: {len(rows):,} rows")

    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"  → {out_path.name}  {mb(out_path)}")


def copy_small_files():
    print(f"\n[3/4] Copying skill_trends.db and skills_stats_cache.json …")
    for src_path in [TRENDS_DB, CACHE_JSON]:
        dst_path = OUT / src_path.name
        shutil.copy2(src_path, dst_path)
        print(f"  → {dst_path.name}  {mb(dst_path)}")


def copy_faiss_indexes():
    print(f"\n[4/4] Copying FAISS indexes …")
    faiss_dst = OUT / "faiss_indexes"
    faiss_dst.mkdir(exist_ok=True)
    for fname in FAISS_FILES:
        src_path = FAISS_SRC / fname
        dst_path = faiss_dst / fname
        if not src_path.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        shutil.copy2(src_path, dst_path)
        print(f"  → faiss_indexes/{fname}  {mb(dst_path)}")


def summary():
    print("\n── Deployment package size ──────────────────────────────")
    total = 0
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            s = p.stat().st_size
            total += s
            rel = p.relative_to(OUT)
            print(f"  {str(rel):<55} {mb(p):>8}")
    if total >= 1_073_741_824:
        print(f"\n  TOTAL: {total / 1_073_741_824:.2f} GB")
    else:
        print(f"\n  TOTAL: {total / 1_048_576:.0f} MB")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    export_app_data_db()
    export_req_resp_slim()
    copy_small_files()
    copy_faiss_indexes()
    summary()
    print(
        "\nDone. Run `python prepare_deploy.py` then commit deploy/ (Git LFS for *.db).\n"
        "Trainings tab: `python precompute_trainings_regional_cache.py` → "
        "app_deploy/trainings_regional_cache.json"
    )
