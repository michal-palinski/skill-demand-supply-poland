#!/usr/bin/env python3
"""Upload app files to existing HF Space (skills-kalm)."""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import whoami, upload_file

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
TOKEN = os.getenv("hf_token_write") or os.getenv("HF_TOKEN")
SPACES_DIR = BASE / "spaces" / "skills-kalm"

def main():
    if not TOKEN:
        raise SystemExit("Ustaw hf_token_write w .env")
    user = whoami(token=TOKEN)["name"]
    repo_id = f"{user}/skills-kalm"
    print(f"Upload do https://huggingface.co/spaces/{repo_id}")
    for fname in ["app.py", "README.md", "requirements.txt"]:
        p = SPACES_DIR / fname
        if p.exists():
            upload_file(path_or_fileobj=str(p), path_in_repo=fname, repo_id=repo_id, repo_type="space", token=TOKEN)
            print(f"  ✓ {fname}")
    skills_csv = BASE / "ESCO dataset - v1.2.1 - classification - pl - csv" / "skills_pl.csv"
    if skills_csv.exists():
        upload_file(path_or_fileobj=str(skills_csv), path_in_repo="skills_pl.csv", repo_id=repo_id, repo_type="space", token=TOKEN)
        print(f"  ✓ skills_pl.csv")
    bur_parquet = BASE / "trainings" / "data" / "bur_competencies_2025.parquet"
    if bur_parquet.exists():
        upload_file(path_or_fileobj=str(bur_parquet), path_in_repo="bur_competencies_2025.parquet", repo_id=repo_id, repo_type="space", token=TOKEN)
        print(f"  ✓ bur_competencies_2025.parquet")
    print("Gotowe.")

if __name__ == "__main__":
    main()
