#!/usr/bin/env python3
"""
Deploy ESCO Skills Search to Hugging Face Space.
- Tworzy Space repo
- Buduje indeks FAISS (embed_skills_kalm_faiss)
- Uploaduje indeks do HF Dataset
- Uploaduje pliki Space
- Ustawia HF_TOKEN w Space (wymaga ręcznego dodania w UI)
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

TOKEN = os.getenv("hf_token_write") or os.getenv("HF_TOKEN")
SPACE_ID = "skills-kalm"  # username/skills-kalm
DATASET_ID = "esco-skills-kalm-index"  # username/esco-skills-kalm-index
SPACES_DIR = BASE / "spaces" / "skills-kalm"


def create_space(token: str) -> str:
    """Create HF Space, return full repo_id (username/skills-kalm)."""
    from huggingface_hub import create_repo, whoami
    user = whoami(token=token)["name"]
    repo_id = f"{user}/{SPACE_ID}"
    # API przyjmuje tylko gradio|docker|static; README z sdk: streamlit nadpisze
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        token=token,
        exist_ok=True,
    )
    print(f"  Space: https://huggingface.co/spaces/{repo_id}")
    return repo_id


def build_faiss_index(token: str) -> tuple[Path, Path] | None:
    """Run embed script or use existing index, return (index_path, meta_path) or None."""
    idx = BASE / "faiss_indexes" / "skills_kalm.index"
    meta = BASE / "faiss_indexes" / "skills_kalm_metadata.pkl"
    if idx.exists() and meta.exists():
        print("  Używam istniejącego indeksu FAISS.")
        return idx, meta
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    env["hf_token_write"] = token
    print("  Uruchamiam embed_skills_kalm_faiss.py (może potrwać 1–2 h na GPU)...")
    proc = subprocess.run(
        [sys.executable, str(BASE / "embed_skills_kalm_faiss.py")],
        cwd=str(BASE),
        env=env,
        timeout=7200,  # 2h
        capture_output=True,
        text=True,
    )
    idx = BASE / "faiss_indexes" / "skills_kalm.index"
    meta = BASE / "faiss_indexes" / "skills_kalm_metadata.pkl"
    if proc.returncode != 0:
        print(f"  BŁĄD: {proc.stderr}")
        return None
    if not idx.exists() or not meta.exists():
        print("  Brak wygenerowanych plików FAISS.")
        return None
    print(f"  Wygenerowano: {idx}, {meta}")
    return idx, meta


def upload_dataset(token: str, index_path: Path, meta_path: Path) -> str:
    """Upload index to HF Dataset, return repo_id."""
    from huggingface_hub import create_repo, whoami, upload_file
    user = whoami(token=token)["name"]
    repo_id = f"{user}/{DATASET_ID}"
    create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
    print(f"  Uploaduję indeks do {repo_id}...")
    for path, name in [(index_path, "skills_kalm.index"), (meta_path, "skills_kalm_metadata.pkl")]:
        upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id, repo_type="dataset", token=token)
    print(f"  Dataset: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def upload_space_files(token: str, repo_id: str, dataset_repo_id: str):
    """Upload app files to Space and add INDEX_HF_REPO to README."""
    from huggingface_hub import upload_file
    readme = (SPACES_DIR / "README.md").read_text(encoding="utf-8")
    if dataset_repo_id and "INDEX_HF_REPO" not in readme:
        readme += f"\n\n## Konfiguracja\n\nW Space Secrets ustaw:\n- `HF_TOKEN` — token Hugging Face\n- `INDEX_HF_REPO` — `{dataset_repo_id}`\n"
    from io import BytesIO
    upload_file(path_or_fileobj=BytesIO(readme.encode("utf-8")), path_in_repo="README.md", repo_id=repo_id, repo_type="space", token=token)
    for fname in ["app.py", "requirements.txt"]:
        p = SPACES_DIR / fname
        if p.exists():
            upload_file(path_or_fileobj=str(p), path_in_repo=fname, repo_id=repo_id, repo_type="space", token=token)
    print("  Pliki Space wysłane.")


def main():
    if not TOKEN:
        print("Ustaw hf_token_write lub HF_TOKEN w .env")
        sys.exit(1)
    os.environ["HF_TOKEN"] = TOKEN

    print("=== Deploy ESCO Skills Search na Hugging Face ===\n")

    print("[1/4] Tworzenie Space...")
    repo_id = create_space(TOKEN)

    print("\n[2/4] Budowanie indeksu FAISS...")
    result = build_faiss_index(TOKEN)
    if result is None:
        print("  Pomijam (uruchom ręcznie: python embed_skills_kalm_faiss.py)")
        dataset_repo_id = None
    else:
        index_path, meta_path = result
        print("\n[3/4] Upload indeksu do datasetu...")
        dataset_repo_id = upload_dataset(TOKEN, index_path, meta_path)
    if dataset_repo_id is None:
        from huggingface_hub import whoami
        user = whoami(token=TOKEN)["name"]
        dataset_repo_id = f"{user}/{DATASET_ID}"
        print(f"  Docelowy dataset: {dataset_repo_id} (uploaduj indeks ręcznie)")

    print("\n[4/4] Upload plików Space...")
    upload_space_files(TOKEN, repo_id, dataset_repo_id)

    print("\n=== Gotowe ===")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print("Ustaw w Space → Settings → Variables and secrets:")
    print(f"  HF_TOKEN = <twój token>")
    print(f"  INDEX_HF_REPO = {dataset_repo_id}")
    print("Wybierz Hardware → A100.")


if __name__ == "__main__":
    main()
