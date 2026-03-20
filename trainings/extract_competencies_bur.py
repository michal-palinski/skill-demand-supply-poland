"""
Extract individual competencies from BUR trainings' cel_edukacyjny field
using Anthropic Message Batches API (Haiku, 50% cost reduction).

Reads from PostgreSQL, sends batch to Claude, saves each chunk to SQLite
(checkpoint) and finally exports parquet. Resume: skips ids already in SQLite
with n_competencies > 0.

Run: python3 trainings/extract_competencies_bur.py
If NumPy/numexpr errors: use system Python, e.g. python3.12 or a fresh venv.
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trainings_pl")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 1024

TEST_MODE = False
TEST_LIMIT = 10

OUT_DIR = BASE_DIR / "trainings" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── SQLite checkpoint (per chunk, process keeps running) ───────────────────────
def sqlite_path_for_run(test_mode: bool) -> Path:
    suffix = "_test" if test_mode else ""
    return OUT_DIR / f"bur_competencies_2025{suffix}.sqlite"


def init_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bur_competencies (
            id INTEGER PRIMARY KEY,
            competencies_json TEXT NOT NULL,
            n_competencies INTEGER NOT NULL,
            chunk_label TEXT,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def load_done_ids(path: Path) -> set[int]:
    """Ids with at least one extracted competency — skip on resume."""
    if not path.exists():
        return set()
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT id FROM bur_competencies WHERE n_competencies > 0"
    )
    out = {row[0] for row in cur.fetchall()}
    conn.close()
    return out


def load_all_from_sqlite(path: Path) -> dict[int, list]:
    if not path.exists():
        return {}
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT id, competencies_json FROM bur_competencies")
    out = {}
    for bid, js in cur.fetchall():
        try:
            out[int(bid)] = json.loads(js)
        except (json.JSONDecodeError, TypeError):
            out[int(bid)] = []
    conn.close()
    return out


def flush_results_to_sqlite(path: Path, results: dict[int, list], chunk_label: str) -> int:
    """Upsert one chunk; returns number of rows written."""
    if not results:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = []
    for bid, comps in results.items():
        if not isinstance(comps, list):
            comps = [str(comps)]
        rows.append(
            (int(bid), json.dumps(comps, ensure_ascii=False), len(comps), chunk_label, now)
        )
    conn.executemany(
        """
        INSERT INTO bur_competencies (id, competencies_json, n_competencies, chunk_label, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            competencies_json=excluded.competencies_json,
            n_competencies=excluded.n_competencies,
            chunk_label=excluded.chunk_label,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM bur_competencies").fetchone()[0]
    conn.close()
    print(f"  [SQLite] +{len(rows)} rows ({chunk_label}) → {n} rows in DB")
    return len(rows)


SYSTEM_PROMPT = """Jesteś ekspertem ds. analizy kompetencji zawodowych.

Z podanego opisu celu edukacyjnego szkolenia wyodrębnij listę KONKRETNYCH, 
POJEDYNCZYCH kompetencji (umiejętności i wiedza), jakie nabywają uczestnicy.

Zasady:
- Każda kompetencja to krótkie, SAMOWYSTARCZALNE sformułowanie (3-12 słów)
- KRYTYCZNE: każda kompetencja musi być zrozumiała BEZ kontekstu szkolenia. 
  Czytelnik nie zna tytułu ani opisu — musi rozumieć o co chodzi z samego sformułowania.
  ŹLE: "umiejętność stosowania metody" (jakiej metody?)
  ŹLE: "znajomość narzędzi" (jakich narzędzi?)
  ŹLE: "analiza danych" (zbyt ogólne jeśli w tekście jest konkret)
  DOBRZE: "umiejętność stosowania metody lean management"
  DOBRZE: "znajomość narzędzi analitycznych w Microsoft Excel"
  DOBRZE: "analiza danych sprzedażowych przy pomocy tabel przestawnych"
- Zawsze dołączaj nazwę dziedziny/technologii/metody, której dotyczy kompetencja
- Nie powtarzaj ogólników jak "podniesienie kompetencji" czy "rozwój umiejętności"
- Rozdzielaj wiedzę od umiejętności: "wiedza z zakresu X" i "umiejętność Y" to osobne pozycje
- Jeśli tekst wymienia kilka rzeczy w jednym zdaniu, rozdziel je na osobne kompetencje
- Wynik musi być poprawnym JSON — tablica stringów, bez komentarzy
- Kompetencje po polsku, w mianowniku lub w formie rzeczownikowej

Odpowiedz WYŁĄCZNIE tablicą JSON, np.:
["kompetencja 1", "kompetencja 2", "kompetencja 3"]"""


# ── Load data from PostgreSQL ────────────────────────────────────────────────
def load_data(limit: int | None = None) -> pd.DataFrame:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    q = """
        SELECT id, tytul, cel_edukacyjny
        FROM bur_services
        WHERE data_rozpoczecia_uslugi >= '2025-01-01'
          AND data_rozpoczecia_uslugi < '2026-01-01'
          AND cel_edukacyjny IS NOT NULL
          AND TRIM(cel_edukacyjny) != ''
    """
    if limit:
        q += f" ORDER BY id LIMIT {limit}"
    else:
        q += " ORDER BY id"
    df = pd.read_sql(text(q), engine)
    print(f"Loaded {len(df)} rows from PostgreSQL")
    return df


# ── Build batch requests ─────────────────────────────────────────────────────
def build_batch_requests(df: pd.DataFrame) -> list[Request]:
    requests = []
    for _, row in df.iterrows():
        cel = str(row["cel_edukacyjny"]).strip()
        if not cel:
            continue
        user_msg = f"Tytuł szkolenia: {row['tytul']}\n\nCel edukacyjny:\n{cel}"
        requests.append(
            Request(
                custom_id=f"bur_{row['id']}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                ),
            )
        )
    return requests


# ── Submit & poll single batch ────────────────────────────────────────────────
BATCH_CHUNK = 10_000
MAX_RETRIES = 3


def run_single_batch(client: anthropic.Anthropic, reqs: list[Request], label: str) -> tuple[dict, list[Request]]:
    """Returns (results_dict, failed_requests_for_retry)."""
    from tqdm import tqdm

    print(f"\n[{label}] Submitting {len(reqs)} requests...")
    for submit_attempt in range(5):
        try:
            batch = client.messages.batches.create(requests=reqs)
            break
        except (anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            wait = 30 * (2 ** submit_attempt)
            print(f"[{label}] Submit failed ({e.__class__.__name__}), retry in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"[{label}] Failed to submit batch after 5 attempts")
    print(f"[{label}] Batch ID: {batch.id}")

    total = len(reqs)
    pbar = tqdm(total=total, desc=f"{label} processing", unit="req")
    prev_done = 0
    poll_count = 0

    while batch.processing_status != "ended":
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        done = counts.succeeded + counts.errored
        pbar.update(done - prev_done)
        prev_done = done
        # API often keeps succeeded/errored=0 until batch ends; show processing for feedback
        proc = getattr(counts, "processing", None)
        postfix = dict(ok=counts.succeeded, err=counts.errored)
        if proc is not None:
            postfix["proc"] = proc
        pbar.set_postfix(**postfix)
        poll_count += 1
        if poll_count % 4 == 0:  # every ~2 min
            print(f"\n  [{label}] status={batch.processing_status} "
                  f"succeeded={counts.succeeded} errored={counts.errored} "
                  f"processing={getattr(counts, 'processing', '?')}")

    pbar.update(total - prev_done)
    pbar.close()
    print(f"[{label}] Done! succeeded={batch.request_counts.succeeded} "
          f"errored={batch.request_counts.errored}")

    results = {}
    failed_ids = set()
    req_by_id = {r["custom_id"]: r for r in reqs}

    for result in tqdm(client.messages.batches.results(batch.id),
                       total=total, desc=f"{label} fetching", unit="res"):
        cid = result.custom_id
        bur_id = int(cid.replace("bur_", ""))
        if result.result.type == "succeeded":
            content = result.result.message.content
            if content and len(content) > 0:
                text_content = content[0].text
                try:
                    competencies = json.loads(text_content)
                except json.JSONDecodeError:
                    competencies = [text_content]
                results[bur_id] = competencies
            else:
                failed_ids.add(cid)
        else:
            failed_ids.add(cid)

    failed_reqs = [req_by_id[cid] for cid in failed_ids if cid in req_by_id]
    if failed_reqs:
        print(f"[{label}] {len(failed_reqs)} requests to retry")
    return results, failed_reqs


def run_all_batches(requests: list[Request], sqlite_path: Path | None = None) -> dict:
    from tqdm import tqdm

    client = anthropic.Anthropic()
    all_results = {}

    chunks = [requests[i:i + BATCH_CHUNK] for i in range(0, len(requests), BATCH_CHUNK)]
    print(f"Total requests: {len(requests)} → {len(chunks)} batch(es) of {BATCH_CHUNK}")

    pending_chunks = list(enumerate(chunks))
    attempt = 0

    while pending_chunks and attempt < MAX_RETRIES:
        attempt += 1
        next_round = []

        for idx, chunk in tqdm(pending_chunks, desc=f"Round {attempt}", unit="batch"):
            label = f"R{attempt} Chunk {idx+1}/{len(chunks)}"
            results, failed = run_single_batch(client, chunk, label)
            all_results.update(results)
            if sqlite_path is not None and results:
                flush_results_to_sqlite(sqlite_path, results, label)
            if failed:
                next_round.append((idx, failed))

        if next_round:
            total_failed = sum(len(f) for _, f in next_round)
            print(f"\n--- Round {attempt} done. {total_failed} requests still failed, retrying... ---\n")
            time.sleep(60)
        pending_chunks = next_round

    if pending_chunks:
        total_failed = sum(len(f) for _, f in pending_chunks)
        print(f"\nWARNING: {total_failed} requests still failed after {MAX_RETRIES} retries")

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    limit = TEST_LIMIT if TEST_MODE else None
    label = f"TEST ({TEST_LIMIT})" if TEST_MODE else "FULL"
    print(f"=== {label} run ===\n")

    sqlite_path = sqlite_path_for_run(TEST_MODE)
    init_sqlite(sqlite_path)
    print(f"SQLite checkpoint: {sqlite_path}")

    df = load_data(limit=limit)
    done_ids = load_done_ids(sqlite_path)
    if done_ids:
        print(f"Resume: skipping {len(done_ids):,} ids already in SQLite (n_competencies > 0)")
    df_todo = df[~df["id"].isin(done_ids)].copy()
    requests = build_batch_requests(df_todo)
    print(f"Built {len(requests)} batch requests (todo / total: {len(requests):,} / {len(df):,})")

    results = run_all_batches(requests, sqlite_path=sqlite_path)
    print(f"\nGot results for {len(results)} rows (this session)")
    succeeded = sum(1 for v in results.values() if v)
    print(f"Succeeded (non-empty, session): {succeeded}")

    from_sqlite = load_all_from_sqlite(sqlite_path)
    merged = {**from_sqlite, **results}
    df["competencies"] = df["id"].map(lambda i: merged.get(i, []))
    df["n_competencies"] = df["competencies"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Preview (first 5 only for full runs)
    n_preview = 5 if not TEST_MODE else len(df)
    print("\n" + "=" * 70)
    for _, row in df.head(n_preview).iterrows():
        print(f"\n[{row['id']}] {row['tytul']}")
        comps = row["competencies"]
        if isinstance(comps, list):
            for c in comps:
                print(f"  • {c}")
        else:
            print("  (no results)")

    # Save
    suffix = "_test" if TEST_MODE else ""
    out_path = OUT_DIR / f"bur_competencies_2025{suffix}.parquet"
    save_df = df[["id", "tytul", "cel_edukacyjny", "competencies", "n_competencies"]].copy()
    save_df["competencies"] = save_df["competencies"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]"
    )
    save_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Avg competencies per training: {df['n_competencies'].mean():.1f}")
    print(f"Total competencies extracted: {df['n_competencies'].sum():,}")
