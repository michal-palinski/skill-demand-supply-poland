#!/usr/bin/env python3
"""
Wczytuje bur_to_esco_kalm_top1.parquet (PyArrow) i zapisuje do SQLite.
Dodaje esco_preferredLabel_en przez dopasowanie esco_conceptUri do skills_en.csv.
Kolumna keep: 1 jeśli similarity >= próg (domyślnie 0.7), w przeciwnym razie 0.
Kody ESCO z comprehensive_esco.db (tabela esco_concepts): po URI dopisuje esco_code,
esco_level, esco_skill_type oraz esco_code_l1..l4 (przodkowie w hierarchii).

Uruchomienie:
  python bur_esco_parquet_to_sqlite.py
  python bur_esco_parquet_to_sqlite.py --parquet ścieżka.parquet --db ścieżka.sqlite

Uwagi (środowisko):
  • Parquet: używany jest ParquetFile().read() zamiast read_table(), żeby nie ładować
    pyarrow.dataset (w Anaconda często ciągnie pandas → błąd NumPy 2 vs numexpr/bottleneck).
  • Stara tabela SQLite bez kolumn esco_*: przed INSERT dodawane są brakujące kolumny (ALTER).
    W razie dziwnego stanu: python bur_esco_parquet_to_sqlite.py --replace
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq  # ParquetFile.read() — bez pyarrow.dataset (unika importu pandas)

BASE = Path(__file__).resolve().parent
DEFAULT_PARQUET = BASE / "trainings" / "data" / "bur_to_esco_kalm_top1.parquet"
DEFAULT_DB = BASE / "trainings" / "data" / "bur_to_esco_kalm_top1.sqlite"
DEFAULT_SKILLS_EN = (
    BASE
    / "ESCO dataset - v1.2.1 - classification - en - csv"
    / "skills_en.csv"
)
DEFAULT_ESCO_DB = BASE / "comprehensive_esco.db"

TABLE = "bur_esco_kalm_top1"
DEFAULT_KEEP_THRESHOLD = 0.7
MAX_HIER_LEVEL = 4


def load_uri_to_esco_codes(esco_db_path: Path) -> dict[str, dict]:
    """
    URI -> {esco_code, esco_level, esco_skill_type, esco_code_l1..l4}.
    Logika przodków jak w precompute_skills_stats.build_uri_to_hierarchy.
    """
    conn = sqlite3.connect(str(esco_db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT code, uri, level, parent_code, skill_type FROM esco_concepts"
    )
    rows = cur.fetchall()
    conn.close()

    code_map: dict[str, dict] = {}
    uri_map: dict[str, str] = {}
    for code, uri, level, parent, stype in rows:
        code_map[code] = {
            "level": level,
            "parent_code": parent,
            "skill_type": stype or "",
        }
        if uri:
            uri_map[str(uri).strip()] = code

    def find_ancestor_at_level(start_code: str, target_level: int) -> str | None:
        visited: set[str] = set()
        current: str | None = start_code
        while current and current not in visited:
            visited.add(current)
            info = code_map.get(current)
            if not info:
                return None
            if info["level"] == target_level:
                return current
            current = info["parent_code"]
        return None

    out: dict[str, dict] = {}
    for uri, leaf_code in uri_map.items():
        info = code_map[leaf_code]
        rec: dict[str, str | int | None] = {
            "esco_code": leaf_code,
            "esco_level": int(info["level"]) if info["level"] is not None else None,
            "esco_skill_type": str(info["skill_type"] or ""),
        }
        for lvl in range(1, MAX_HIER_LEVEL + 1):
            ac = find_ancestor_at_level(leaf_code, lvl)
            rec[f"esco_code_l{lvl}"] = ac
        out[uri] = rec
    return out


def load_uri_to_en_label(skills_en_path: Path) -> dict[str, str]:
    """conceptUri -> preferredLabel (EN)."""
    out: dict[str, str] = {}
    with open(skills_en_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = (row.get("conceptUri") or "").strip()
            if not uri:
                continue
            out[uri] = (row.get("preferredLabel") or "").strip()
    return out


def ensure_column(cur: sqlite3.Cursor, table: str, col: str, ddl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if col not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


# Kolumny dodawane do istniejącej tabeli (CREATE IF NOT EXISTS może zostawić starą wersję)
SQLITE_EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("esco_preferredLabel_en", "esco_preferredLabel_en TEXT"),
    ("keep", "keep INTEGER NOT NULL DEFAULT 0"),
    ("esco_code", "esco_code TEXT"),
    ("esco_level", "esco_level INTEGER"),
    ("esco_skill_type", "esco_skill_type TEXT"),
    ("esco_code_l1", "esco_code_l1 TEXT"),
    ("esco_code_l2", "esco_code_l2 TEXT"),
    ("esco_code_l3", "esco_code_l3 TEXT"),
    ("esco_code_l4", "esco_code_l4 TEXT"),
]


def read_parquet_simple(path: Path):
    """Odczyt parquet bez pq.read_table (nie ładuje pyarrow.dataset → pandas)."""
    with pq.ParquetFile(str(path)) as pf:
        return pf.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--skills-en", type=Path, default=DEFAULT_SKILLS_EN)
    ap.add_argument(
        "--esco-db",
        type=Path,
        default=DEFAULT_ESCO_DB,
        help="comprehensive_esco.db (esco_concepts: uri → code + hierarchia)",
    )
    ap.add_argument("--replace", action="store_true", help="Usuń tabelę jeśli istnieje")
    ap.add_argument(
        "--keep-threshold",
        type=float,
        default=DEFAULT_KEEP_THRESHOLD,
        help=f"keep=1 gdy similarity >= ta wartość (domyślnie {DEFAULT_KEEP_THRESHOLD})",
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    if not args.skills_en.exists():
        raise FileNotFoundError(
            f"Brak {args.skills_en} — podaj --skills-en lub dodaj plik ESCO EN."
        )
    if not args.esco_db.exists():
        raise FileNotFoundError(
            f"Brak {args.esco_db} — podaj --esco-db lub dodaj comprehensive_esco.db."
        )

    print("Ładowanie comprehensive_esco.db (URI → kody ESCO)…")
    uri_to_codes = load_uri_to_esco_codes(args.esco_db)
    print(f"  {len(uri_to_codes):,} URI w esco_concepts")

    print("Ładowanie skills_en.csv (URI → label EN)…")
    uri_to_en = load_uri_to_en_label(args.skills_en)
    print(f"  {len(uri_to_en):,} skills EN")

    table_pa = read_parquet_simple(args.parquet)
    names = list(table_pa.column_names)
    if "esco_conceptUri" not in names:
        raise ValueError(f"Brak kolumny esco_conceptUri w parquet: {names}")
    if "similarity" not in names:
        raise ValueError(f"Brak kolumny similarity w parquet: {names}")

    cols_py = [table_pa.column(i).to_pylist() for i in range(table_pa.num_columns)]
    n = table_pa.num_rows
    rows = list(zip(*cols_py))

    uri_col_i = names.index("esco_conceptUri")
    sim_col_i = names.index("similarity")
    thr = float(args.keep_threshold)
    en_for_row = [uri_to_en.get((r[uri_col_i] or "").strip(), "") for r in rows]
    keep_for_row: list[int] = []
    for r in rows:
        try:
            s = float(r[sim_col_i])
        except (TypeError, ValueError):
            s = 0.0
        keep_for_row.append(1 if s >= thr else 0)
    extra_cols = [
        "esco_preferredLabel_en",
        "keep",
        "esco_code",
        "esco_level",
        "esco_skill_type",
        "esco_code_l1",
        "esco_code_l2",
        "esco_code_l3",
        "esco_code_l4",
    ]
    code_rows: list[tuple[str | int | None, ...]] = []
    for r in rows:
        uri = (r[uri_col_i] or "").strip()
        crec = uri_to_codes.get(uri)
        if crec:
            code_rows.append(
                (
                    crec["esco_code"],
                    crec["esco_level"],
                    crec["esco_skill_type"],
                    crec["esco_code_l1"],
                    crec["esco_code_l2"],
                    crec["esco_code_l3"],
                    crec["esco_code_l4"],
                )
            )
        else:
            code_rows.append((None, None, None, None, None, None, None))

    names_out = names + extra_cols
    rows_out = [
        tuple(list(r) + [en, k] + list(cr))
        for r, en, k, cr in zip(rows, en_for_row, keep_for_row, code_rows)
    ]

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.db))
    cur = conn.cursor()
    if args.replace:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bur_competency TEXT NOT NULL,
            bur_bur_ids_json TEXT,
            bur_n_trainings INTEGER,
            esco_preferredLabel TEXT,
            esco_conceptUri TEXT,
            similarity REAL,
            esco_preferredLabel_en TEXT,
            keep INTEGER NOT NULL DEFAULT 0,
            esco_code TEXT,
            esco_level INTEGER,
            esco_skill_type TEXT,
            esco_code_l1 TEXT,
            esco_code_l2 TEXT,
            esco_code_l3 TEXT,
            esco_code_l4 TEXT
        )
        """
    )
    for col, ddl in SQLITE_EXTRA_COLUMNS:
        ensure_column(cur, TABLE, col, ddl)

    cur.execute(f"DELETE FROM {TABLE}")

    placeholders = ",".join("?" * len(names_out))
    insert_cols = ",".join(names_out)
    cur.executemany(
        f"INSERT INTO {TABLE} ({insert_cols}) VALUES ({placeholders})",
        rows_out,
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_bur ON {TABLE} (bur_competency)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_esco_uri ON {TABLE} (esco_conceptUri)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_esco_code ON {TABLE} (esco_code)"
    )
    conn.commit()
    conn.close()

    n_matched = sum(1 for e in en_for_row if e)
    n_keep = sum(keep_for_row)
    n_codes = sum(1 for cr in code_rows if cr[0])
    print(f"Zapisano {n:,} wierszy → {args.db} (tabela {TABLE})")
    print(f"  Z angielskim labelem (URI w skills_en): {n_matched:,} / {n:,}")
    print(f"  Z kodem ESCO (URI w comprehensive_esco.db): {n_codes:,} / {n:,}")
    print(f"  keep=1 (similarity >= {thr}): {n_keep:,} / {n:,}")


if __name__ == "__main__":
    main()
