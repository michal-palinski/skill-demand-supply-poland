#!/usr/bin/env python3
"""
Agregacja BUR (szkolenia) × województwo × grupa ESCO (L1 / L2).

Źródło: parquet z usługami BUR i dopasowanymi skillami ESCO (np. bur_trainings_0126.parquet).
Wyjście: lekki JSON dla Streamlit — bez jednostkowych rekordów w aplikacji.

Uruchomienie (z katalogu projektu) — typowy układ: **szkolenia** + **mapowanie ESCO**:

  python precompute_trainings_regional_cache.py \\
    --parquet trainings/data/bur_trainings_0126.parquet \\
    --esco-sqlite trainings/data/bur_to_esco_kalm_top1.sqlite

Jeśli w jednym parquet są już `esco_conceptUri` + `id`/`bur_bur_ids_json`, wystarczy --parquet.

Domyślnie zapis: app_deploy/trainings_regional_cache.json

Parquet szkoleń: `id`, `adres` (JSON BUR → nazwaWojewodztwa).
Źródło ESCO (gdy brak URI w parquet): SQLite/parquet KaLM z kolumnami
`esco_conceptUri`, `bur_bur_ids_json`, `similarity` i/lub `keep`.

Parquet: ParquetFile.iter_batches (bez pyarrow.dataset → pandas).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
DEFAULT_PARQUET = BASE / "trainings" / "data" / "bur_trainings_0126.parquet"
DEFAULT_ESCO_DB = BASE / "comprehensive_esco.db"
DEFAULT_SKILLS_CACHE = BASE / "app_deploy" / "skills_stats_cache.json"
DEFAULT_OUT = BASE / "app_deploy" / "trainings_regional_cache.json"
DEFAULT_SIM_THRESHOLD = 0.7
DEFAULT_ESCO_SQLITE = BASE / "trainings" / "data" / "bur_to_esco_kalm_top1.sqlite"
DEFAULT_ESCO_PARQUET = BASE / "trainings" / "data" / "bur_to_esco_kalm_top1.parquet"
BUR_ESCO_TABLE = "bur_esco_kalm_top1"

MAX_HIER_LEVEL = 4

_POLISH_VOIVODESHIPS = frozenset({
    "dolnośląskie", "kujawsko-pomorskie", "lubelskie", "lubuskie", "łódzkie",
    "małopolskie", "mazowieckie", "opolskie", "podkarpackie", "podlaskie",
    "pomorskie", "śląskie", "świętokrzyskie", "warmińsko-mazurskie",
    "wielkopolskie", "zachodniopomorskie",
})


def _canonical_voiv(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    low = name.strip().lower().replace("woj.", "").strip()
    if low in _POLISH_VOIVODESHIPS:
        return low
    for pl in _POLISH_VOIVODESHIPS:
        if pl in low or low in pl:
            return pl
    return None


def _voiv_from_adres_value(adres: Any) -> Optional[str]:
    """Wyciąga województwo z pola adres (BUR API: nazwaWojewodztwa)."""
    if adres is None:
        return None
    if isinstance(adres, str):
        s = adres.strip()
        if not s:
            return None
        if s.startswith("{"):
            try:
                return _voiv_from_adres_value(json.loads(s))
            except (json.JSONDecodeError, TypeError):
                pass
        return _canonical_voiv(s)
    if isinstance(adres, dict):
        kra = adres.get("nazwaKraju") or adres.get("nazwa_kraju")
        if kra and "polsk" not in str(kra).lower():
            return None
        nw = (
            adres.get("nazwaWojewodztwa")
            or adres.get("nazwa_wojewodztwa")
            or adres.get("wojewodztwo_nazwa")
        )
        if nw:
            return _canonical_voiv(str(nw))
        nested = adres.get("wojewodztwo") or adres.get("Wojewodztwo")
        if isinstance(nested, dict):
            n = nested.get("nazwa") or nested.get("nazwaWojewodztwa")
            if n:
                return _canonical_voiv(str(n))
        return None
    return None


def _is_esco_group_code(code: str) -> bool:
    if not code:
        return False
    return (
        code.startswith("S") or code.startswith("T") or code.startswith("L")
        or code[0].isdigit()
    )


def build_uri_to_hierarchy(esco_db_path: Path) -> dict[str, dict]:
    import sqlite3

    conn = sqlite3.connect(str(esco_db_path))
    c = conn.cursor()
    c.execute(
        "SELECT code, uri, title, level, parent_code, skill_type FROM esco_concepts"
    )
    rows = c.fetchall()
    conn.close()

    code_map: dict[str, dict] = {}
    uri_map: dict[str, str] = {}
    for code, uri, title, level, parent, stype in rows:
        code_map[code] = {
            "uri": uri,
            "title": title,
            "level": level,
            "parent_code": parent,
            "skill_type": stype,
        }
        if uri:
            uri_map[str(uri).strip()] = code

    def find_ancestor(code: str, target_level: int):
        visited: set[str] = set()
        current: Optional[str] = code
        while current and current not in visited:
            visited.add(current)
            info = code_map.get(current)
            if not info:
                return None, None
            if info["level"] == target_level:
                return current, info["title"]
            current = info["parent_code"]
        return None, None

    result: dict[str, dict] = {}
    for uri, code in uri_map.items():
        info = code_map[code]
        ancestors: dict[int, dict] = {}
        for lvl in range(1, MAX_HIER_LEVEL + 1):
            a_code, a_title = find_ancestor(code, lvl)
            if a_code:
                ancestors[lvl] = {"code": a_code, "title": a_title}
        if 1 not in ancestors:
            continue
        result[uri] = {
            "code": code,
            "title": info["title"],
            "level": info["level"],
            "skill_type": info["skill_type"],
            "ancestors": ancestors,
        }
    return result


def _match_uris_for_group(uri_hier: dict, group_level: int, group_code: str) -> set[str]:
    out: set[str] = set()
    for uri, h in uri_hier.items():
        anc = h.get("ancestors") or {}
        ent = anc.get(group_level) or {}
        if ent.get("code") == group_code:
            out.add(uri)
    return out


def _detect_columns(names: list[str]) -> dict[str, Optional[str]]:
    lower = {n.lower(): n for n in names}

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in lower:
                return lower[c]
        for c in cands:
            if c in names:
                return c
        return None

    return {
        "id": pick("id", "bur_id", "service_id", "training_id", "bur_service_id", "numer_id"),
        "adres": pick("adres", "address", "adres_json"),
        "uri": pick("esco_concepturi", "esco_concept_uri", "esco_uri", "concept_uri"),
        "sim": pick("similarity", "sim", "score"),
        "keep": pick("keep"),
        "ids_json": pick("bur_bur_ids_json", "bur_ids_json", "training_ids_json", "bur_bur_ids"),
    }


def _parse_id_list(raw: Any) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _norm_training_id(v: Any):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return str(v).strip()


def _use_uri_from_thresholds(
    keep_val: Any,
    sim_val: Any,
    *,
    has_keep: bool,
    has_sim: bool,
    sim_threshold: float,
) -> bool:
    if has_keep and keep_val is not None:
        if isinstance(keep_val, bool):
            return keep_val
        if isinstance(keep_val, (int, float)):
            return keep_val != 0
        try:
            return int(keep_val) != 0
        except (TypeError, ValueError):
            return False
    if has_sim and sim_val is not None:
        try:
            return float(sim_val) >= sim_threshold
        except (TypeError, ValueError):
            return False
    return True


def merge_uris_from_sqlite(
    tid_to_uris: dict[Any, set[str]],
    sqlite_path: Path,
    sim_threshold: float,
) -> int:
    """Dopisuje URI ESCO do tid z tabeli bur_esco_kalm_top1. Zwraca liczbę wierszy źródła."""
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({BUR_ESCO_TABLE})")
    cols = [r[1] for r in cur.fetchall()]
    if "esco_conceptUri" not in cols or "bur_bur_ids_json" not in cols:
        conn.close()
        raise ValueError(
            f"{sqlite_path}: tabela {BUR_ESCO_TABLE} — brak esco_conceptUri lub bur_bur_ids_json. "
            f"Kolumny: {cols}"
        )
    has_keep = "keep" in cols
    has_sim = "similarity" in cols
    select = ["esco_conceptUri", "bur_bur_ids_json"]
    if has_keep:
        select.append("keep")
    if has_sim:
        select.append("similarity")
    sel = ", ".join(select)
    n_src = 0
    for row in cur.execute(f"SELECT {sel} FROM {BUR_ESCO_TABLE}"):
        n_src += 1
        d = dict(zip(select, row))
        uri_s = (str(d["esco_conceptUri"] or "")).strip()
        if not uri_s:
            continue
        k = d.get("keep") if has_keep else None
        s = d.get("similarity") if has_sim else None
        if not _use_uri_from_thresholds(
            k, s, has_keep=has_keep, has_sim=has_sim, sim_threshold=sim_threshold
        ):
            continue
        for tid in _parse_id_list(d["bur_bur_ids_json"]):
            tid = _norm_training_id(tid)
            if tid is not None:
                tid_to_uris[tid].add(uri_s)
    conn.close()
    return n_src


def merge_uris_from_esco_parquet(
    tid_to_uris: dict[Any, set[str]],
    parquet_path: Path,
    sim_threshold: float,
) -> int:
    """Jak merge_uris_from_sqlite, ale wiersze z parquet KaLM (bez pyarrow.dataset)."""
    n_src = 0
    with pq.ParquetFile(str(parquet_path)) as pf:
        names = pf.schema_arrow.names
        cmap = _detect_columns(names)
        uri_c = cmap["uri"]
        ids_c = cmap["ids_json"]
        if not uri_c or not ids_c:
            raise ValueError(
                f"{parquet_path}: potrzebne kolumny esco_conceptUri i bur_bur_ids_json, jest: {names}"
            )
        keep_c = cmap["keep"]
        sim_c = cmap["sim"]
        read_cols = [c for c in [uri_c, ids_c, keep_c, sim_c] if c]
        for batch in pf.iter_batches(batch_size=50_000, columns=read_cols):
            d = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                n_src += 1
                uri_s = (str(d[uri_c][i] or "")).strip()
                if not uri_s:
                    continue
                k = d[keep_c][i] if keep_c else None
                s = d[sim_c][i] if sim_c else None
                has_keep = keep_c is not None
                has_sim = sim_c is not None
                if not _use_uri_from_thresholds(
                    k, s, has_keep=has_keep, has_sim=has_sim, sim_threshold=sim_threshold
                ):
                    continue
                for tid in _parse_id_list(d[ids_c][i]):
                    tid = _norm_training_id(tid)
                    if tid is not None:
                        tid_to_uris[tid].add(uri_s)
    return n_src


def load_skills_group_codes(skills_cache_path: Path) -> tuple[list[str], list[str]]:
    with open(skills_cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    tree = cache.get("tree") or {}
    l1_codes: list[str] = []
    l2_codes: list[str] = []
    for code, node in tree.items():
        if not _is_esco_group_code(code):
            continue
        l1_codes.append(code)
        for l2 in (node.get("children") or {}):
            l2_codes.append(l2)
    l1_codes = sorted(set(l1_codes))
    l2_codes = sorted(set(l2_codes))
    return l1_codes, l2_codes


def main() -> None:
    ap = argparse.ArgumentParser(description="BUR trainings × voiv × ESCO L1/L2 → JSON cache")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--esco-db", type=Path, default=DEFAULT_ESCO_DB)
    ap.add_argument("--skills-cache", type=Path, default=DEFAULT_SKILLS_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help="similarity >= próg (gdy brak kolumny keep)",
    )
    ap.add_argument(
        "--esco-sqlite",
        type=Path,
        default=None,
        help="bur_to_esco_kalm_top1.sqlite (esco_conceptUri + bur_bur_ids_json), gdy parquet szkoleń bez URI",
    )
    ap.add_argument(
        "--esco-parquet",
        type=Path,
        default=None,
        help="alternatywa: bur_to_esco_kalm_top1.parquet",
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Brak pliku parquet: {args.parquet}")
    if not args.esco_db.exists():
        raise FileNotFoundError(f"Brak {args.esco_db}")
    if not args.skills_cache.exists():
        raise FileNotFoundError(f"Brak {args.skills_cache} — uruchom najpierw precompute_skills_stats / export.")

    l1_codes, l2_codes = load_skills_group_codes(args.skills_cache)
    print(f"Grupy L1 (skills): {len(l1_codes)}, L2: {len(l2_codes)}", flush=True)

    print("Budowanie hierarchii ESCO (URI → L1–L4)…", flush=True)
    uri_hier = build_uri_to_hierarchy(args.esco_db)
    print(f"  {len(uri_hier):,} URI", flush=True)

    # Precompute URI sets per (level, code)
    uri_sets_l1 = {c: _match_uris_for_group(uri_hier, 1, c) for c in l1_codes}
    uri_sets_l2 = {c: _match_uris_for_group(uri_hier, 2, c) for c in l2_codes}

    esco_sqlite = args.esco_sqlite
    esco_parquet = args.esco_parquet

    with pq.ParquetFile(str(args.parquet)) as pf:
        names = pf.schema_arrow.names
        colmap = _detect_columns(names)
        print(f"Kolumny parquet (szkolenia): {names}", flush=True)
        print(f"Mapowanie: {colmap}", flush=True)

        id_col = colmap["id"]
        adres_col = colmap["adres"]
        uri_col = colmap["uri"]
        sim_col = colmap["sim"]
        keep_col = colmap["keep"]
        ids_json_col = colmap["ids_json"]

        use_external_esco = not uri_col
        if use_external_esco:
            if esco_sqlite is None and esco_parquet is None:
                if DEFAULT_ESCO_SQLITE.is_file():
                    esco_sqlite = DEFAULT_ESCO_SQLITE
                    print(f"Używam domyślnego ESCO SQLite: {esco_sqlite}", flush=True)
                elif DEFAULT_ESCO_PARQUET.is_file():
                    esco_parquet = DEFAULT_ESCO_PARQUET
                    print(f"Używam domyślnego ESCO parquet: {esco_parquet}", flush=True)
            if esco_sqlite is None and esco_parquet is None:
                raise ValueError(
                    "Parquet szkoleń nie ma kolumny esco_conceptUri. "
                    "Dołącz mapowanie KaLM, np.:\n"
                    "  --esco-sqlite trainings/data/bur_to_esco_kalm_top1.sqlite\n"
                    "lub zbuduj najpierw SQLite: python bur_esco_parquet_to_sqlite.py"
                )
            if not id_col:
                raise ValueError(
                    "Przy łączeniu z bur_to_esco potrzebna kolumna `id` (lub inna wykrywana jako id) w parquet szkoleń."
                )
        else:
            if not id_col and not ids_json_col:
                raise ValueError("Potrzebna kolumna id szkolenia lub bur_bur_ids_json.")

        if not adres_col:
            print(
                "UWAGA: brak kolumny adres — województwo nie zostanie przypisane "
                "(dodaj kolumnę `adres` z JSON BUR).",
                flush=True,
            )

        if use_external_esco:
            cols_read = [c for c in [id_col, adres_col] if c]
        else:
            cols_read = [c for c in [id_col, adres_col, uri_col, sim_col, keep_col, ids_json_col] if c]

        tid_to_voiv: dict[Any, str] = {}
        tid_to_uris: dict[Any, set[str]] = defaultdict(set)
        n_rows = 0

        for batch in pf.iter_batches(batch_size=50_000, columns=cols_read):
            d = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                n_rows += 1
                voiv = None
                if adres_col:
                    voiv = _voiv_from_adres_value(d[adres_col][i])

                if use_external_esco:
                    tid = _norm_training_id(d[id_col][i]) if id_col else None
                    if tid is not None and voiv is not None and tid not in tid_to_voiv:
                        tid_to_voiv[tid] = voiv
                    continue

                uri = d[uri_col][i] if uri_col else None
                uri_s = (str(uri).strip() if uri is not None else "") or ""

                use_uri = False
                if uri_s:
                    if keep_col is not None:
                        k = d[keep_col][i]
                        if k is None:
                            use_uri = False
                        elif isinstance(k, bool):
                            use_uri = k
                        elif isinstance(k, (int, float)):
                            use_uri = k != 0
                        else:
                            try:
                                use_uri = int(k) != 0
                            except (TypeError, ValueError):
                                use_uri = False
                    elif sim_col is not None:
                        s = d[sim_col][i]
                        try:
                            use_uri = float(s) >= args.sim_threshold
                        except (TypeError, ValueError):
                            use_uri = False
                    else:
                        use_uri = True

                tids: list = []
                if id_col:
                    tid = _norm_training_id(d[id_col][i])
                    if tid is not None:
                        tids.append(tid)
                if ids_json_col:
                    tids.extend(_parse_id_list(d[ids_json_col][i]))

                for tid in tids:
                    tid = _norm_training_id(tid)
                    if tid is None:
                        continue
                    if voiv is not None:
                        if tid not in tid_to_voiv:
                            tid_to_voiv[tid] = voiv
                    if use_uri and uri_s:
                        tid_to_uris[tid].add(uri_s)

    n_esco_rows = 0
    if use_external_esco:
        if esco_sqlite is not None:
            if not esco_sqlite.is_file():
                raise FileNotFoundError(esco_sqlite)
            print(f"Ładowanie dopasowań ESCO z SQLite ({esco_sqlite})…", flush=True)
            n_esco_rows = merge_uris_from_sqlite(
                tid_to_uris, esco_sqlite, args.sim_threshold
            )
        else:
            assert esco_parquet is not None
            if not esco_parquet.is_file():
                raise FileNotFoundError(esco_parquet)
            print(f"Ładowanie dopasowań ESCO z parquet ({esco_parquet})…", flush=True)
            n_esco_rows = merge_uris_from_esco_parquet(
                tid_to_uris, esco_parquet, args.sim_threshold
            )
        print(f"  Wierszy w źródle ESCO: {n_esco_rows:,}", flush=True)

    print(f"Przetworzono {n_rows:,} wierszy parquet (szkolenia).", flush=True)
    print(f"Szkolenia z województwem: {len(tid_to_voiv):,}", flush=True)
    print(f"Szkolenia z ≥1 URI (po progu): {sum(1 for t, u in tid_to_uris.items() if u):,}", flush=True)

    voiv_to_tids: dict[str, set[Any]] = defaultdict(set)
    for tid, v in tid_to_voiv.items():
        voiv_to_tids[v].add(tid)

    all_voivs = sorted(voiv_to_tids.keys())

    def aggregate_level(codes: list[str], uri_sets: dict[str, set[str]], label: str) -> dict:
        block: dict[str, Any] = {}
        for gcode in codes:
            mus = uri_sets.get(gcode) or set()
            if not mus:
                continue
            by_voiv: dict[str, dict[str, float | int]] = {}
            nat_tot = 0
            nat_hit = 0
            for voiv in all_voivs:
                tids = voiv_to_tids[voiv]
                n_tot = len(tids)
                if n_tot == 0:
                    continue
                n_hit = sum(1 for tid in tids if tid_to_uris[tid] & mus)
                pct = round(100.0 * n_hit / n_tot, 2) if n_tot else 0.0
                by_voiv[voiv] = {
                    "n_trainings": n_tot,
                    "n_with_group": n_hit,
                    "pct_trainings": pct,
                }
                nat_tot += n_tot
                nat_hit += n_hit
            nat_pct = round(100.0 * nat_hit / nat_tot, 2) if nat_tot else 0.0
            block[gcode] = {
                "national_n_trainings": nat_tot,
                "national_n_with_group": nat_hit,
                "national_pct": nat_pct,
                "n_uris_in_group": len(mus),
                "by_voivodeship": by_voiv,
            }
        print(f"  Zapisano {len(block)} grup {label}", flush=True)
        return block

    meta = {
        "source_parquet": str(args.parquet.resolve()),
        "similarity_threshold": args.sim_threshold,
        "n_parquet_rows": n_rows,
        "n_trainings_with_voivodeship": len(tid_to_voiv),
        "voivodeships": all_voivs,
        "external_esco": use_external_esco,
    }
    if esco_sqlite is not None:
        meta["source_esco_sqlite"] = str(esco_sqlite.resolve())
    if esco_parquet is not None:
        meta["source_esco_parquet"] = str(esco_parquet.resolve())
    if use_external_esco:
        meta["n_esco_source_rows"] = n_esco_rows

    out_obj = {
        "meta": meta,
        "L1": aggregate_level(l1_codes, uri_sets_l1, "L1"),
        "L2": aggregate_level(l2_codes, uri_sets_l2, "L2"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Zapisano → {args.out} ({args.out.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
