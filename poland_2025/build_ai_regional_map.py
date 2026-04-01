#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REGIONAL_METRICS_CSV = BASE / "data" / "regional_metrics.csv"
BRIDGE_CSV = ROOT / "lightcast_aiml" / "bridge_esco" / "lightcast_to_esco_bridge_top1.csv"
ESCO_DB = ROOT / "comprehensive_esco.db"
ESCO_DICT = ROOT / "esco_dictionary.json"
JOBS_DB = ROOT / "jobs_database.db"
OUTPUT_JS = BASE / "assets" / "ai_regional_map_data.js"

VOIV_ORDER = [
    "dolnośląskie",
    "kujawsko-pomorskie",
    "lubelskie",
    "lubuskie",
    "łódzkie",
    "małopolskie",
    "mazowieckie",
    "opolskie",
    "podkarpackie",
    "podlaskie",
    "pomorskie",
    "śląskie",
    "świętokrzyskie",
    "warmińsko-mazurskie",
    "wielkopolskie",
    "zachodniopomorskie",
]

VOIV_ALIASES = {
    "dolnośląskie": ["dolnośląskie", "lower silesia", "lower-silesia"],
    "kujawsko-pomorskie": [
        "kujawsko-pomorskie",
        "kujawsko pomorskie",
        "kuyavian-pomeranian",
        "kuyavian pomeranian",
    ],
    "lubelskie": ["lubelskie", "lublin voivodeship"],
    "lubuskie": ["lubuskie", "lubusz"],
    "łódzkie": ["łódzkie", "lodzkie", "lodz voivodeship"],
    "małopolskie": ["małopolskie", "malopolskie", "lesser poland", "lesser-poland"],
    "mazowieckie": ["mazowieckie", "masovian", "mazovia"],
    "opolskie": ["opolskie", "opole voivodeship"],
    "podkarpackie": ["podkarpackie", "subcarpathian"],
    "podlaskie": ["podlaskie", "podlasie"],
    "pomorskie": ["pomorskie", "pomeranian"],
    "śląskie": ["śląskie", "slaskie", "silesian"],
    "świętokrzyskie": ["świętokrzyskie", "swietokrzyskie", "holy cross"],
    "warmińsko-mazurskie": [
        "warmińsko-mazurskie",
        "warminsko-mazurskie",
        "warmian-masurian",
        "warmian masurian",
    ],
    "wielkopolskie": ["wielkopolskie", "greater poland", "greater-poland"],
    "zachodniopomorskie": [
        "zachodniopomorskie",
        "west pomeranian",
        "western pomeranian",
    ],
}

CITY_TO_VOIV = {
    r"warszaw": "mazowieckie",
    r"radom": "mazowieckie",
    r"płock": "mazowieckie",
    r"plock": "mazowieckie",
    r"siedlc": "mazowieckie",
    r"pruszk[óo]w": "mazowieckie",
    r"krak[óo]w": "małopolskie",
    r"tarn[óo]w": "małopolskie",
    r"nowy sącz": "małopolskie",
    r"nowy sacz": "małopolskie",
    r"wroc[łl]aw": "dolnośląskie",
    r"legnic": "dolnośląskie",
    r"wałbrzych": "dolnośląskie",
    r"walbrzych": "dolnośląskie",
    r"jelenia g[óo]ra": "dolnośląskie",
    r"pozna[nń]": "wielkopolskie",
    r"kalisz": "wielkopolskie",
    r"konin": "wielkopolskie",
    r"pi[łl]a": "wielkopolskie",
    r"gda[nń]sk": "pomorskie",
    r"gdyni": "pomorskie",
    r"sopot": "pomorskie",
    r"słupsk": "pomorskie",
    r"slupsk": "pomorskie",
    r"[łl][óo]d[zź]": "łódzkie",
    r"piotrk[óo]w": "łódzkie",
    r"skierniewic": "łódzkie",
    r"katowic": "śląskie",
    r"gliwic": "śląskie",
    r"sosnowiec": "śląskie",
    r"tychy": "śląskie",
    r"dąbrowa g[óo]rnicza": "śląskie",
    r"dabrowa gornicza": "śląskie",
    r"częstochow": "śląskie",
    r"czestochow": "śląskie",
    r"bielsko": "śląskie",
    r"zabrze": "śląskie",
    r"bytom": "śląskie",
    r"rybnik": "śląskie",
    r"szczecin": "zachodniopomorskie",
    r"koszalin": "zachodniopomorskie",
    r"świnoujście": "zachodniopomorskie",
    r"swinoujscie": "zachodniopomorskie",
    r"lublin": "lubelskie",
    r"zamość": "lubelskie",
    r"zamosc": "lubelskie",
    r"chełm": "lubelskie",
    r"chelm": "lubelskie",
    r"bydgoszcz": "kujawsko-pomorskie",
    r"toru[nń]": "kujawsko-pomorskie",
    r"włocławek": "kujawsko-pomorskie",
    r"wloclawek": "kujawsko-pomorskie",
    r"grudziądz": "kujawsko-pomorskie",
    r"grudziadz": "kujawsko-pomorskie",
    r"rzesz[óo]w": "podkarpackie",
    r"krosno": "podkarpackie",
    r"mielec": "podkarpackie",
    r"przemyśl": "podkarpackie",
    r"przemysl": "podkarpackie",
    r"tarnobrzeg": "podkarpackie",
    r"bia[łl]ystok": "podlaskie",
    r"łomża": "podlaskie",
    r"lomza": "podlaskie",
    r"suwałk": "podlaskie",
    r"suwalk": "podlaskie",
    r"kielc": "świętokrzyskie",
    r"ostrowiec": "świętokrzyskie",
    r"starachowic": "świętokrzyskie",
    r"olsztyn": "warmińsko-mazurskie",
    r"elbląg": "warmińsko-mazurskie",
    r"elblag": "warmińsko-mazurskie",
    r"ełk": "warmińsko-mazurskie",
    r"elk": "warmińsko-mazurskie",
    r"opol": "opolskie",
    r"kędzierzyn": "opolskie",
    r"kedzierzyn": "opolskie",
    r"nysa": "opolskie",
    r"zielona g[óo]ra": "lubuskie",
    r"gorz[óo]w": "lubuskie",
}

FOREIGN_MARKERS = [
    "zagranica",
    "niemcy",
    "germany",
    "netherlands",
    "holand",
    "belgia",
    "belgium",
    "france",
    "francja",
    "austria",
    "norweg",
    "denmark",
    "dania",
    "sweden",
    "szwec",
    "wielka brytania",
    "united kingdom",
    "usa",
    "united states",
]


def load_regional_base() -> dict[str, dict]:
    with REGIONAL_METRICS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        voiv = row["voivodeship"]
        out[voiv] = {
            "voivodeship": voiv,
            "labour_force_2025_avg": int(float(row["labour_force_2025_avg"])),
            "offers": int(float(row["offers"])),
        }
    return out


def load_bridge() -> tuple[set[str], dict[str, dict]]:
    with BRIDGE_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ai_codes = {row["esco_code"] for row in rows}
    code_meta = {
        row["esco_code"]: {
            "label": row["esco_label"],
            "group": row["esco_group"],
        }
        for row in rows
    }
    return ai_codes, code_meta


def load_label_to_code() -> dict[str, str]:
    with ESCO_DICT.open(encoding="utf-8") as f:
        esco_dict = json.load(f)
    label_to_uri = {
        str(label).strip().lower(): meta.get("uri")
        for label, meta in esco_dict.items()
        if meta.get("uri")
    }

    conn = sqlite3.connect(str(ESCO_DB))
    cur = conn.cursor()
    cur.execute("SELECT code, uri FROM esco_concepts WHERE uri IS NOT NULL")
    uri_to_code = {uri: code for code, uri in cur.fetchall()}
    conn.close()

    out = {}
    for label, uri in label_to_uri.items():
        code = uri_to_code.get(uri)
        if code:
            out[label] = code
    return out


def build_location_matchers():
    alias_patterns = []
    for voiv, aliases in VOIV_ALIASES.items():
        for alias in aliases:
            alias_patterns.append((voiv, re.compile(re.escape(alias), re.IGNORECASE)))
    city_patterns = [
        (voiv, re.compile(pattern, re.IGNORECASE))
        for pattern, voiv in CITY_TO_VOIV.items()
    ]
    return alias_patterns, city_patterns


def infer_voivodeship(
    location: str | None,
    alias_patterns: list[tuple[str, re.Pattern]],
    city_patterns: list[tuple[str, re.Pattern]],
) -> str | None:
    if not location:
        return None
    text = str(location).strip().lower()
    if not text:
        return None
    if any(marker in text for marker in FOREIGN_MARKERS):
        return None

    matches: list[tuple[int, str]] = []
    for voiv, pattern in alias_patterns:
        match = pattern.search(text)
        if match:
            matches.append((match.start(), voiv))
    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    city_matches: list[tuple[int, str]] = []
    for voiv, pattern in city_patterns:
        match = pattern.search(text)
        if match:
            city_matches.append((match.start(), voiv))
    if city_matches:
        city_matches.sort(key=lambda item: item[0])
        return city_matches[0][1]

    return None


def log_likelihood_g2(k11: int, k12: int, k21: int, k22: int) -> float:
    total = k11 + k12 + k21 + k22
    if total == 0:
        return 0.0

    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0

    expected = (
        row1 * col1 / total,
        row1 * col2 / total,
        row2 * col1 / total,
        row2 * col2 / total,
    )
    observed = (k11, k12, k21, k22)

    g2 = 0.0
    for obs, exp in zip(observed, expected):
        if obs > 0 and exp > 0:
            g2 += obs * math.log(obs / exp)
    g2 *= 2.0

    # Keep only positively characteristic concepts for the focal voivodeship.
    return g2 if k11 > expected[0] else -g2


def compute_payload() -> dict:
    regional_base = load_regional_base()
    ai_codes, code_meta = load_bridge()
    label_to_code = load_label_to_code()
    alias_patterns, city_patterns = build_location_matchers()

    ai_offer_counts = Counter()
    concept_counts = defaultdict(Counter)
    total_ai_offers = 0
    mapped_ai_offers = 0
    unmapped_ai_offers = 0

    conn = sqlite3.connect(str(JOBS_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, location, skills_esco_contextual FROM job_ads "
        "WHERE skills_esco_contextual IS NOT NULL AND trim(skills_esco_contextual) != ''"
    )

    while True:
        batch = cur.fetchmany(5000)
        if not batch:
            break
        for _, location, payload in batch:
            try:
                items = json.loads(payload)
            except Exception:
                continue

            ai_codes_in_offer: set[str] = set()
            for item in items:
                label = str(item.get("esco") or item.get("label") or "").strip().lower()
                if not label:
                    continue
                code = label_to_code.get(label)
                if code in ai_codes:
                    ai_codes_in_offer.add(code)

            if not ai_codes_in_offer:
                continue

            total_ai_offers += 1
            voiv = infer_voivodeship(location, alias_patterns, city_patterns)
            if not voiv:
                unmapped_ai_offers += 1
                continue

            mapped_ai_offers += 1
            ai_offer_counts[voiv] += 1
            for code in ai_codes_in_offer:
                concept_counts[voiv][code] += 1

    conn.close()

    global_concept_counts = Counter()
    for voiv in VOIV_ORDER:
        global_concept_counts.update(concept_counts[voiv])

    mapped_total_ai_offers = sum(ai_offer_counts.values())

    rows = []
    for voiv in VOIV_ORDER:
        base = regional_base[voiv]
        region_ai_offers = ai_offer_counts[voiv]
        outside_ai_offers = mapped_total_ai_offers - region_ai_offers
        scored = []
        for code, count in concept_counts[voiv].items():
            outside_count = global_concept_counts[code] - count
            absent_in_region = max(region_ai_offers - count, 0)
            absent_outside_region = max(outside_ai_offers - outside_count, 0)
            g2 = log_likelihood_g2(
                count,
                absent_in_region,
                outside_count,
                absent_outside_region,
            )
            scored.append(
                {
                    "code": code,
                    "label": code_meta[code]["label"],
                    "group": code_meta[code]["group"],
                    "count": count,
                    "g2": round(g2, 6),
                }
            )
        scored = [item for item in scored if item["g2"] > 0]
        scored.sort(key=lambda item: (-item["g2"], -item["count"], item["label"]))
        top_skills = scored[:3]
        top_skills_text = ", ".join(item["label"] for item in top_skills) if top_skills else "No AI skills detected"

        ai_offers = ai_offer_counts[voiv]
        rows.append(
            {
                "voivodeship": voiv,
                "offers": base["offers"],
                "labour_force_2025_avg": base["labour_force_2025_avg"],
                "ai_offers": ai_offers,
                "ai_offers_per_100k_lf": round(ai_offers * 100000 / base["labour_force_2025_avg"], 2),
                "ai_offer_share_pct": round(ai_offers * 100 / base["offers"], 2) if base["offers"] else 0.0,
                "top_characteristic_skills": top_skills,
                "top_characteristic_skills_text": top_skills_text,
            }
        )

    rows.sort(key=lambda item: item["ai_offers_per_100k_lf"], reverse=True)
    top_row = rows[0]

    return {
        "rows": rows,
        "meta": {
            "ai_offers_total": total_ai_offers,
            "ai_offers_total_mapped": mapped_ai_offers,
            "ai_offers_total_unmapped": unmapped_ai_offers,
            "top_voivodeship": top_row["voivodeship"],
            "top_ai_offers_per_100k_lf": top_row["ai_offers_per_100k_lf"],
        },
    }


def main() -> None:
    payload = compute_payload()
    OUTPUT_JS.write_text(
        "window.__AI_REGIONAL_REPORT__ = "
        + json.dumps(payload, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_JS}")
    print(
        "Mapped AI offers:",
        payload["meta"]["ai_offers_total_mapped"],
        "/",
        payload["meta"]["ai_offers_total"],
    )


if __name__ == "__main__":
    main()
