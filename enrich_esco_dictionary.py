#!/usr/bin/env python3
"""
Enrich esco_dictionary.json with hierarchy data from ESCO API
for all Polish skill labels in jobs_database.db missing from the dictionary.

Uses concurrent requests for speed (~10 workers).
"""

import json
import sqlite3
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

DB_PATH = "jobs_database.db"
DICT_PATH = "esco_dictionary.json"
OUTPUT_PATH = "esco_dictionary.json"

API_SEARCH = "https://ec.europa.eu/esco/api/search"
API_RESOURCE = "https://ec.europa.eu/esco/api/resource/skill"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

WORKERS = 10
SAVE_EVERY = 500

dict_lock = Lock()


def api_get(url: str, retries=3) -> dict | None:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    return None


def fetch_one(label: str) -> tuple[str, dict | None]:
    """Search ESCO API for label, return (label, info_dict or None)."""
    # Step 1: search for URI
    url = (
        f"{API_SEARCH}?text={urllib.parse.quote(label)}"
        f"&language=pl&type=skill&full=false&limit=1"
    )
    data = api_get(url)
    if not data:
        return label, None

    results = data.get("_embedded", {}).get("results", [])
    if not results or results[0].get("title", "").lower() != label.lower():
        return label, None

    uri = results[0].get("uri", "")
    if not uri:
        return label, None

    # Step 2: get skill details + ancestors
    url2 = f"{API_RESOURCE}?uri={urllib.parse.quote(uri)}&language=pl"
    data2 = api_get(url2)
    if not data2:
        return label, None

    ancestors = data2.get("_embedded", {}).get("ancestors", [])
    if not ancestors:
        return label, None

    hierarchy = []
    for a in ancestors[1:]:
        title = a.get("title", "")
        a_uri = a.get("_links", {}).get("self", {}).get("uri", "")
        code = a_uri.rsplit("/", 1)[-1] if a_uri else "?"
        hierarchy.append({"code": code, "label": title})

    skill_type_link = data2.get("_links", {}).get("hasSkillType", {})
    if isinstance(skill_type_link, list) and skill_type_link:
        type_title = skill_type_link[0].get("title", "")
    elif isinstance(skill_type_link, dict):
        type_title = skill_type_link.get("title", "")
    else:
        type_title = ""

    if not hierarchy:
        return label, None

    return label, {"uri": uri, "type": type_title, "hierarchy": hierarchy}


def get_unmatched_labels(esco_dict: dict) -> list[tuple[str, int]]:
    existing_keys = set(esco_dict.keys())

    print("Scanning jobs_database.db…", flush=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT skills_esco_contextual FROM job_ads "
        "WHERE skills_esco_contextual IS NOT NULL AND skills_esco_contextual != ''"
    )
    counter: Counter = Counter()
    for (raw,) in c:
        try:
            for item in json.loads(raw):
                esco = item.get("esco", "").strip()
                if esco:
                    counter[esco] += 1
        except Exception:
            pass
    conn.close()

    return [(label, cnt) for label, cnt in counter.most_common() if label not in existing_keys]


def main():
    print("Loading dictionary…", flush=True)
    with open(DICT_PATH, encoding="utf-8") as f:
        esco_dict = json.load(f)
    print(f"  {len(esco_dict):,} entries", flush=True)

    unmatched = get_unmatched_labels(esco_dict)
    print(f"  {len(unmatched):,} unmatched labels to enrich", flush=True)
    if not unmatched:
        print("Nothing to do.")
        return

    labels = [l for l, _ in unmatched]
    added = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, label): label for label in labels}

        for i, future in enumerate(as_completed(futures), 1):
            label, info = future.result()
            if info:
                with dict_lock:
                    esco_dict[label] = info
                added += 1
            else:
                failed += 1

            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(labels) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(labels)}] added={added} failed={failed} "
                    f"rate={rate:.1f}/s ETA={remaining/60:.1f}min",
                    flush=True,
                )

            if added > 0 and added % SAVE_EVERY == 0:
                with dict_lock:
                    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(esco_dict, f, ensure_ascii=False, indent=2)
                print(f"  Checkpoint: {len(esco_dict):,} entries saved", flush=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(esco_dict, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Added {added:,}, failed {failed:,}.", flush=True)
    print(f"Dictionary: {len(esco_dict):,} entries → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
