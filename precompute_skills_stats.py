#!/usr/bin/env python3
"""
Pre-compute ESCO skill category stats by matching job offer skills (via URI)
to the comprehensive_esco.db hierarchy (levels 1–4).

Pipeline:
  1. esco_dictionary.json: Polish skill label → URI
  2. comprehensive_esco.db: URI → code + ancestors at levels 1-4
  3. jobs_database.db: offers → Polish ESCO labels → aggregate counts

Output: skills_stats_cache.json  (nested tree L1→L2→L3→L4→leaf skills)
"""

import json
import sqlite3
from collections import Counter

JOBS_DB = "jobs_database.db"
ESCO_DB = "comprehensive_esco.db"
DICT_PATH = "esco_dictionary.json"
OUTPUT_PATH = "skills_stats_cache.json"

MAX_LEVEL = 4


def build_uri_to_hierarchy(esco_db_path: str) -> dict:
    conn = sqlite3.connect(esco_db_path)
    c = conn.cursor()
    c.execute("SELECT code, uri, title, level, parent_code, skill_type FROM esco_concepts")
    rows = c.fetchall()
    conn.close()

    code_map = {}
    uri_map = {}
    for code, uri, title, level, parent, stype in rows:
        code_map[code] = {
            "uri": uri, "title": title, "level": level,
            "parent_code": parent, "skill_type": stype,
        }
        if uri:
            uri_map[uri] = code

    def find_ancestor(code: str, target_level: int):
        visited = set()
        current = code
        while current and current not in visited:
            visited.add(current)
            info = code_map.get(current)
            if not info:
                return None, None
            if info["level"] == target_level:
                return current, info["title"]
            current = info["parent_code"]
        return None, None

    result = {}
    for uri, code in uri_map.items():
        info = code_map[code]
        ancestors = {}
        for lvl in range(1, MAX_LEVEL + 1):
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


def _ensure_path(tree: dict, path: list[tuple[str, str]]):
    """Walk/create nested dicts along path of (code, title) pairs."""
    node = tree
    for code, title in path:
        if code not in node:
            node[code] = {"title": title, "children": {}, "count": 0}
        node = node[code]["children"]
    return node


def main():
    print("Building URI → hierarchy from comprehensive_esco.db…", flush=True)
    uri_hier = build_uri_to_hierarchy(ESCO_DB)
    print(f"  {len(uri_hier):,} URIs with hierarchy", flush=True)

    print("Loading esco_dictionary.json (label → URI)…", flush=True)
    with open(DICT_PATH, encoding="utf-8") as f:
        esco_dict = json.load(f)
    label_to_uri = {k: v["uri"] for k, v in esco_dict.items() if "uri" in v}
    print(f"  {len(label_to_uri):,} labels with URIs", flush=True)

    print("Scanning jobs_database.db…", flush=True)
    conn = sqlite3.connect(JOBS_DB)
    c = conn.cursor()
    c.execute(
        "SELECT skills_esco_contextual FROM job_ads "
        "WHERE skills_esco_contextual IS NOT NULL AND skills_esco_contextual != ''"
    )

    total_offers = 0
    total_mentions = 0
    matched = 0
    unmatched_count = 0

    tree: dict = {}
    unmatched_skills: Counter = Counter()

    for (raw,) in c:
        try:
            items = json.loads(raw)
        except Exception:
            continue
        total_offers += 1
        for item in items:
            esco_label = item.get("esco", "").strip()
            if not esco_label:
                continue
            total_mentions += 1

            uri = label_to_uri.get(esco_label)
            hier = uri_hier.get(uri) if uri else None
            if not hier:
                unmatched_count += 1
                unmatched_skills[esco_label] += 1
                continue

            ancestors = hier["ancestors"]
            skill_title = hier["title"]
            skill_level = hier["level"]

            # Build path from L1 down to the deepest available ancestor (up to L4)
            path = []
            for lvl in range(1, MAX_LEVEL + 1):
                if lvl in ancestors:
                    path.append((ancestors[lvl]["code"], ancestors[lvl]["title"]))

            # Ensure path exists in tree and increment counts at each level
            node = tree
            for code, title in path:
                if code not in node:
                    node[code] = {"title": title, "children": {}, "count": 0}
                node[code]["count"] += 1
                node = node[code]["children"]

            # Add leaf skill count at the deepest node
            if skill_title not in node:
                node[skill_title] = {"title": skill_title, "children": {}, "count": 0}
            node[skill_title]["count"] += 1

            matched += 1

    conn.close()

    print(f"  {total_offers:,} offers, {total_mentions:,} mentions", flush=True)
    print(f"  Matched: {matched:,} ({matched/total_mentions*100:.1f}%)", flush=True)
    print(f"  Unmatched: {unmatched_count:,} ({unmatched_count/total_mentions*100:.1f}%)", flush=True)

    top_unmatched = [
        {"skill": k, "count": v}
        for k, v in unmatched_skills.most_common(100)
    ]

    output = {
        "meta": {
            "total_offers": total_offers,
            "total_mentions": total_mentions,
            "matched_mentions": matched,
            "unmatched_mentions": unmatched_count,
        },
        "tree": tree,
        "top_unmatched": top_unmatched,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {OUTPUT_PATH}", flush=True)

    # Summary
    print("\n--- L1 summary ---")
    for code in sorted(tree.keys()):
        node = tree[code]
        print(f"  {code:6} {node['count']:>10,}  {node['title']}")

    print("\n--- L2 summary (top 30) ---")
    l2_list = []
    for l1_code, l1_node in tree.items():
        for l2_code, l2_node in l1_node["children"].items():
            l2_list.append((l2_code, l2_node["title"], l2_node["count"]))
    l2_list.sort(key=lambda x: -x[2])
    for code, title, cnt in l2_list[:30]:
        print(f"  {code:8} {cnt:>10,}  {title}")


if __name__ == "__main__":
    main()
