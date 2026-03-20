#!/usr/bin/env python3
"""
Pre-compute weekly skill trend data from jobs_database.db.

For each ESCO skill label × week:
  - mention_count: how many times the skill appears
  - offer_count: how many distinct offers mention it

Also stores total offers per week for % calculation.

Output: skill_trends.db (SQLite — fast indexed lookups)
"""

import json
import sqlite3
from collections import defaultdict

JOBS_DB = "jobs_database.db"
OUTPUT_DB = "skill_trends.db"


def main():
    print("Creating output database…", flush=True)
    out = sqlite3.connect(OUTPUT_DB)
    oc = out.cursor()
    oc.execute("DROP TABLE IF EXISTS period_totals")
    oc.execute("DROP TABLE IF EXISTS skill_period")
    oc.execute("DROP TABLE IF EXISTS skill_labels")
    oc.execute("""
        CREATE TABLE period_totals (
            period TEXT PRIMARY KEY,
            total_offers INTEGER
        )
    """)
    oc.execute("""
        CREATE TABLE skill_period (
            skill_id INTEGER,
            period TEXT,
            mention_count INTEGER,
            offer_count INTEGER,
            PRIMARY KEY (skill_id, period)
        )
    """)
    oc.execute("""
        CREATE TABLE skill_labels (
            skill_id INTEGER PRIMARY KEY,
            label TEXT UNIQUE,
            total_mentions INTEGER,
            total_offers INTEGER
        )
    """)
    out.commit()

    print("Scanning jobs_database.db…", flush=True)
    conn = sqlite3.connect(JOBS_DB)
    c = conn.cursor()
    c.execute("""
        SELECT strftime('%Y-W%W', posted_date) as week, skills_esco_contextual
        FROM job_ads
        WHERE skills_esco_contextual IS NOT NULL
          AND skills_esco_contextual != ''
          AND posted_date IS NOT NULL
    """)

    period_totals = defaultdict(int)
    skill_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    processed = 0
    for period, raw in c:
        try:
            items = json.loads(raw)
        except Exception:
            continue

        period_totals[period] += 1
        seen_skills = set()

        for item in items:
            esco = item.get("esco", "").strip()
            if not esco:
                continue
            skill_data[esco][period][0] += 1
            if esco not in seen_skills:
                skill_data[esco][period][1] += 1
                seen_skills.add(esco)

        processed += 1
        if processed % 100_000 == 0:
            print(f"  {processed:,} offers…", flush=True)

    conn.close()
    print(f"  Done: {processed:,} offers, {len(skill_data):,} unique skills, "
          f"{len(period_totals)} weeks", flush=True)

    oc.executemany(
        "INSERT INTO period_totals VALUES (?, ?)",
        sorted(period_totals.items()),
    )

    print("Writing to skill_trends.db…", flush=True)
    skill_id = 0
    batch_labels = []
    batch_period = []

    for label in sorted(skill_data.keys()):
        periods = skill_data[label]
        total_m = sum(v[0] for v in periods.values())
        total_o = sum(v[1] for v in periods.values())
        batch_labels.append((skill_id, label, total_m, total_o))

        for period, (mentions, offers) in sorted(periods.items()):
            batch_period.append((skill_id, period, mentions, offers))

        skill_id += 1

        if len(batch_labels) % 5000 == 0:
            oc.executemany("INSERT INTO skill_labels VALUES (?, ?, ?, ?)", batch_labels)
            oc.executemany("INSERT INTO skill_period VALUES (?, ?, ?, ?)", batch_period)
            batch_labels.clear()
            batch_period.clear()

    if batch_labels:
        oc.executemany("INSERT INTO skill_labels VALUES (?, ?, ?, ?)", batch_labels)
        oc.executemany("INSERT INTO skill_period VALUES (?, ?, ?, ?)", batch_period)

    oc.execute("CREATE INDEX idx_label ON skill_labels(label)")
    oc.execute("CREATE INDEX idx_label_lower ON skill_labels(label COLLATE NOCASE)")
    oc.execute("CREATE INDEX idx_total_mentions ON skill_labels(total_mentions DESC)")
    oc.execute("CREATE INDEX idx_skill_period ON skill_period(skill_id)")

    out.commit()
    out.close()

    print(f"\nSaved → {OUTPUT_DB}", flush=True)
    print(f"  {skill_id:,} skills, {len(period_totals)} weeks", flush=True)


if __name__ == "__main__":
    main()
