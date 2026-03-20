#!/usr/bin/env python3
"""
Take random 1000 job offers, split requirements & responsibilities
into individual items, clean them, and generate embeddings.
"""

import sqlite3
import re
import numpy as np
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import voyageai

load_dotenv()

DB_PATH = "jobs_database.db"
OUTPUT_DB = "req_resp_embeddings.db"
MODEL = "voyage-3-large"
DIMENSIONS = 1024
BATCH_SIZE = 128  # voyage API batch limit


def get_random_offers(n=1000):
    """Get n random offers with non-empty requirements and responsibilities."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, title, requirements, responsibilities
        FROM job_ads
        WHERE requirements IS NOT NULL AND requirements != ''
          AND responsibilities IS NOT NULL AND responsibilities != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,))
    rows = c.fetchall()
    conn.close()
    print(f"Fetched {len(rows)} random offers")
    return rows


def clean_item(text):
    """Clean a single requirement/responsibility item."""
    # Strip whitespace
    text = text.strip()
    # Remove leading bullets: •, -, *, >, ●, ○, ■, □, ▪, ▸, ‣, ◦
    text = re.sub(r'^[\s]*[•\-\*>●○■□▪▸‣◦►➤➢✓✔☑→⇒]+[\s]*', '', text)
    # Remove leading numbers/letters with dots/parens: "1.", "1)", "a.", "a)"
    text = re.sub(r'^[\s]*[\d]+[.\)]\s*', '', text)
    text = re.sub(r'^[\s]*[a-zA-Z][.\)]\s*', '', text)
    # Remove semicolons at end
    text = text.rstrip(';').rstrip(',').strip()
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_and_clean(raw_text):
    """Split a multi-item text into individual cleaned items."""
    if not raw_text:
        return []

    # Split by newlines first
    lines = raw_text.split('\n')

    # If only one line, try splitting by bullet characters
    if len(lines) == 1:
        lines = re.split(r'[•●○■□▪]', raw_text)

    # If still one chunk, try splitting by semicolons
    if len(lines) == 1:
        lines = raw_text.split(';')

    items = []
    for line in lines:
        cleaned = clean_item(line)
        # Skip empty or very short items (noise)
        if len(cleaned) >= 5:
            items.append(cleaned)

    return items


def create_output_db():
    """Create the output database with tables for items and embeddings."""
    conn = sqlite3.connect(OUTPUT_DB)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS items")
    c.execute("DROP TABLE IF EXISTS embeddings")
    c.execute("DROP TABLE IF EXISTS sample_offers")

    c.execute("""
        CREATE TABLE sample_offers (
            job_id INTEGER PRIMARY KEY,
            title TEXT,
            requirements_raw TEXT,
            responsibilities_raw TEXT
        )
    """)

    c.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            type TEXT NOT NULL,  -- 'requirement' or 'responsibility'
            text_clean TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES sample_offers(job_id)
        )
    """)

    c.execute("""
        CREATE TABLE embeddings (
            item_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (item_id) REFERENCES items(id)
        )
    """)

    conn.commit()
    return conn


def save_offers_and_items(conn, offers):
    """Parse offers, save raw + cleaned items. Return list of (item_id, text)."""
    c = conn.cursor()

    total_req = 0
    total_resp = 0

    for job_id, title, req_raw, resp_raw in tqdm(offers, desc="Parsing items"):
        c.execute(
            "INSERT INTO sample_offers VALUES (?, ?, ?, ?)",
            (job_id, title, req_raw, resp_raw)
        )

        # Requirements
        req_items = split_and_clean(req_raw)
        for item in req_items:
            c.execute(
                "INSERT INTO items (job_id, type, text_clean) VALUES (?, 'requirement', ?)",
                (job_id, item)
            )
            total_req += 1

        # Responsibilities
        resp_items = split_and_clean(resp_raw)
        for item in resp_items:
            c.execute(
                "INSERT INTO items (job_id, type, text_clean) VALUES (?, 'responsibility', ?)",
                (job_id, item)
            )
            total_resp += 1

    conn.commit()

    print(f"Total requirements:     {total_req:,}")
    print(f"Total responsibilities:  {total_resp:,}")
    print(f"Total items:            {total_req + total_resp:,}")

    # Fetch all items for embedding
    c.execute("SELECT id, text_clean FROM items ORDER BY id")
    all_items = c.fetchall()
    return all_items


def generate_embeddings(items, conn):
    """Generate embeddings in batches and save to DB."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not found in .env")

    vo = voyageai.Client(api_key=api_key)
    c = conn.cursor()

    total = len(items)
    print(f"\nGenerating embeddings for {total:,} items using {MODEL} ({DIMENSIONS}d)...")

    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
        batch = items[i:i + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        result = vo.embed(
            texts=texts,
            model=MODEL,
            input_type="document",
            output_dimension=DIMENSIONS
        )

        rows = []
        for item_id, emb in zip(ids, result.embeddings):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            rows.append((item_id, emb_blob))

        c.executemany("INSERT INTO embeddings (item_id, embedding) VALUES (?, ?)", rows)
        conn.commit()

    print("Embeddings saved!")


def print_summary(conn):
    """Print summary stats."""
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM sample_offers")
    n_offers = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM items WHERE type = 'requirement'")
    n_req = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM items WHERE type = 'responsibility'")
    n_resp = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM embeddings")
    n_emb = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT text_clean) FROM items")
    n_unique = c.fetchone()[0]

    # Show some examples
    c.execute("""
        SELECT i.type, i.text_clean
        FROM items i
        ORDER BY RANDOM() LIMIT 10
    """)
    examples = c.fetchall()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Sample offers:          {n_offers:,}")
    print(f"Requirements:           {n_req:,}")
    print(f"Responsibilities:       {n_resp:,}")
    print(f"Total items:            {n_req + n_resp:,}")
    print(f"Unique items:           {n_unique:,}")
    print(f"Embeddings:             {n_emb:,}")
    print(f"Model:                  {MODEL}")
    print(f"Dimensions:             {DIMENSIONS}")
    print(f"Output DB:              {OUTPUT_DB}")
    print(f"\nExamples:")
    for typ, text in examples:
        tag = "REQ " if typ == "requirement" else "RESP"
        print(f"  [{tag}] {text[:90]}")
    print(f"{'='*60}")


def main():
    print(f"Model: {MODEL}, Dimensions: {DIMENSIONS}")
    print(f"Output: {OUTPUT_DB}\n")

    # 1. Get random offers
    offers = get_random_offers(1000)

    # 2. Create output DB
    conn = create_output_db()

    # 3. Parse and save items
    all_items = save_offers_and_items(conn, offers)

    # 4. Generate embeddings
    generate_embeddings(all_items, conn)

    # 5. Summary
    print_summary(conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
