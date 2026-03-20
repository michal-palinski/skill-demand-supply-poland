#!/usr/bin/env python3
"""Efficient title cleaning using ID-based pagination."""

import sqlite3
import re
from tqdm import tqdm
import time

def clean_title(title):
    """Clean job title according to rules."""
    if not title:
        return ""
    t = title
    # Remove brackets
    t = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', t)
    # Remove | and after
    t = t.split('|')[0]
    # Remove "with" and after
    t = re.sub(r'\s+with\s+.*', '', t, flags=re.IGNORECASE)
    # Handle hyphens
    if t.count(' - ') >= 2:
        parts = t.split(' - ')
        t = parts[1] if len(parts) >= 3 else t
    elif t.count(' - ') == 1:
        parts = t.split(' - ')
        if len(parts) == 2:
            suffix_lower = parts[1].strip().lower()
            remove_keywords = ['team', 'category', 'department', 'remote', 'hybrid', 'office']
            if any(suffix_lower.startswith(kw) for kw in remove_keywords):
                t = parts[0]
    # Remove hashtags
    t = re.sub(r'#\w+', '', t)
    # Clean spaces
    t = re.sub(r'\s+', ' ', t).strip(' -.,;:')
    return t or title


def clean_job_ads_table(conn):
    """Clean job_ads table using efficient ID-based pagination."""
    print("\n" + "="*60)
    print("CLEANING job_ads TABLE")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Get total count and ID range
    cursor.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM job_ads")
    total, min_id, max_id = cursor.fetchone()
    print(f"Total rows: {total:,}")
    print(f"ID range: {min_id} to {max_id}")
    
    batch_size = 1000
    changed = 0
    start_time = time.time()
    
    # Use ID-based pagination
    current_id = min_id
    
    with tqdm(total=total, desc="Processing", unit="rows") as pbar:
        while current_id <= max_id:
            # Fetch batch based on ID range
            cursor.execute(
                "SELECT id, title FROM job_ads WHERE id >= ? AND id < ? AND title IS NOT NULL",
                (current_id, current_id + batch_size)
            )
            batch = cursor.fetchall()
            
            if not batch:
                current_id += batch_size
                continue
            
            # Clean and update
            updates = []
            for row_id, title in batch:
                cleaned = clean_title(title)
                updates.append((cleaned, row_id))
                if cleaned != title:
                    changed += 1
            
            cursor.executemany(
                "UPDATE job_ads SET job_title_clean = ? WHERE id = ?",
                updates
            )
            conn.commit()
            
            pbar.update(len(batch))
            current_id += batch_size
    
    elapsed = time.time() - start_time
    print(f"\n✓ Processed {total:,} rows in {elapsed:.1f}s")
    print(f"✓ Modified {changed:,} titles ({changed/total*100:.1f}%)")


def clean_embeddings_table(conn):
    """Clean job_title_embeddings table using efficient ID-based pagination."""
    print("\n" + "="*60)
    print("CLEANING job_title_embeddings TABLE")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Get total count and ID range
    cursor.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM job_title_embeddings")
    total, min_id, max_id = cursor.fetchone()
    print(f"Total rows: {total:,}")
    print(f"ID range: {min_id} to {max_id}")
    
    batch_size = 1000
    changed = 0
    start_time = time.time()
    
    # Use ID-based pagination
    current_id = min_id
    
    with tqdm(total=total, desc="Processing", unit="rows") as pbar:
        while current_id <= max_id:
            # Fetch batch based on ID range
            cursor.execute(
                "SELECT id, job_title FROM job_title_embeddings WHERE id >= ? AND id < ?",
                (current_id, current_id + batch_size)
            )
            batch = cursor.fetchall()
            
            if not batch:
                current_id += batch_size
                continue
            
            # Clean and update
            updates = []
            for row_id, title in batch:
                cleaned = clean_title(title)
                updates.append((cleaned, row_id))
                if cleaned != title:
                    changed += 1
            
            cursor.executemany(
                "UPDATE job_title_embeddings SET job_title_clean = ? WHERE id = ?",
                updates
            )
            conn.commit()
            
            pbar.update(len(batch))
            current_id += batch_size
    
    elapsed = time.time() - start_time
    print(f"\n✓ Processed {total:,} rows in {elapsed:.1f}s")
    print(f"✓ Modified {changed:,} titles ({changed/total*100:.1f}%)")


def show_examples(conn):
    """Show some examples of cleaned titles."""
    print("\n" + "="*60)
    print("EXAMPLES OF CLEANED TITLES")
    print("="*60)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, job_title_clean 
        FROM job_ads 
        WHERE title != job_title_clean 
        LIMIT 10
    """)
    
    for i, (orig, clean) in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}.")
        print(f"  Original: {orig}")
        print(f"  Cleaned:  {clean}")


def main():
    print("="*60)
    print("EFFICIENT JOB TITLE CLEANING")
    print("="*60)
    
    conn = sqlite3.connect("jobs_database.db", timeout=60.0)
    
    # Optimize for bulk updates
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = 10000")
    
    # Clean both tables
    clean_job_ads_table(conn)
    clean_embeddings_table(conn)
    
    # Create indexes
    print("\n" + "="*60)
    print("CREATING INDEXES")
    print("="*60)
    
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_ads_clean_title ON job_ads(job_title_clean)")
        print("✓ Created index on job_ads.job_title_clean")
    except Exception as e:
        print(f"⚠ Index on job_ads: {e}")
    
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_clean_title ON job_title_embeddings(job_title_clean)")
        print("✓ Created index on job_title_embeddings.job_title_clean")
    except Exception as e:
        print(f"⚠ Index on embeddings: {e}")
    
    conn.commit()
    
    # Show examples
    show_examples(conn)
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_ads WHERE job_title_clean IS NOT NULL")
    unique_clean_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_title_embeddings WHERE job_title_clean IS NOT NULL")
    unique_clean_emb = cursor.fetchone()[0]
    
    print(f"\nUnique cleaned titles in job_ads: {unique_clean_ads:,}")
    print(f"Unique cleaned titles in job_title_embeddings: {unique_clean_emb:,}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✓ CLEANING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
