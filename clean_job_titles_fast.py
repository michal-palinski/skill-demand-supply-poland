#!/usr/bin/env python3
"""
Optimized script to clean job titles - faster version with better batching.
"""

import sqlite3
import re

def clean_job_title(title: str) -> str:
    """Clean job title according to rules."""
    if not title:
        return ""
    
    original = title
    
    # 1. Remove brackets
    title = re.sub(r'\([^)]*\)', '', title)
    title = re.sub(r'\[[^\]]*\]', '', title)
    title = re.sub(r'\{[^}]*\}', '', title)
    
    # 2. Remove "|" and after
    if '|' in title:
        title = title.split('|')[0]
    
    # 3. Remove "with" and after
    title = re.sub(r'\s+with\s+.*', '', title, flags=re.IGNORECASE)
    
    # 4. Handle hyphens
    hyphen_count = title.count(' - ')
    
    if hyphen_count >= 2:
        parts = title.split(' - ')
        if len(parts) >= 3:
            title = parts[1].strip()
    elif hyphen_count == 1:
        parts = title.split(' - ')
        if len(parts) == 2:
            suffix = parts[1].strip().lower()
            remove_suffixes = ['team', 'category', 'department', 'dept', 'division',
                             'remote', 'hybrid', 'office', 'onsite']
            should_remove = any(suffix.startswith(kw) for kw in remove_suffixes)
            if should_remove:
                title = parts[0].strip()
    
    # 5. Remove hashtags
    title = re.sub(r'#\w+', '', title)
    
    # Clean whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    title = title.strip(' -.,;:')
    
    return title if title else original


def main():
    print("="*60)
    print("FAST JOB TITLE CLEANING")
    print("="*60)
    
    conn = sqlite3.connect("jobs_database.db", timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cursor = conn.cursor()
    
    # Add columns if needed
    try:
        cursor.execute("ALTER TABLE job_ads ADD COLUMN job_title_clean TEXT")
    except:
        pass
    
    try:
        cursor.execute("ALTER TABLE job_title_embeddings ADD COLUMN job_title_clean TEXT")
    except:
        pass
    
    conn.commit()
    
    # Clean job_ads table
    print("\nCleaning job_ads...")
    cursor.execute("SELECT id, title FROM job_ads WHERE title IS NOT NULL")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Processing {total:,} titles")
    
    for i in range(0, len(rows), 5000):
        batch = rows[i:i+5000]
        updates = [(clean_job_title(title), row_id) for row_id, title in batch]
        cursor.executemany("UPDATE job_ads SET job_title_clean = ? WHERE id = ?", updates)
        conn.commit()
        print(f"  {i+len(batch):,}/{total:,}")
    
    # Clean job_title_embeddings
    print("\nCleaning job_title_embeddings...")
    cursor.execute("SELECT id, job_title FROM job_title_embeddings")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"Processing {total:,} titles")
    
    for i in range(0, len(rows), 5000):
        batch = rows[i:i+5000]
        updates = [(clean_job_title(title), row_id) for row_id, title in batch]
        cursor.executemany("UPDATE job_title_embeddings SET job_title_clean = ? WHERE id = ?", updates)
        conn.commit()
        print(f"  {i+len(batch):,}/{total:,}")
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLES:")
    cursor.execute("""
        SELECT title, job_title_clean 
        FROM job_ads 
        WHERE title != job_title_clean 
        LIMIT 10
    """)
    for orig, clean in cursor.fetchall():
        print(f"{orig[:60]:60s} → {clean[:60]}")
    
    # Stats
    cursor.execute("SELECT COUNT(*) FROM job_ads WHERE title != job_title_clean")
    changed = cursor.fetchone()[0]
    print(f"\n✓ Modified {changed:,} titles in job_ads")
    
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings WHERE job_title != job_title_clean")
    changed = cursor.fetchone()[0]
    print(f"✓ Modified {changed:,} titles in job_title_embeddings")
    
    conn.close()
    print("\n✓ DONE!")


if __name__ == "__main__":
    main()
