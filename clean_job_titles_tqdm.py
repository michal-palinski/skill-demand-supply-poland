#!/usr/bin/env python3
"""
Clean job titles with tqdm progress bars.
"""

import sqlite3
import re
from tqdm import tqdm

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
    print("JOB TITLE CLEANING WITH PROGRESS")
    print("="*60)
    
    conn = sqlite3.connect("jobs_database.db", timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cursor = conn.cursor()
    
    # Add columns if needed
    print("\nAdding columns...")
    try:
        cursor.execute("ALTER TABLE job_ads ADD COLUMN job_title_clean TEXT")
        print("  Added job_title_clean to job_ads")
    except:
        print("  Column job_title_clean already exists in job_ads")
    
    try:
        cursor.execute("ALTER TABLE job_title_embeddings ADD COLUMN job_title_clean TEXT")
        print("  Added job_title_clean to job_title_embeddings")
    except:
        print("  Column job_title_clean already exists in job_title_embeddings")
    
    conn.commit()
    
    # Clean job_ads table
    print("\n1. Cleaning job_ads table...")
    cursor.execute("SELECT id, title FROM job_ads WHERE title IS NOT NULL")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"   Total titles to process: {total:,}")
    
    batch_size = 10000
    changed_count = 0
    
    with tqdm(total=total, desc="job_ads", unit="titles") as pbar:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            updates = []
            
            for row_id, title in batch:
                cleaned = clean_job_title(title)
                updates.append((cleaned, row_id))
                if cleaned != title:
                    changed_count += 1
            
            cursor.executemany("UPDATE job_ads SET job_title_clean = ? WHERE id = ?", updates)
            conn.commit()
            pbar.update(len(batch))
    
    print(f"   ✓ Modified {changed_count:,} titles")
    
    # Clean job_title_embeddings table
    print("\n2. Cleaning job_title_embeddings table...")
    cursor.execute("SELECT id, job_title FROM job_title_embeddings")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"   Total titles to process: {total:,}")
    
    changed_count = 0
    
    with tqdm(total=total, desc="embeddings", unit="titles") as pbar:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            updates = []
            
            for row_id, title in batch:
                cleaned = clean_job_title(title)
                updates.append((cleaned, row_id))
                if cleaned != title:
                    changed_count += 1
            
            cursor.executemany("UPDATE job_title_embeddings SET job_title_clean = ? WHERE id = ?", updates)
            conn.commit()
            pbar.update(len(batch))
    
    print(f"   ✓ Modified {changed_count:,} titles")
    
    # Create indexes
    print("\n3. Creating indexes...")
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_ads_clean_title ON job_ads(job_title_clean)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_clean_title ON job_title_embeddings(job_title_clean)")
        print("   ✓ Indexes created")
    except Exception as e:
        print(f"   Warning: {e}")
    
    conn.commit()
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLES OF CLEANED TITLES:")
    print("="*60)
    cursor.execute("""
        SELECT title, job_title_clean 
        FROM job_ads 
        WHERE title != job_title_clean 
        LIMIT 15
    """)
    
    for orig, clean in cursor.fetchall():
        orig_short = (orig[:55] + '...') if len(orig) > 55 else orig
        clean_short = (clean[:55] + '...') if len(clean) > 55 else clean
        print(f"Original: {orig_short}")
        print(f"Cleaned:  {clean_short}")
        print("-" * 60)
    
    # Stats
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    
    cursor.execute("SELECT COUNT(*) FROM job_ads WHERE title IS NOT NULL")
    total_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_ads WHERE title != job_title_clean")
    changed_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_ads WHERE job_title_clean IS NOT NULL")
    unique_clean_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    total_emb = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings WHERE job_title != job_title_clean")
    changed_emb = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_title_embeddings WHERE job_title_clean IS NOT NULL")
    unique_clean_emb = cursor.fetchone()[0]
    
    print(f"\njob_ads table:")
    print(f"  Total titles: {total_ads:,}")
    print(f"  Modified: {changed_ads:,} ({changed_ads/total_ads*100:.1f}%)")
    print(f"  Unique cleaned titles: {unique_clean_ads:,}")
    
    print(f"\njob_title_embeddings table:")
    print(f"  Total titles: {total_emb:,}")
    print(f"  Modified: {changed_emb:,} ({changed_emb/total_emb*100:.1f}%)")
    print(f"  Unique cleaned titles: {unique_clean_emb:,}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✓ CLEANING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
