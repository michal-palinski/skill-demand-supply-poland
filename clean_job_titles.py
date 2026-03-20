#!/usr/bin/env python3
"""
Script to create job_title_clean column in both job_ads and job_title_embeddings tables.
Cleans job titles by removing brackets, pipes, hashtags, "with" clauses, and handling hyphens.
"""

import sqlite3
import re
from typing import Optional


def clean_job_title(title: str) -> str:
    """
    Clean job title according to specified rules:
    1. Remove text in brackets (parentheses, square brackets, curly braces)
    2. Remove "|" and everything after
    3. Remove "with..." and everything after
    4. Handle hyphens:
       - If multiple hyphens, keep only text between first and second hyphen
       - If single hyphen with "Team" or similar suffixes, remove suffix
    5. Remove hashtags
    """
    if not title:
        return ""
    
    original = title
    
    # 1. Remove content in brackets (all types)
    # Remove parentheses: (...)
    title = re.sub(r'\([^)]*\)', '', title)
    # Remove square brackets: [...]
    title = re.sub(r'\[[^\]]*\]', '', title)
    # Remove curly braces: {...}
    title = re.sub(r'\{[^}]*\}', '', title)
    
    # 2. Remove "|" and everything after
    if '|' in title:
        title = title.split('|')[0]
    
    # 3. Remove "with" and everything after (case insensitive)
    title = re.sub(r'\s+with\s+.*', '', title, flags=re.IGNORECASE)
    
    # 4. Handle hyphens
    # Count hyphens
    hyphen_count = title.count(' - ')
    
    if hyphen_count >= 2:
        # Multiple hyphens: keep text between first and second hyphen
        # e.g., "#Program stażowy - Sprzedawca Samochodów - Porsche Bronowice Skoda"
        # → "Sprzedawca Samochodów"
        parts = title.split(' - ')
        if len(parts) >= 3:
            title = parts[1].strip()
    elif hyphen_count == 1:
        # Single hyphen: check if it's followed by common suffixes to remove
        parts = title.split(' - ')
        if len(parts) == 2:
            suffix = parts[1].strip().lower()
            # Common suffixes to remove (team names, locations, etc.)
            remove_suffixes = [
                'team', 'category', 'department', 'dept', 'division',
                'remote', 'hybrid', 'office', 'onsite'
            ]
            
            # Check if suffix starts with any of these keywords
            should_remove = any(suffix.startswith(keyword) for keyword in remove_suffixes)
            
            if should_remove:
                # Remove the suffix
                title = parts[0].strip()
            # Otherwise keep the full title with hyphen
    
    # 5. Remove hashtags
    # Remove hashtags at the beginning or anywhere in the text
    title = re.sub(r'#\w+', '', title)
    
    # Clean up extra whitespace
    title = re.sub(r'\s+', ' ', title)
    title = title.strip()
    
    # Remove leading/trailing punctuation that might be left
    title = title.strip(' -.,;:')
    
    return title if title else original


def add_clean_column_to_table(cursor, table_name: str, title_column: str) -> None:
    """Add job_title_clean column to a table if it doesn't exist."""
    # Check if column exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'job_title_clean' not in columns:
        print(f"Adding job_title_clean column to {table_name}...", flush=True)
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN job_title_clean TEXT")
    else:
        print(f"Column job_title_clean already exists in {table_name}", flush=True)


def clean_titles_in_table(cursor, table_name: str, title_column: str, id_column: str = 'id') -> int:
    """Clean titles in a specific table."""
    print(f"\nCleaning titles in {table_name}...", flush=True)
    
    # Get all titles
    cursor.execute(f"SELECT {id_column}, {title_column} FROM {table_name} WHERE {title_column} IS NOT NULL")
    rows = cursor.fetchall()
    
    total = len(rows)
    print(f"Processing {total:,} titles...", flush=True)
    
    cleaned_count = 0
    batch_updates = []
    
    processed = 0
    for row_id, title in rows:
        cleaned = clean_job_title(title)
        batch_updates.append((cleaned, row_id))
        processed += 1
        
        if cleaned != title:
            cleaned_count += 1
        
        # Batch update every 1000 rows
        if len(batch_updates) >= 1000:
            cursor.executemany(
                f"UPDATE {table_name} SET job_title_clean = ? WHERE {id_column} = ?",
                batch_updates
            )
            batch_updates = []
            print(f"  Processed {processed:,}/{total:,} titles...", flush=True)
    
    # Update remaining
    if batch_updates:
        cursor.executemany(
            f"UPDATE {table_name} SET job_title_clean = ? WHERE {id_column} = ?",
            batch_updates
        )
    
    print(f"✓ Cleaned {cleaned_count:,} titles (total processed: {total:,})", flush=True)
    
    return cleaned_count


def show_examples(cursor) -> None:
    """Show some examples of cleaned titles."""
    print("\n" + "="*80)
    print("EXAMPLES OF CLEANED TITLES")
    print("="*80)
    
    # Get some examples where cleaning made a difference
    cursor.execute("""
        SELECT title, job_title_clean 
        FROM job_ads 
        WHERE title != job_title_clean 
        AND title IS NOT NULL 
        AND job_title_clean IS NOT NULL
        LIMIT 20
    """)
    
    examples = cursor.fetchall()
    
    for original, cleaned in examples:
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 80)


def create_indexes(cursor) -> None:
    """Create indexes on job_title_clean columns for better query performance."""
    print("\nCreating indexes on job_title_clean columns...", flush=True)
    
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_ads_clean_title ON job_ads(job_title_clean)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_clean_title ON job_title_embeddings(job_title_clean)")
        print("✓ Indexes created", flush=True)
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}", flush=True)


def show_statistics(cursor) -> None:
    """Show statistics about cleaned titles."""
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    # Count in job_ads
    cursor.execute("SELECT COUNT(*) FROM job_ads WHERE title IS NOT NULL")
    total_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_ads WHERE title != job_title_clean")
    cleaned_ads = cursor.fetchone()[0]
    
    # Count in job_title_embeddings
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    total_embeddings = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings WHERE job_title != job_title_clean")
    cleaned_embeddings = cursor.fetchone()[0]
    
    # Count unique cleaned titles
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_ads WHERE job_title_clean IS NOT NULL")
    unique_clean_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT job_title_clean) FROM job_title_embeddings WHERE job_title_clean IS NOT NULL")
    unique_clean_embeddings = cursor.fetchone()[0]
    
    print(f"\nTable: job_ads")
    print(f"  Total titles: {total_ads:,}")
    print(f"  Titles modified: {cleaned_ads:,} ({cleaned_ads/total_ads*100:.1f}%)")
    print(f"  Unique cleaned titles: {unique_clean_ads:,}")
    
    print(f"\nTable: job_title_embeddings")
    print(f"  Total titles: {total_embeddings:,}")
    print(f"  Titles modified: {cleaned_embeddings:,} ({cleaned_embeddings/total_embeddings*100:.1f}%)")
    print(f"  Unique cleaned titles: {unique_clean_embeddings:,}")
    
    print("="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("JOB TITLE CLEANING SCRIPT")
    print("="*80)
    
    db_path = "jobs_database.db"
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cursor = conn.cursor()
        
        # Add columns to both tables
        print("\n1. Adding job_title_clean columns...")
        add_clean_column_to_table(cursor, "job_ads", "title")
        add_clean_column_to_table(cursor, "job_title_embeddings", "job_title")
        conn.commit()
        
        # Clean titles in job_ads table
        print("\n2. Cleaning titles in job_ads table...")
        cleaned_ads = clean_titles_in_table(cursor, "job_ads", "title", "id")
        conn.commit()
        
        # Clean titles in job_title_embeddings table
        print("\n3. Cleaning titles in job_title_embeddings table...")
        cleaned_embeddings = clean_titles_in_table(cursor, "job_title_embeddings", "job_title", "id")
        conn.commit()
        
        # Create indexes
        print("\n4. Creating indexes...")
        create_indexes(cursor)
        conn.commit()
        
        # Show examples
        show_examples(cursor)
        
        # Show statistics
        show_statistics(cursor)
        
        conn.close()
        
        print("\n" + "="*80)
        print("✓ JOB TITLE CLEANING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
