#!/usr/bin/env python3
"""
Script to reprocess and save embeddings from batch_output.jsonl to database.
This handles any database locking issues and ensures clean saves.
"""

import os
import sqlite3
import numpy as np
import json
from typing import List

SQLITE_DB = "jobs_database.db"
BATCH_OUTPUT_FILE = "batch_output.jsonl"
BATCH_SIZE = 128


def unlock_and_prepare_db(db_path: str) -> None:
    """Unlock database and prepare for new embeddings."""
    print("Preparing database...", flush=True)
    
    # Close any existing connections
    conn = sqlite3.connect(db_path)
    
    # Set pragmas for better performance and to avoid locks
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=10000")
    
    # Drop and recreate embeddings table
    print("Recreating job_title_embeddings table...", flush=True)
    conn.execute("DROP TABLE IF EXISTS job_title_embeddings")
    conn.execute("""
        CREATE TABLE job_title_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX idx_job_title ON job_title_embeddings(job_title)")
    
    conn.commit()
    conn.close()
    print("✓ Database prepared", flush=True)


def get_unique_job_titles(db_path: str) -> List[str]:
    """Get unique job titles in order."""
    print("\nGetting unique job titles from job_ads...", flush=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT title FROM job_ads WHERE title IS NOT NULL ORDER BY title")
    titles = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    print(f"Found {len(titles)} unique titles", flush=True)
    
    # Return first 100k (what was processed)
    return titles[:100000]


def parse_and_save_embeddings(results_file: str, titles: List[str], db_path: str) -> int:
    """Parse batch results and save embeddings to database with proper transaction handling."""
    print(f"\nParsing embeddings from {results_file}...", flush=True)
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!", flush=True)
        return 0
    
    # Connect with optimized settings
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    cursor = conn.cursor()
    
    embeddings_saved = 0
    batch_inserts = []
    
    print("Processing batch output file...", flush=True)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            result = json.loads(line)
            
            # Check if request was successful
            if result.get('response', {}).get('status_code') != 200:
                print(f"Warning: Request {result.get('custom_id')} failed", flush=True)
                continue
            
            # Extract embeddings from response
            body = result['response']['body']
            data = body['data']
            
            # Each request contains multiple titles and their embeddings
            for item in data:
                embedding = item['embedding']
                index = item['index']
                
                # Calculate which title this embedding belongs to
                custom_id = result['custom_id']
                request_num = int(custom_id.split('_')[1])
                global_index = request_num * BATCH_SIZE + index
                
                if global_index >= len(titles):
                    print(f"Warning: Index {global_index} out of range (max: {len(titles)})", flush=True)
                    continue
                
                job_title = titles[global_index]
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                batch_inserts.append((job_title, embedding_bytes))
                embeddings_saved += 1
                
                # Insert in batches of 1000 for better performance
                if len(batch_inserts) >= 1000:
                    cursor.executemany(
                        "INSERT OR REPLACE INTO job_title_embeddings (job_title, embedding) VALUES (?, ?)",
                        batch_inserts
                    )
                    conn.commit()
                    batch_inserts = []
                    print(f"  Saved {embeddings_saved} embeddings so far...", flush=True)
            
            if line_num % 100 == 0:
                print(f"  Processed {line_num} result lines...", flush=True)
    
    # Insert remaining
    if batch_inserts:
        cursor.executemany(
            "INSERT OR REPLACE INTO job_title_embeddings (job_title, embedding) VALUES (?, ?)",
            batch_inserts
        )
        conn.commit()
    
    conn.close()
    print(f"\n✓ Successfully saved {embeddings_saved} embeddings to database", flush=True)
    
    return embeddings_saved


def verify_embeddings(db_path: str) -> None:
    """Verify saved embeddings."""
    print("\nVerifying embeddings...", flush=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count embeddings
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    count = cursor.fetchone()[0]
    print(f"Total embeddings in database: {count:,}", flush=True)
    
    # Check a sample embedding
    cursor.execute("SELECT job_title, embedding FROM job_title_embeddings LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        job_title, embedding_bytes = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        print(f"\nSample embedding:", flush=True)
        print(f"  Job title: {job_title}", flush=True)
        print(f"  Dimensions: {len(embedding)}", flush=True)
        print(f"  First 5 values: {embedding[:5]}", flush=True)
    
    # Get some stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total_ads,
            COUNT(DISTINCT title) as unique_titles
        FROM job_ads
    """)
    total_ads, unique_titles = cursor.fetchone()
    
    conn.close()
    
    print(f"\nDatabase statistics:", flush=True)
    print(f"  Total job ads: {total_ads:,}", flush=True)
    print(f"  Unique titles: {unique_titles:,}", flush=True)
    print(f"  Embeddings saved: {count:,}", flush=True)
    print(f"  Coverage: {count/unique_titles*100:.1f}%", flush=True)


def main():
    """Main execution function."""
    print("="*60, flush=True)
    print("REPROCESSING EMBEDDINGS TO DATABASE", flush=True)
    print("="*60, flush=True)
    
    # Check if output file exists
    if not os.path.exists(BATCH_OUTPUT_FILE):
        print(f"\nError: {BATCH_OUTPUT_FILE} not found!", flush=True)
        print("Make sure batch processing has completed first.", flush=True)
        return
    
    try:
        # Step 1: Unlock and prepare database
        unlock_and_prepare_db(SQLITE_DB)
        
        # Step 2: Get titles in correct order
        titles = get_unique_job_titles(SQLITE_DB)
        
        # Step 3: Parse and save embeddings
        embeddings_saved = parse_and_save_embeddings(BATCH_OUTPUT_FILE, titles, SQLITE_DB)
        
        if embeddings_saved == 0:
            print("\n✗ No embeddings were saved!", flush=True)
            return
        
        # Step 4: Verify
        verify_embeddings(SQLITE_DB)
        
        print("\n" + "="*60, flush=True)
        print("✓ PROCESS COMPLETED SUCCESSFULLY!", flush=True)
        print("="*60, flush=True)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
