#!/usr/bin/env python3
"""
Save remaining embeddings from batch_output_remaining.jsonl to database.
"""

import os
import sqlite3
import numpy as np
import json
import pandas as pd
from typing import List

SQLITE_DB = "jobs_database.db"
BATCH_OUTPUT_FILE = "batch_output_remaining.jsonl"
BATCH_SIZE = 128


def get_remaining_titles(db_path: str) -> List[str]:
    """Get job titles that don't have embeddings yet, in correct order."""
    print("Getting titles that need embeddings...", flush=True)
    conn = sqlite3.connect(db_path)
    
    # Get all unique titles in order
    query_all = "SELECT DISTINCT title FROM job_ads WHERE title IS NOT NULL ORDER BY title"
    all_titles = pd.read_sql_query(query_all, conn)['title'].tolist()
    
    # Get titles that already have embeddings
    query_existing = "SELECT job_title FROM job_title_embeddings"
    existing_titles = set(pd.read_sql_query(query_existing, conn)['job_title'].tolist())
    
    conn.close()
    
    # Get titles without embeddings (preserving order)
    remaining_titles = [t for t in all_titles if t not in existing_titles]
    
    print(f"Total unique titles: {len(all_titles):,}", flush=True)
    print(f"Already have embeddings: {len(existing_titles):,}", flush=True)
    print(f"Remaining to process: {len(remaining_titles):,}", flush=True)
    
    return remaining_titles


def save_embeddings_from_file(results_file: str, titles: List[str], db_path: str) -> int:
    """Parse batch results and save embeddings with proper locking."""
    print(f"\nProcessing {results_file}...", flush=True)
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!", flush=True)
        return 0
    
    file_size_mb = os.path.getsize(results_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB", flush=True)
    
    # Open database with extended timeout and WAL mode
    conn = sqlite3.connect(db_path, timeout=60.0, isolation_level='DEFERRED')
    
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=10000")
    
    cursor = conn.cursor()
    
    embeddings_saved = 0
    batch_inserts = []
    
    print("Parsing embeddings...", flush=True)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                
                # Check if request was successful
                if result.get('response', {}).get('status_code') != 200:
                    print(f"Warning: Request {result.get('custom_id')} failed", flush=True)
                    continue
                
                # Extract embeddings
                body = result['response']['body']
                data = body['data']
                
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
                    
                    # Insert in batches of 500
                    if len(batch_inserts) >= 500:
                        try:
                            cursor.executemany(
                                "INSERT OR REPLACE INTO job_title_embeddings (job_title, embedding) VALUES (?, ?)",
                                batch_inserts
                            )
                            conn.commit()
                            batch_inserts = []
                            print(f"  ✓ Saved {embeddings_saved} embeddings so far...", flush=True)
                        except sqlite3.OperationalError as e:
                            print(f"  Warning: {e}. Retrying...", flush=True)
                            import time
                            time.sleep(1)
                            conn.commit()
                            batch_inserts = []
                
                if line_num % 100 == 0:
                    print(f"  Processed {line_num} result lines...", flush=True)
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", flush=True)
                continue
    
    # Insert remaining
    if batch_inserts:
        try:
            cursor.executemany(
                "INSERT OR REPLACE INTO job_title_embeddings (job_title, embedding) VALUES (?, ?)",
                batch_inserts
            )
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Warning during final commit: {e}", flush=True)
    
    conn.close()
    print(f"\n✓ Successfully saved {embeddings_saved} new embeddings", flush=True)
    
    return embeddings_saved


def verify_final_state(db_path: str) -> None:
    """Verify the final state of embeddings."""
    print("\n" + "="*60, flush=True)
    print("FINAL DATABASE STATE", flush=True)
    print("="*60, flush=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count embeddings
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    total_embeddings = cursor.fetchone()[0]
    
    # Count unique titles
    cursor.execute("SELECT COUNT(DISTINCT title) FROM job_ads WHERE title IS NOT NULL")
    unique_titles = cursor.fetchone()[0]
    
    # Count job ads
    cursor.execute("SELECT COUNT(*) FROM job_ads")
    total_ads = cursor.fetchone()[0]
    
    conn.close()
    
    coverage = (total_embeddings / unique_titles * 100) if unique_titles > 0 else 0
    
    print(f"Total job ads: {total_ads:,}", flush=True)
    print(f"Unique job titles: {unique_titles:,}", flush=True)
    print(f"Embeddings generated: {total_embeddings:,}", flush=True)
    print(f"Coverage: {coverage:.1f}%", flush=True)
    print("="*60, flush=True)


def main():
    """Main execution function."""
    print("="*60, flush=True)
    print("SAVING REMAINING EMBEDDINGS TO DATABASE", flush=True)
    print("="*60, flush=True)
    
    # Check if output file exists
    if not os.path.exists(BATCH_OUTPUT_FILE):
        print(f"\nError: {BATCH_OUTPUT_FILE} not found!", flush=True)
        return
    
    try:
        # Get titles that need embeddings
        remaining_titles = get_remaining_titles(SQLITE_DB)
        
        if len(remaining_titles) == 0:
            print("\n✓ All titles already have embeddings!", flush=True)
            verify_final_state(SQLITE_DB)
            return
        
        # Save embeddings
        embeddings_saved = save_embeddings_from_file(
            BATCH_OUTPUT_FILE, 
            remaining_titles, 
            SQLITE_DB
        )
        
        # Verify final state
        verify_final_state(SQLITE_DB)
        
        if embeddings_saved > 0:
            print(f"\n✓ Successfully saved {embeddings_saved} new embeddings!", flush=True)
        else:
            print("\n⚠ No new embeddings were saved", flush=True)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
