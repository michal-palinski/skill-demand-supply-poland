#!/usr/bin/env python3
"""
Script to process pracujpl_2025.parquet file using VoyageAI Batch API:
1. Load parquet file to SQLite database
2. Extract unique job titles
3. Create batch input JSONL file
4. Upload to VoyageAI
5. Create batch job
6. Monitor progress
7. Download and process results
8. Save embeddings to database

Using Batch API provides 33% discount and better throughput!
"""

import os
import pandas as pd
import sqlite3
import numpy as np
from typing import List, Dict
import time
import json
from dotenv import load_dotenv
import requests
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
PARQUET_FILE = "jobads/data/pracujpl_2025.parquet"
SQLITE_DB = "jobs_database.db"
BATCH_INPUT_FILE = "batch_input.jsonl"
BATCH_OUTPUT_FILE = "batch_output.jsonl"
VOYAGE_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 128  # Number of titles per batch request (max 1000 per request)
MAX_INPUTS_PER_BATCH = 100000  # VoyageAI limit

# VoyageAI API endpoints
VOYAGEAI_API_BASE = "https://api.voyageai.com/v1"


def load_parquet_to_sqlite(parquet_path: str, db_path: str) -> None:
    """Load parquet file into SQLite database."""
    print(f"Loading parquet file: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} job ads with {len(df.columns)} columns", flush=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Save dataframe to SQLite
    print(f"Saving to SQLite database: {db_path}", flush=True)
    df.to_sql('job_ads', conn, if_exists='replace', index=False)
    
    # Create index on title for faster queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON job_ads(title)")
    
    conn.commit()
    conn.close()
    print("Data successfully saved to SQLite database", flush=True)


def get_unique_job_titles(db_path: str) -> List[str]:
    """Extract unique job titles from database."""
    print("\nExtracting unique job titles...", flush=True)
    conn = sqlite3.connect(db_path)
    
    query = "SELECT DISTINCT title FROM job_ads WHERE title IS NOT NULL ORDER BY title"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    unique_titles = df['title'].tolist()
    print(f"Found {len(unique_titles)} unique job titles", flush=True)
    
    return unique_titles


def create_embeddings_table(db_path: str) -> None:
    """Create table for storing embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop table if exists and create new one
    cursor.execute("DROP TABLE IF EXISTS job_title_embeddings")
    cursor.execute("""
        CREATE TABLE job_title_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on job_title
    cursor.execute("CREATE INDEX idx_job_title ON job_title_embeddings(job_title)")
    
    conn.commit()
    conn.close()
    print("Created job_title_embeddings table", flush=True)


def create_batch_input_file(titles: List[str], output_file: str, batch_size: int = 128) -> None:
    """Create JSONL file for batch processing."""
    print(f"\nCreating batch input file: {output_file}", flush=True)
    print(f"Grouping {len(titles)} titles into requests of {batch_size} titles each", flush=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        request_id = 0
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            
            # Create batch request object
            request_obj = {
                "custom_id": f"request_{request_id}",
                "body": {
                    "input": batch
                }
            }
            
            f.write(json.dumps(request_obj, ensure_ascii=False) + '\n')
            request_id += 1
    
    num_requests = request_id
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Created {num_requests} batch requests in {output_file} ({file_size_mb:.2f} MB)", flush=True)


def upload_batch_file(file_path: str, api_key: str) -> str:
    """Upload batch input file to VoyageAI."""
    print(f"\nUploading batch file to VoyageAI...", flush=True)
    
    url = f"{VOYAGEAI_API_BASE}/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with open(file_path, 'rb') as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"purpose": "batch"}
        
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        
    result = response.json()
    file_id = result['id']
    print(f"✓ File uploaded successfully! File ID: {file_id}", flush=True)
    print(f"  Filename: {result['filename']}, Size: {result['bytes']} bytes", flush=True)
    
    return file_id


def create_batch_job(input_file_id: str, api_key: str) -> str:
    """Create batch job for embeddings."""
    print(f"\nCreating batch job...", flush=True)
    
    url = f"{VOYAGEAI_API_BASE}/batches"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "endpoint": "/v1/embeddings",
        "completion_window": "12h",
        "request_params": {
            "model": VOYAGE_MODEL,
            "input_type": "document",
            "output_dimension": EMBEDDING_DIMENSIONS
        },
        "input_file_id": input_file_id,
        "metadata": {
            "corpus": "job titles",
            "source": "pracujpl_2025"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    batch_id = result['id']
    print(f"✓ Batch job created successfully! Batch ID: {batch_id}", flush=True)
    print(f"  Status: {result['status']}", flush=True)
    print(f"  Expected completion: {result.get('expected_completion_at', 'N/A')}", flush=True)
    
    return batch_id


def check_batch_status(batch_id: str, api_key: str) -> Dict:
    """Check status of batch job."""
    url = f"{VOYAGEAI_API_BASE}/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def monitor_batch_job(batch_id: str, api_key: str, poll_interval: int = 30) -> Dict:
    """Monitor batch job until completion."""
    print(f"\nMonitoring batch job: {batch_id}", flush=True)
    print("This may take up to 12 hours, but typically completes much faster.", flush=True)
    print(f"Polling every {poll_interval} seconds...\n", flush=True)
    
    while True:
        status_info = check_batch_status(batch_id, api_key)
        status = status_info['status']
        request_counts = status_info.get('request_counts', {})
        
        total = request_counts.get('total', 0)
        completed = request_counts.get('completed', 0)
        failed = request_counts.get('failed', 0)
        
        progress_pct = (completed / total * 100) if total > 0 else 0
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status} | "
              f"Progress: {completed}/{total} ({progress_pct:.1f}%) | "
              f"Failed: {failed}", flush=True)
        
        # Check if job is complete
        if status in ['completed', 'failed', 'cancelled']:
            print(f"\n✓ Batch job finished with status: {status}", flush=True)
            return status_info
        
        # Wait before next poll
        time.sleep(poll_interval)


def download_batch_results(file_id: str, api_key: str, output_file: str) -> None:
    """Download batch results file."""
    print(f"\nDownloading batch results...", flush=True)
    
    url = f"{VOYAGEAI_API_BASE}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with open(output_file, 'wb') as f:
        f.write(response.content)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Downloaded results to {output_file} ({file_size_mb:.2f} MB)", flush=True)


def parse_and_save_embeddings(results_file: str, db_path: str) -> None:
    """Parse batch results and save embeddings to database."""
    print(f"\nParsing and saving embeddings to database...", flush=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Parse JSONL results file
    embeddings_saved = 0
    
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
                
                # We need to map back to the original title
                # The custom_id tells us which request, the index tells us position within request
                custom_id = result['custom_id']
                request_num = int(custom_id.split('_')[1])
                
                # Calculate global title index
                global_index = request_num * BATCH_SIZE + index
                
                # Note: We'll need to maintain the title order
                # For now, we'll store with a placeholder and update later
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                # We'll use a temporary storage approach
                cursor.execute(
                    "INSERT INTO job_title_embeddings (id, job_title, embedding) VALUES (?, ?, ?)",
                    (global_index, f"_temp_{global_index}", embedding_bytes)
                )
                
                embeddings_saved += 1
            
            if line_num % 100 == 0:
                print(f"  Processed {line_num} result lines, saved {embeddings_saved} embeddings...", flush=True)
    
    conn.commit()
    print(f"✓ Saved {embeddings_saved} embeddings to database", flush=True)
    
    return conn, embeddings_saved


def map_embeddings_to_titles(conn: sqlite3.Connection, titles: List[str]) -> None:
    """Map saved embeddings back to their original titles."""
    print(f"\nMapping embeddings to job titles...", flush=True)
    
    cursor = conn.cursor()
    
    # Update each embedding with its correct title
    for idx, title in enumerate(titles):
        cursor.execute(
            "UPDATE job_title_embeddings SET job_title = ? WHERE id = ?",
            (title, idx)
        )
        
        if (idx + 1) % 10000 == 0:
            print(f"  Mapped {idx + 1}/{len(titles)} titles...", flush=True)
    
    conn.commit()
    conn.close()
    print(f"✓ All embeddings mapped to titles", flush=True)


def print_summary(db_path: str) -> None:
    """Print summary statistics."""
    conn = sqlite3.connect(db_path)
    
    # Count job ads
    job_ads_count = pd.read_sql_query("SELECT COUNT(*) as count FROM job_ads", conn).iloc[0]['count']
    
    # Count unique titles
    unique_titles_count = pd.read_sql_query("SELECT COUNT(DISTINCT title) as count FROM job_ads", conn).iloc[0]['count']
    
    # Count embeddings
    embeddings_count = pd.read_sql_query("SELECT COUNT(*) as count FROM job_title_embeddings", conn).iloc[0]['count']
    
    # Get sample embedding to check dimensions
    sample_embedding = pd.read_sql_query("SELECT embedding FROM job_title_embeddings LIMIT 1", conn).iloc[0]['embedding']
    embedding_array = np.frombuffer(sample_embedding, dtype=np.float32)
    
    conn.close()
    
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"Total job ads: {job_ads_count:,}", flush=True)
    print(f"Unique job titles: {unique_titles_count:,}", flush=True)
    print(f"Embeddings generated: {embeddings_count:,}", flush=True)
    print(f"Embedding dimensions: {len(embedding_array)}", flush=True)
    print(f"Model: {VOYAGE_MODEL}", flush=True)
    print(f"Database file: {db_path}", flush=True)
    print("="*60, flush=True)


def main():
    """Main execution function."""
    print("="*60, flush=True)
    print("JOB ADS PROCESSING WITH EMBEDDINGS (BATCH API)", flush=True)
    print("Using VoyageAI Batch API for 33% cost savings!", flush=True)
    print("="*60, flush=True)
    
    # Check if parquet file exists
    if not os.path.exists(PARQUET_FILE):
        print(f"Error: Parquet file not found: {PARQUET_FILE}", flush=True)
        return
    
    # Get VoyageAI API key from environment
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY not found in environment variables", flush=True)
        print("Please set it in .env file or export it", flush=True)
        return
    
    try:
        # Step 1: Load parquet to SQLite (skip if DB already exists)
        if not os.path.exists(SQLITE_DB):
            load_parquet_to_sqlite(PARQUET_FILE, SQLITE_DB)
        else:
            print(f"SQLite database already exists: {SQLITE_DB}", flush=True)
        
        # Step 2: Get unique job titles
        unique_titles = get_unique_job_titles(SQLITE_DB)
        
        if len(unique_titles) == 0:
            print("No job titles found!", flush=True)
            return
        
        if len(unique_titles) > MAX_INPUTS_PER_BATCH:
            print(f"Warning: {len(unique_titles)} titles exceeds batch limit of {MAX_INPUTS_PER_BATCH}", flush=True)
            print(f"Processing first {MAX_INPUTS_PER_BATCH} titles only", flush=True)
            unique_titles = unique_titles[:MAX_INPUTS_PER_BATCH]
        
        # Step 3: Create embeddings table
        create_embeddings_table(SQLITE_DB)
        
        # Step 4: Create batch input JSONL file
        create_batch_input_file(unique_titles, BATCH_INPUT_FILE, BATCH_SIZE)
        
        # Step 5: Upload batch file
        file_id = upload_batch_file(BATCH_INPUT_FILE, api_key)
        
        # Step 6: Create batch job
        batch_id = create_batch_job(file_id, api_key)
        
        # Step 7: Monitor batch job
        final_status = monitor_batch_job(batch_id, api_key)
        
        if final_status['status'] != 'completed':
            print(f"Batch job did not complete successfully: {final_status['status']}", flush=True)
            return
        
        # Step 8: Download results
        output_file_id = final_status['output_file_id']
        download_batch_results(output_file_id, api_key, BATCH_OUTPUT_FILE)
        
        # Step 9: Parse and save embeddings
        conn, embeddings_saved = parse_and_save_embeddings(BATCH_OUTPUT_FILE, SQLITE_DB)
        
        # Step 10: Map embeddings to titles
        map_embeddings_to_titles(conn, unique_titles)
        
        # Print summary
        print_summary(SQLITE_DB)
        
        print("\n✓ Process completed successfully!", flush=True)
        print(f"\nGenerated files:", flush=True)
        print(f"  - SQLite database: {SQLITE_DB}", flush=True)
        print(f"  - Batch input: {BATCH_INPUT_FILE}", flush=True)
        print(f"  - Batch output: {BATCH_OUTPUT_FILE}", flush=True)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
