#!/usr/bin/env python3
"""
Script to process the remaining 86,030 job titles (titles 100,001 to 186,030).
This is a continuation of the initial batch processing.
"""

import os
import pandas as pd
import sqlite3
import numpy as np
from typing import List
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
SQLITE_DB = "jobs_database.db"
BATCH_INPUT_FILE = "batch_input_remaining.jsonl"
BATCH_OUTPUT_FILE = "batch_output_remaining.jsonl"
VOYAGE_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 128
START_INDEX = 100000  # Start from the 100,001st title

# VoyageAI API endpoints
VOYAGEAI_API_BASE = "https://api.voyageai.com/v1"


def get_remaining_titles(db_path: str) -> List[str]:
    """Get job titles that don't have embeddings yet."""
    conn = sqlite3.connect(db_path)
    
    # Get all unique titles
    query_all = "SELECT DISTINCT title FROM job_ads WHERE title IS NOT NULL ORDER BY title"
    all_titles = pd.read_sql_query(query_all, conn)['title'].tolist()
    
    # Get titles that already have embeddings
    query_existing = "SELECT job_title FROM job_title_embeddings"
    existing_titles = set(pd.read_sql_query(query_existing, conn)['job_title'].tolist())
    
    conn.close()
    
    # Get titles without embeddings
    remaining_titles = [t for t in all_titles if t not in existing_titles]
    
    print(f"Total unique titles: {len(all_titles):,}")
    print(f"Titles with embeddings: {len(existing_titles):,}")
    print(f"Remaining titles: {len(remaining_titles):,}")
    
    return remaining_titles


def create_batch_input_file(titles: List[str], output_file: str, batch_size: int = 128) -> None:
    """Create JSONL file for batch processing."""
    print(f"\nCreating batch input file: {output_file}", flush=True)
    print(f"Grouping {len(titles)} titles into requests of {batch_size} titles each", flush=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        request_id = 0
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            
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
            "corpus": "job titles (remaining)",
            "source": "pracujpl_2025",
            "batch": "2"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    batch_id = result['id']
    print(f"✓ Batch job created successfully! Batch ID: {batch_id}", flush=True)
    print(f"  Status: {result['status']}", flush=True)
    
    return batch_id


def monitor_batch_job(batch_id: str, api_key: str, poll_interval: int = 30) -> dict:
    """Monitor batch job until completion."""
    print(f"\nMonitoring batch job: {batch_id}", flush=True)
    print(f"Polling every {poll_interval} seconds...\n", flush=True)
    
    url = f"{VOYAGEAI_API_BASE}/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        status_info = response.json()
        
        status = status_info['status']
        request_counts = status_info.get('request_counts', {})
        
        total = request_counts.get('total', 0)
        completed = request_counts.get('completed', 0)
        failed = request_counts.get('failed', 0)
        
        progress_pct = (completed / total * 100) if total > 0 else 0
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status} | "
              f"Progress: {completed}/{total} ({progress_pct:.1f}%) | "
              f"Failed: {failed}", flush=True)
        
        if status in ['completed', 'failed', 'cancelled']:
            print(f"\n✓ Batch job finished with status: {status}", flush=True)
            return status_info
        
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


def save_embeddings_to_db(results_file: str, titles: List[str], db_path: str) -> int:
    """Parse batch results and save embeddings to database."""
    print(f"\nParsing and saving embeddings to database...", flush=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    embeddings_saved = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            result = json.loads(line)
            
            if result.get('response', {}).get('status_code') != 200:
                print(f"Warning: Request {result.get('custom_id')} failed", flush=True)
                continue
            
            body = result['response']['body']
            data = body['data']
            
            for item in data:
                embedding = item['embedding']
                index = item['index']
                
                custom_id = result['custom_id']
                request_num = int(custom_id.split('_')[1])
                
                global_index = request_num * BATCH_SIZE + index
                
                if global_index >= len(titles):
                    print(f"Warning: Index {global_index} out of range", flush=True)
                    continue
                
                job_title = titles[global_index]
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                cursor.execute(
                    "INSERT OR REPLACE INTO job_title_embeddings (job_title, embedding) VALUES (?, ?)",
                    (job_title, embedding_bytes)
                )
                
                embeddings_saved += 1
            
            if line_num % 100 == 0:
                print(f"  Processed {line_num} result lines, saved {embeddings_saved} embeddings...", flush=True)
                conn.commit()
    
    conn.commit()
    conn.close()
    print(f"✓ Saved {embeddings_saved} embeddings to database", flush=True)
    
    return embeddings_saved


def main():
    """Main execution function."""
    print("="*60, flush=True)
    print("PROCESSING REMAINING JOB TITLES", flush=True)
    print("="*60, flush=True)
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY not found in environment variables", flush=True)
        return
    
    try:
        # Get remaining titles
        remaining_titles = get_remaining_titles(SQLITE_DB)
        
        if len(remaining_titles) == 0:
            print("\n✓ All titles already have embeddings!", flush=True)
            return
        
        if len(remaining_titles) > 100000:
            print(f"\nWarning: {len(remaining_titles)} titles exceeds 100k limit", flush=True)
            print(f"Processing first 100,000 only", flush=True)
            remaining_titles = remaining_titles[:100000]
        
        # Create batch input
        create_batch_input_file(remaining_titles, BATCH_INPUT_FILE, BATCH_SIZE)
        
        # Upload file
        file_id = upload_batch_file(BATCH_INPUT_FILE, api_key)
        
        # Create batch job
        batch_id = create_batch_job(file_id, api_key)
        
        # Monitor progress
        final_status = monitor_batch_job(batch_id, api_key)
        
        if final_status['status'] != 'completed':
            print(f"Batch job did not complete successfully: {final_status['status']}", flush=True)
            return
        
        # Download results
        output_file_id = final_status['output_file_id']
        download_batch_results(output_file_id, api_key, BATCH_OUTPUT_FILE)
        
        # Save to database
        embeddings_saved = save_embeddings_to_db(BATCH_OUTPUT_FILE, remaining_titles, SQLITE_DB)
        
        # Print final statistics
        conn = sqlite3.connect(SQLITE_DB)
        total_embeddings = pd.read_sql_query("SELECT COUNT(*) as count FROM job_title_embeddings", conn).iloc[0]['count']
        unique_titles = pd.read_sql_query("SELECT COUNT(DISTINCT title) as count FROM job_ads", conn).iloc[0]['count']
        conn.close()
        
        print("\n" + "="*60, flush=True)
        print("FINAL STATISTICS", flush=True)
        print("="*60, flush=True)
        print(f"Total embeddings in database: {total_embeddings:,}", flush=True)
        print(f"Total unique titles: {unique_titles:,}", flush=True)
        print(f"Coverage: {total_embeddings/unique_titles*100:.1f}%", flush=True)
        print("="*60, flush=True)
        
        print("\n✓ Process completed successfully!", flush=True)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
