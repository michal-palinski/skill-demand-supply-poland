#!/usr/bin/env python3
"""
Generate embeddings for occupation names (nazwa) from kzis_occupations_descriptions.db
using VoyageAI Batch API.
"""

import os
import sqlite3
import numpy as np
import json
import time
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Configuration
SOURCE_DB = "kzis_occupations_descriptions.db"
BATCH_INPUT_FILE = "occupation_embeddings_input.jsonl"
BATCH_OUTPUT_FILE = "occupation_embeddings_output.jsonl"
VOYAGE_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 128  # Titles per request

VOYAGEAI_API_BASE = "https://api.voyageai.com/v1"


def get_occupation_names(db_path: str) -> List[tuple]:
    """Get all occupation names from the database."""
    print(f"Reading occupation names from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all names with their IDs
    cursor.execute("SELECT id, nazwa FROM opisy_zawodow ORDER BY id")
    occupations = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(occupations)} occupations")
    return occupations


def create_embeddings_table(db_path: str) -> None:
    """Create table for storing embeddings."""
    print("\nCreating embeddings table...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS occupation_embeddings")
    cursor.execute("""
        CREATE TABLE occupation_embeddings (
            id INTEGER PRIMARY KEY,
            occupation_id INTEGER NOT NULL UNIQUE,
            nazwa TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (occupation_id) REFERENCES opisy_zawodow(id)
        )
    """)
    
    cursor.execute("CREATE INDEX idx_occupation_id ON occupation_embeddings(occupation_id)")
    cursor.execute("CREATE INDEX idx_nazwa ON occupation_embeddings(nazwa)")
    
    conn.commit()
    conn.close()
    print("✓ Created occupation_embeddings table")


def create_batch_input_file(occupations: List[tuple], output_file: str) -> None:
    """Create JSONL file for batch processing."""
    print(f"\nCreating batch input file: {output_file}")
    
    # Extract just the names for batching
    names = [name for _, name in occupations]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        request_id = 0
        for i in range(0, len(names), BATCH_SIZE):
            batch = names[i:i + BATCH_SIZE]
            
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
    print(f"✓ Created {num_requests} batch requests ({file_size_mb:.2f} MB)")


def upload_batch_file(file_path: str, api_key: str) -> str:
    """Upload batch input file to VoyageAI."""
    print(f"\nUploading to VoyageAI...")
    
    url = f"{VOYAGEAI_API_BASE}/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with open(file_path, 'rb') as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"purpose": "batch"}
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        
    result = response.json()
    file_id = result['id']
    print(f"✓ File uploaded! ID: {file_id}")
    return file_id


def create_batch_job(input_file_id: str, api_key: str) -> str:
    """Create batch job for embeddings."""
    print(f"\nCreating batch job...")
    
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
            "source": "kzis_occupations_descriptions",
            "table": "opisy_zawodow",
            "column": "nazwa"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    batch_id = result['id']
    print(f"✓ Batch job created! ID: {batch_id}")
    print(f"  Status: {result['status']}")
    return batch_id


def monitor_batch_job(batch_id: str, api_key: str, poll_interval: int = 10) -> dict:
    """Monitor batch job until completion."""
    print(f"\nMonitoring batch job: {batch_id}")
    print(f"Polling every {poll_interval} seconds...\n")
    
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
        
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status:12s} | "
              f"Progress: {completed:3d}/{total:3d} ({progress_pct:5.1f}%) | "
              f"Failed: {failed}", flush=True)
        
        if status in ['completed', 'failed', 'cancelled']:
            print(f"\n✓ Batch job finished: {status}")
            return status_info
        
        time.sleep(poll_interval)


def download_batch_results(file_id: str, api_key: str, output_file: str) -> None:
    """Download batch results file."""
    print(f"\nDownloading results...")
    
    url = f"{VOYAGEAI_API_BASE}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with open(output_file, 'wb') as f:
        f.write(response.content)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Downloaded {output_file} ({file_size_mb:.2f} MB)")


def save_embeddings_to_db(results_file: str, occupations: List[tuple], db_path: str) -> int:
    """Parse results and save embeddings to database."""
    print(f"\nSaving embeddings to database...")
    
    # Create mapping of index to occupation info
    index_to_occupation = {i: (occ_id, name) for i, (occ_id, name) in enumerate(occupations)}
    
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    
    embeddings_saved = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            result = json.loads(line)
            
            if result.get('response', {}).get('status_code') != 200:
                print(f"Warning: Request {result.get('custom_id')} failed")
                continue
            
            body = result['response']['body']
            data = body['data']
            
            for item in data:
                embedding = item['embedding']
                index = item['index']
                
                # Calculate global index
                custom_id = result['custom_id']
                request_num = int(custom_id.split('_')[1])
                global_index = request_num * BATCH_SIZE + index
                
                if global_index not in index_to_occupation:
                    continue
                
                occ_id, name = index_to_occupation[global_index]
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                cursor.execute(
                    "INSERT INTO occupation_embeddings (occupation_id, nazwa, embedding) VALUES (?, ?, ?)",
                    (occ_id, name, embedding_bytes)
                )
                
                embeddings_saved += 1
    
    conn.commit()
    conn.close()
    
    print(f"✓ Saved {embeddings_saved} embeddings")
    return embeddings_saved


def show_examples(db_path: str) -> None:
    """Show some examples."""
    print("\n" + "="*60)
    print("SAMPLE OCCUPATIONS WITH EMBEDDINGS")
    print("="*60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT o.kod, e.nazwa, LENGTH(e.embedding)/4 as dimensions
        FROM occupation_embeddings e
        JOIN opisy_zawodow o ON e.occupation_id = o.id
        LIMIT 10
    """)
    
    for kod, nazwa, dims in cursor.fetchall():
        print(f"{kod}: {nazwa} (dims: {dims})")
    
    conn.close()


def main():
    """Main execution function."""
    print("="*60)
    print("OCCUPATION NAME EMBEDDINGS - BATCH API")
    print("="*60)
    
    # Get API key
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY not found")
        return
    
    try:
        # Step 1: Get occupation names
        occupations = get_occupation_names(SOURCE_DB)
        
        # Step 2: Create embeddings table
        create_embeddings_table(SOURCE_DB)
        
        # Step 3: Create batch input file
        create_batch_input_file(occupations, BATCH_INPUT_FILE)
        
        # Step 4: Upload file
        file_id = upload_batch_file(BATCH_INPUT_FILE, api_key)
        
        # Step 5: Create batch job
        batch_id = create_batch_job(file_id, api_key)
        
        # Step 6: Monitor progress
        final_status = monitor_batch_job(batch_id, api_key)
        
        if final_status['status'] != 'completed':
            print(f"Batch job failed: {final_status['status']}")
            return
        
        # Step 7: Download results
        output_file_id = final_status['output_file_id']
        download_batch_results(output_file_id, api_key, BATCH_OUTPUT_FILE)
        
        # Step 8: Save to database
        embeddings_saved = save_embeddings_to_db(BATCH_OUTPUT_FILE, occupations, SOURCE_DB)
        
        # Step 9: Show examples
        show_examples(SOURCE_DB)
        
        # Final summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total occupations: {len(occupations):,}")
        print(f"Embeddings generated: {embeddings_saved:,}")
        print(f"Model: {VOYAGE_MODEL}")
        print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
        print(f"Database: {SOURCE_DB}")
        print("="*60)
        
        print("\n✓ Process completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
