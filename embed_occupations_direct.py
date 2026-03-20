#!/usr/bin/env python3
"""
Generate embeddings for KZIS occupations using direct API (faster for small dataset).
"""

import os
import sqlite3
import numpy as np
from typing import List
from dotenv import load_dotenv
import voyageai
from tqdm import tqdm

load_dotenv()

SOURCE_DB = "kzis_occupations_descriptions.db"
VOYAGE_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 128  # Per API call


def get_occupations(db_path: str) -> List[tuple]:
    """Get all occupation names."""
    print(f"Reading occupations from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nazwa FROM opisy_zawodow ORDER BY id")
    occupations = cursor.fetchall()
    conn.close()
    print(f"Found {len(occupations)} occupations")
    return occupations


def create_embeddings_table(db_path: str) -> None:
    """Create embeddings table."""
    print("\nCreating table...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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
    print("✓ Table created")


def generate_embeddings(occupations: List[tuple], api_key: str) -> List[tuple]:
    """Generate embeddings using direct API."""
    print(f"\nGenerating embeddings with {VOYAGE_MODEL}...")
    vo = voyageai.Client(api_key=api_key)
    
    all_embeddings = []
    names = [name for _, name in occupations]
    
    # Process in batches
    total_batches = (len(names) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=len(names), desc="Processing", unit="occupations") as pbar:
        for i in range(0, len(names), BATCH_SIZE):
            batch_names = names[i:i + BATCH_SIZE]
            batch_ids = [occ_id for occ_id, _ in occupations[i:i + BATCH_SIZE]]
            
            # Call API
            result = vo.embed(
                texts=batch_names,
                model=VOYAGE_MODEL,
                input_type="document",
                output_dimension=EMBEDDING_DIMENSIONS
            )
            
            # Store with IDs
            for occ_id, name, embedding in zip(batch_ids, batch_names, result.embeddings):
                all_embeddings.append((occ_id, name, embedding))
            
            pbar.update(len(batch_names))
    
    print(f"✓ Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def save_to_db(embeddings: List[tuple], db_path: str) -> None:
    """Save embeddings to database."""
    print("\nSaving to database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for occ_id, name, embedding in tqdm(embeddings, desc="Saving", unit="rows"):
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        cursor.execute(
            "INSERT INTO occupation_embeddings (occupation_id, nazwa, embedding) VALUES (?, ?, ?)",
            (occ_id, name, embedding_bytes)
        )
    
    conn.commit()
    conn.close()
    print(f"✓ Saved {len(embeddings)} embeddings")


def show_sample(db_path: str) -> None:
    """Show sample embeddings."""
    print("\n" + "="*60)
    print("SAMPLE EMBEDDINGS")
    print("="*60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT o.kod, e.nazwa, LENGTH(e.embedding)/4 as dims
        FROM occupation_embeddings e
        JOIN opisy_zawodow o ON e.occupation_id = o.id
        LIMIT 5
    """)
    
    for kod, nazwa, dims in cursor.fetchall():
        print(f"{kod}: {nazwa} ({dims} dims)")
    
    conn.close()


def main():
    print("="*60)
    print("KZIS OCCUPATION EMBEDDINGS (DIRECT API)")
    print("="*60)
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY not found")
        return
    
    try:
        # Get occupations
        occupations = get_occupations(SOURCE_DB)
        
        # Create table
        create_embeddings_table(SOURCE_DB)
        
        # Generate embeddings
        embeddings = generate_embeddings(occupations, api_key)
        
        # Save to DB
        save_to_db(embeddings, SOURCE_DB)
        
        # Show sample
        show_sample(SOURCE_DB)
        
        print("\n" + "="*60)
        print(f"✓ COMPLETED!")
        print(f"Total: {len(embeddings)} embeddings")
        print(f"Model: {VOYAGE_MODEL}")
        print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
