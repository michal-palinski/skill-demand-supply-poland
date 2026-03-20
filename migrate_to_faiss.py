#!/usr/bin/env python3
"""
Migrate embeddings from SQLite to FAISS vector store.
This will significantly reduce database size.
"""

import sqlite3
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import os


def export_job_embeddings_to_faiss(db_path: str, output_dir: str = "faiss_indexes"):
    """Export job title embeddings from SQLite to FAISS."""
    print("="*60)
    print("EXPORTING JOB EMBEDDINGS TO FAISS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    total = cursor.fetchone()[0]
    print(f"\nExporting {total:,} job title embeddings...")
    
    # Load embeddings and metadata
    cursor.execute("SELECT id, job_title, embedding FROM job_title_embeddings ORDER BY id")
    
    ids = []
    titles = []
    embeddings = []
    
    for row_id, title, emb_bytes in tqdm(cursor.fetchall(), desc="Loading", unit="rows"):
        ids.append(row_id)
        titles.append(title)
        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        embeddings.append(embedding)
    
    conn.close()
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    
    print(f"\nCreating FAISS index...")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {len(embeddings_array):,}")
    
    # Create FAISS index (using IndexFlatIP for cosine similarity)
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # Save FAISS index
    index_file = os.path.join(output_dir, "job_titles.index")
    faiss.write_index(index, index_file)
    print(f"✓ Saved FAISS index: {index_file}")
    
    # Save metadata mapping
    metadata = {
        'ids': ids,
        'titles': titles,
        'dimension': dimension,
        'count': len(ids)
    }
    metadata_file = os.path.join(output_dir, "job_titles_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_file}")
    
    return len(embeddings_array)


def export_kzis_embeddings_to_faiss(db_path: str, output_dir: str = "faiss_indexes"):
    """Export KZIS occupation embeddings to FAISS."""
    print("\n" + "="*60)
    print("EXPORTING KZIS EMBEDDINGS TO FAISS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM occupation_embeddings")
    total = cursor.fetchone()[0]
    print(f"\nExporting {total:,} KZIS embeddings...")
    
    cursor.execute("""
        SELECT id, occupation_id, nazwa, embedding 
        FROM occupation_embeddings 
        ORDER BY id
    """)
    
    ids = []
    occupation_ids = []
    names = []
    embeddings = []
    
    for row_id, occ_id, name, emb_bytes in tqdm(cursor.fetchall(), desc="Loading", unit="rows"):
        ids.append(row_id)
        occupation_ids.append(occ_id)
        names.append(name)
        embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        embeddings.append(embedding)
    
    conn.close()
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    
    print(f"\nCreating FAISS index...")
    print(f"  Dimension: {dimension}")
    print(f"  Vectors: {len(embeddings_array):,}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # Save index
    index_file = os.path.join(output_dir, "kzis_occupations.index")
    faiss.write_index(index, index_file)
    print(f"✓ Saved FAISS index: {index_file}")
    
    # Save metadata
    metadata = {
        'ids': ids,
        'occupation_ids': occupation_ids,
        'names': names,
        'dimension': dimension,
        'count': len(ids)
    }
    metadata_file = os.path.join(output_dir, "kzis_occupations_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_file}")
    
    return len(embeddings_array)


def remove_embeddings_from_db(db_path: str, table: str, embedding_col: str):
    """Remove embedding column from database."""
    print(f"\nRemoving {embedding_col} column from {table}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current size
    cursor.execute("PRAGMA page_count")
    page_count_before = cursor.fetchone()[0]
    cursor.execute("PRAGMA page_size")
    page_size = cursor.fetchone()[0]
    size_before_mb = (page_count_before * page_size) / (1024 * 1024)
    
    # Drop embedding column (SQLite requires recreating table)
    print(f"  Current size: {size_before_mb:.1f} MB")
    
    # Get table schema without embedding column
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    
    # Build new table schema
    new_columns = []
    for col in columns:
        col_id, name, col_type, notnull, default, pk = col
        if name != embedding_col:
            constraints = []
            if notnull:
                constraints.append("NOT NULL")
            if default:
                constraints.append(f"DEFAULT {default}")
            if pk:
                constraints.append("PRIMARY KEY")
            
            col_def = f"{name} {col_type}"
            if constraints:
                col_def += " " + " ".join(constraints)
            new_columns.append(col_def)
    
    # Create new table without embedding
    temp_table = f"{table}_new"
    col_names = [col[1] for col in columns if col[1] != embedding_col]
    
    cursor.execute(f"CREATE TABLE {temp_table} ({', '.join(new_columns)})")
    cursor.execute(f"INSERT INTO {temp_table} SELECT {', '.join(col_names)} FROM {table}")
    cursor.execute(f"DROP TABLE {table}")
    cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table}")
    
    # Recreate indexes (except for embedding)
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,))
    indexes = cursor.fetchall()
    for idx_name, idx_sql in indexes:
        if idx_sql and embedding_col not in idx_sql:
            try:
                cursor.execute(idx_sql)
            except:
                pass
    
    conn.commit()
    
    # VACUUM to reclaim space
    print("  Running VACUUM to reclaim space...")
    conn.execute("VACUUM")
    
    # Get new size
    cursor.execute("PRAGMA page_count")
    page_count_after = cursor.fetchone()[0]
    size_after_mb = (page_count_after * page_size) / (1024 * 1024)
    
    saved_mb = size_before_mb - size_after_mb
    print(f"  New size: {size_after_mb:.1f} MB")
    print(f"  ✓ Saved {saved_mb:.1f} MB")
    
    conn.close()


def main():
    """Main execution."""
    print("="*60)
    print("MIGRATE EMBEDDINGS TO FAISS")
    print("="*60)
    print("\nThis will:")
    print("  1. Export embeddings to FAISS indexes")
    print("  2. Remove embedding columns from SQLite")
    print("  3. Reduce database size significantly")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    JOBS_DB = "jobs_database.db"
    KZIS_DB = "kzis_occupations_descriptions.db"
    OUTPUT_DIR = "faiss_indexes"
    
    try:
        # Export job embeddings
        job_count = export_job_embeddings_to_faiss(JOBS_DB, OUTPUT_DIR)
        
        # Export KZIS embeddings
        kzis_count = export_kzis_embeddings_to_faiss(KZIS_DB, OUTPUT_DIR)
        
        # Remove embeddings from databases
        print("\n" + "="*60)
        print("REMOVING EMBEDDINGS FROM DATABASES")
        print("="*60)
        
        remove_embeddings_from_db(JOBS_DB, "job_title_embeddings", "embedding")
        remove_embeddings_from_db(KZIS_DB, "occupation_embeddings", "embedding")
        
        # Final summary
        print("\n" + "="*60)
        print("MIGRATION COMPLETE!")
        print("="*60)
        print(f"\nFAISS indexes created in: {OUTPUT_DIR}/")
        print(f"  - job_titles.index ({job_count:,} vectors)")
        print(f"  - kzis_occupations.index ({kzis_count:,} vectors)")
        print(f"\nMetadata files:")
        print(f"  - job_titles_metadata.pkl")
        print(f"  - kzis_occupations_metadata.pkl")
        print(f"\nDatabases cleaned:")
        print(f"  - {JOBS_DB}")
        print(f"  - {KZIS_DB}")
        
        # Check final sizes
        jobs_size = os.path.getsize(JOBS_DB) / (1024 * 1024)
        kzis_size = os.path.getsize(KZIS_DB) / (1024 * 1024)
        print(f"\nFinal database sizes:")
        print(f"  - {JOBS_DB}: {jobs_size:.1f} MB")
        print(f"  - {KZIS_DB}: {kzis_size:.1f} MB")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
