#!/usr/bin/env python3
"""
Calculate cosine similarity between job titles from offers and KZIS occupations.
Find top 3 closest matches and save to database.
"""

import sqlite3
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_job_embeddings(jobs_db_path: str) -> Tuple[List[str], np.ndarray]:
    """Load job title embeddings from jobs database."""
    print(f"Loading job embeddings from {jobs_db_path}...")
    conn = sqlite3.connect(jobs_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT job_title, embedding FROM job_title_embeddings ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    
    titles = []
    embeddings = []
    
    for title, embedding_bytes in rows:
        titles.append(title)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(embedding)
    
    print(f"✓ Loaded {len(titles):,} job title embeddings")
    return titles, np.array(embeddings)


def load_kzis_embeddings(kzis_db_path: str) -> Tuple[List[int], List[str], np.ndarray]:
    """Load KZIS occupation embeddings."""
    print(f"\nLoading KZIS embeddings from {kzis_db_path}...")
    conn = sqlite3.connect(kzis_db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT occupation_id, nazwa, embedding 
        FROM occupation_embeddings 
        ORDER BY occupation_id
    """)
    rows = cursor.fetchall()
    conn.close()
    
    occupation_ids = []
    occupation_names = []
    embeddings = []
    
    for occ_id, name, embedding_bytes in rows:
        occupation_ids.append(occ_id)
        occupation_names.append(name)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(embedding)
    
    print(f"✓ Loaded {len(occupation_names):,} KZIS occupation embeddings")
    return occupation_ids, occupation_names, np.array(embeddings)


def create_matches_table(jobs_db_path: str) -> None:
    """Create table for storing job-to-KZIS matches."""
    print("\nCreating job_kzis_matches table...")
    conn = sqlite3.connect(jobs_db_path)
    cursor = conn.cursor()
    
    # Drop and create table
    cursor.execute("DROP TABLE IF EXISTS job_kzis_matches")
    cursor.execute("""
        CREATE TABLE job_kzis_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            rank INTEGER NOT NULL,
            kzis_occupation_id INTEGER NOT NULL,
            kzis_occupation_name TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_match UNIQUE (job_title, rank)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_job_title_match ON job_kzis_matches(job_title)")
    cursor.execute("CREATE INDEX idx_kzis_occ_id ON job_kzis_matches(kzis_occupation_id)")
    cursor.execute("CREATE INDEX idx_similarity ON job_kzis_matches(similarity_score DESC)")
    
    conn.commit()
    conn.close()
    print("✓ Created job_kzis_matches table")


def find_top_matches(
    job_titles: List[str],
    job_embeddings: np.ndarray,
    kzis_ids: List[int],
    kzis_names: List[str],
    kzis_embeddings: np.ndarray,
    top_k: int = 3
) -> List[Tuple[str, int, int, str, float]]:
    """
    Find top K closest KZIS occupations for each job title.
    Returns: List of (job_title, rank, kzis_id, kzis_name, similarity)
    """
    print(f"\nCalculating similarities for {len(job_titles):,} job titles...")
    print(f"Finding top {top_k} matches for each...")
    
    all_matches = []
    
    # Process in batches with progress bar
    with tqdm(total=len(job_titles), desc="Processing", unit="jobs") as pbar:
        for idx, (job_title, job_embedding) in enumerate(zip(job_titles, job_embeddings)):
            # Calculate similarities with all KZIS occupations
            similarities = []
            for kzis_idx, kzis_embedding in enumerate(kzis_embeddings):
                sim = cosine_similarity(job_embedding, kzis_embedding)
                similarities.append((kzis_idx, sim))
            
            # Sort by similarity (descending) and get top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_k]
            
            # Store results
            for rank, (kzis_idx, similarity) in enumerate(top_matches, 1):
                all_matches.append((
                    job_title,
                    rank,
                    kzis_ids[kzis_idx],
                    kzis_names[kzis_idx],
                    round(similarity, 2)  # Round to 2 decimals
                ))
            
            pbar.update(1)
    
    print(f"✓ Generated {len(all_matches):,} matches")
    return all_matches


def save_matches_to_db(matches: List[Tuple], jobs_db_path: str) -> None:
    """Save matches to database."""
    print("\nSaving matches to database...")
    conn = sqlite3.connect(jobs_db_path)
    cursor = conn.cursor()
    
    # Insert in batches
    batch_size = 1000
    for i in tqdm(range(0, len(matches), batch_size), desc="Saving", unit="batch"):
        batch = matches[i:i + batch_size]
        cursor.executemany("""
            INSERT INTO job_kzis_matches 
            (job_title, rank, kzis_occupation_id, kzis_occupation_name, similarity_score)
            VALUES (?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
    
    conn.close()
    print(f"✓ Saved {len(matches):,} matches to database")


def show_examples(jobs_db_path: str, n: int = 10) -> None:
    """Show some example matches."""
    print("\n" + "="*80)
    print("EXAMPLE MATCHES")
    print("="*80)
    
    conn = sqlite3.connect(jobs_db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT job_title, rank, kzis_occupation_name, similarity_score
        FROM job_kzis_matches
        WHERE job_title IN (
            SELECT DISTINCT job_title FROM job_kzis_matches LIMIT ?
        )
        ORDER BY job_title, rank
    """, (n,))
    
    current_job = None
    for job_title, rank, kzis_name, similarity in cursor.fetchall():
        if job_title != current_job:
            if current_job is not None:
                print()
            print(f"\nJob: {job_title}")
            current_job = job_title
        
        print(f"  {rank}. {kzis_name:50s} (similarity: {similarity:.2f})")
    
    conn.close()


def show_statistics(jobs_db_path: str) -> None:
    """Show statistics about matches."""
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    conn = sqlite3.connect(jobs_db_path)
    cursor = conn.cursor()
    
    # Count total matches
    cursor.execute("SELECT COUNT(*) FROM job_kzis_matches")
    total_matches = cursor.fetchone()[0]
    
    # Count unique job titles
    cursor.execute("SELECT COUNT(DISTINCT job_title) FROM job_kzis_matches")
    unique_jobs = cursor.fetchone()[0]
    
    # Count unique KZIS occupations matched
    cursor.execute("SELECT COUNT(DISTINCT kzis_occupation_id) FROM job_kzis_matches")
    unique_kzis = cursor.fetchone()[0]
    
    # Average similarity by rank
    cursor.execute("""
        SELECT rank, 
               AVG(similarity_score) as avg_sim,
               MIN(similarity_score) as min_sim,
               MAX(similarity_score) as max_sim
        FROM job_kzis_matches
        GROUP BY rank
        ORDER BY rank
    """)
    rank_stats = cursor.fetchall()
    
    # Top matched KZIS occupations
    cursor.execute("""
        SELECT kzis_occupation_name, COUNT(*) as match_count
        FROM job_kzis_matches
        GROUP BY kzis_occupation_name
        ORDER BY match_count DESC
        LIMIT 10
    """)
    top_kzis = cursor.fetchall()
    
    conn.close()
    
    print(f"\nTotal matches: {total_matches:,}")
    print(f"Unique job titles: {unique_jobs:,}")
    print(f"Unique KZIS occupations matched: {unique_kzis:,}")
    
    print("\nSimilarity by rank:")
    for rank, avg_sim, min_sim, max_sim in rank_stats:
        print(f"  Rank {rank}: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")
    
    print("\nTop 10 most matched KZIS occupations:")
    for i, (name, count) in enumerate(top_kzis, 1):
        print(f"  {i:2d}. {name:50s} ({count:,} matches)")
    
    print("="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("JOB TITLES TO KZIS OCCUPATIONS MATCHING")
    print("="*80)
    
    JOBS_DB = "jobs_database.db"
    KZIS_DB = "kzis_occupations_descriptions.db"
    
    try:
        # Step 1: Load embeddings
        job_titles, job_embeddings = load_job_embeddings(JOBS_DB)
        kzis_ids, kzis_names, kzis_embeddings = load_kzis_embeddings(KZIS_DB)
        
        # Step 2: Create matches table
        create_matches_table(JOBS_DB)
        
        # Step 3: Find top matches
        matches = find_top_matches(
            job_titles,
            job_embeddings,
            kzis_ids,
            kzis_names,
            kzis_embeddings,
            top_k=3
        )
        
        # Step 4: Save to database
        save_matches_to_db(matches, JOBS_DB)
        
        # Step 5: Show examples and statistics
        show_examples(JOBS_DB, n=5)
        show_statistics(JOBS_DB)
        
        print("\n" + "="*80)
        print("✓ MATCHING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {JOBS_DB}")
        print(f"Table: job_kzis_matches")
        print(f"  - job_title: Title from job offers")
        print(f"  - rank: 1, 2, or 3 (best to worst match)")
        print(f"  - kzis_occupation_id: KZIS occupation ID")
        print(f"  - kzis_occupation_name: KZIS occupation name")
        print(f"  - similarity_score: Cosine similarity (0.00 to 1.00)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
