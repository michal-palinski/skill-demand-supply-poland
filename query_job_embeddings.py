#!/usr/bin/env python3
"""
Utility script to query and search job embeddings using similarity search.
"""

import os
import sqlite3
import numpy as np
from typing import List, Tuple
import voyageai
from dotenv import load_dotenv

load_dotenv()

SQLITE_DB = "jobs_database.db"
VOYAGE_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024


def get_all_job_titles(db_path: str) -> List[str]:
    """Get all job titles that have embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT job_title FROM job_title_embeddings ORDER BY job_title")
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles


def get_embedding_from_db(db_path: str, job_title: str) -> np.ndarray:
    """Retrieve embedding for a specific job title."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM job_title_embeddings WHERE job_title = ?", (job_title,))
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        raise ValueError(f"Job title not found: {job_title}")
    
    embedding = np.frombuffer(result[0], dtype=np.float32)
    return embedding


def get_all_embeddings(db_path: str) -> Tuple[List[str], np.ndarray]:
    """Get all job titles and their embeddings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT job_title, embedding FROM job_title_embeddings ORDER BY id")
    
    titles = []
    embeddings = []
    
    for row in cursor.fetchall():
        titles.append(row[0])
        embedding = np.frombuffer(row[1], dtype=np.float32)
        embeddings.append(embedding)
    
    conn.close()
    
    return titles, np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_jobs(query_title: str, db_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find the most similar job titles to the query."""
    # Get query embedding
    query_embedding = get_embedding_from_db(db_path, query_title)
    
    # Get all embeddings
    all_titles, all_embeddings = get_all_embeddings(db_path)
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(all_embeddings):
        if all_titles[i] == query_title:
            continue  # Skip the query itself
        
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((all_titles[i], similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def search_with_custom_query(query_text: str, db_path: str, api_key: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar jobs using a custom query text (not necessarily in the database)."""
    print(f"Generating embedding for query: '{query_text}'")
    
    # Generate embedding for the query
    vo = voyageai.Client(api_key=api_key)
    result = vo.embed(
        texts=[query_text],
        model=VOYAGE_MODEL,
        input_type="query",
        output_dimension=EMBEDDING_DIMENSIONS
    )
    query_embedding = np.array(result.embeddings[0], dtype=np.float32)
    
    # Get all embeddings
    all_titles, all_embeddings = get_all_embeddings(db_path)
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(all_embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((all_titles[i], similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def main():
    """Example usage of the embedding query functions."""
    print("="*60)
    print("JOB EMBEDDINGS QUERY TOOL")
    print("="*60)
    
    # Example 1: Find similar jobs to an existing job title
    print("\n1. Finding jobs similar to 'Data Scientist':")
    print("-" * 60)
    
    try:
        similar_jobs = find_similar_jobs("Data Scientist", SQLITE_DB, top_k=10)
        for i, (job_title, similarity) in enumerate(similar_jobs, 1):
            print(f"{i:2d}. {job_title:50s} (similarity: {similarity:.4f})")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTip: Use get_all_job_titles() to see available job titles")
    
    # Example 2: Search with custom query
    print("\n\n2. Searching for 'machine learning engineer positions':")
    print("-" * 60)
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if api_key:
        try:
            results = search_with_custom_query(
                "machine learning engineer positions", 
                SQLITE_DB, 
                api_key, 
                top_k=10
            )
            for i, (job_title, similarity) in enumerate(results, 1):
                print(f"{i:2d}. {job_title:50s} (similarity: {similarity:.4f})")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("VOYAGE_API_KEY not found. Skipping custom query search.")
    
    # Show database statistics
    print("\n\n3. Database Statistics:")
    print("-" * 60)
    
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM job_ads")
    total_ads = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT title) FROM job_ads")
    unique_titles = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM job_title_embeddings")
    total_embeddings = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Total job ads: {total_ads:,}")
    print(f"Unique job titles: {unique_titles:,}")
    print(f"Job titles with embeddings: {total_embeddings:,}")
    print(f"Coverage: {total_embeddings/unique_titles*100:.1f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
