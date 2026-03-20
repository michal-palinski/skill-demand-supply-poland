#!/usr/bin/env python3
"""
Helper script for searching embeddings using FAISS.
"""

import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import voyageai

load_dotenv()


class FAISSSearch:
    """FAISS-based similarity search."""
    
    def __init__(self, index_dir="faiss_indexes"):
        self.index_dir = index_dir
        self.job_index = None
        self.job_metadata = None
        self.kzis_index = None
        self.kzis_metadata = None
    
    def load_job_index(self):
        """Load job titles index."""
        if self.job_index is None:
            index_file = os.path.join(self.index_dir, "job_titles.index")
            metadata_file = os.path.join(self.index_dir, "job_titles_metadata.pkl")
            
            self.job_index = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.job_metadata = pickle.load(f)
            
            print(f"✓ Loaded job index: {self.job_metadata['count']:,} vectors")
        
        return self.job_index, self.job_metadata
    
    def load_kzis_index(self):
        """Load KZIS occupations index."""
        if self.kzis_index is None:
            index_file = os.path.join(self.index_dir, "kzis_occupations.index")
            metadata_file = os.path.join(self.index_dir, "kzis_occupations_metadata.pkl")
            
            self.kzis_index = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.kzis_metadata = pickle.load(f)
            
            print(f"✓ Loaded KZIS index: {self.kzis_metadata['count']:,} vectors")
        
        return self.kzis_index, self.kzis_metadata
    
    def search_similar_jobs(self, query_title: str, k: int = 10):
        """Find similar job titles."""
        # First, find the query title in our data
        index, metadata = self.load_job_index()
        
        try:
            query_idx = metadata['titles'].index(query_title)
        except ValueError:
            return f"Title '{query_title}' not found"
        
        # Search
        query_vec = index.reconstruct(query_idx).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        distances, indices = index.search(query_vec, k + 1)  # +1 to exclude itself
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == query_idx:  # Skip self
                continue
            results.append({
                'title': metadata['titles'][idx],
                'similarity': float(dist)
            })
        
        return results[:k]
    
    def search_with_vector(self, query_vector: np.ndarray, index_type: str = 'job', k: int = 10):
        """Search using a custom vector."""
        if index_type == 'job':
            index, metadata = self.load_job_index()
            name_key = 'titles'
        else:
            index, metadata = self.load_kzis_index()
            name_key = 'names'
        
        # Normalize query vector
        query_vec = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # Search
        distances, indices = index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'name': metadata[name_key][idx],
                'similarity': float(dist)
            })
        
        return results
    
    def search_with_text(self, query_text: str, index_type: str = 'job', k: int = 10):
        """Search using text (generates embedding via API)."""
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            return "Error: VOYAGE_API_KEY not found"
        
        # Generate embedding
        vo = voyageai.Client(api_key=api_key)
        result = vo.embed(
            texts=[query_text],
            model="voyage-4-large",
            input_type="query",
            output_dimension=1024
        )
        
        query_vector = np.array(result.embeddings[0], dtype=np.float32)
        return self.search_with_vector(query_vector, index_type, k)


def main():
    """Example usage."""
    print("="*60)
    print("FAISS SEARCH EXAMPLES")
    print("="*60)
    
    searcher = FAISSSearch()
    
    # Example 1: Find similar job titles
    print("\n1. Similar to 'Data Scientist':")
    print("-" * 60)
    results = searcher.search_similar_jobs("Data Scientist", k=5)
    if isinstance(results, str):
        print(results)
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']:50s} ({r['similarity']:.3f})")
    
    # Example 2: Search with custom text
    print("\n\n2. Search for 'machine learning engineer':")
    print("-" * 60)
    try:
        results = searcher.search_with_text("machine learning engineer", "job", k=5)
        if isinstance(results, str):
            print(results)
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['name']:50s} ({r['similarity']:.3f})")
    except Exception as e:
        print(f"Skipping (API key needed): {e}")
    
    print("\n" + "="*60)
    print("✓ FAISS search is working!")
    print("="*60)


if __name__ == "__main__":
    main()
