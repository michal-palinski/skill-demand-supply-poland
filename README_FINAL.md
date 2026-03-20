# Job Search System - Final Documentation

Complete system for job offer embeddings, KZIS occupation matching, and semantic search.

## 🎯 Overview

This system processes job offers from pracuj.pl, generates embeddings using VoyageAI, matches them with standardized KZIS occupations, and provides a web interface for semantic search.

## 📊 Data Summary

### Job Offers Database (`jobs_database.db` - 2.3 GB)
- **Total job ads**: 749,569
- **Unique job titles**: 186,030
- **Job-to-KZIS matches**: 558,090 (3 per job title)

### KZIS Occupations (`kzis_occupations_descriptions.db` - 7.5 MB)
- **Total occupations**: 2,945
- **All with embeddings**: Yes

### FAISS Vector Indexes (`faiss_indexes/`)
- **job_titles.index**: 186,030 vectors (1024 dimensions)
- **kzis_occupations.index**: 2,945 vectors (1024 dimensions)
- **Metadata files**: Pickled mappings for IDs and names

## 🗂️ Project Structure

```
├── jobs_database.db                      # Main job offers database
├── kzis_occupations_descriptions.db      # KZIS occupations database
├── faiss_indexes/                        # Vector indexes
│   ├── job_titles.index
│   ├── job_titles_metadata.pkl
│   ├── kzis_occupations.index
│   └── kzis_occupations_metadata.pkl
│
├── app_search.py                         # 🌐 Main Streamlit app
├── faiss_search.py                       # FAISS search utilities
│
├── process_jobs_embeddings_batch.py      # Generate job embeddings (batch 1)
├── save_remaining_embeddings.py          # Save remaining embeddings
├── embed_occupations_direct.py           # Generate KZIS embeddings
├── match_jobs_to_kzis.py                 # Match jobs to KZIS
├── migrate_to_faiss.py                   # Migrate to FAISS
│
└── requirements.txt                      # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create `.env` file:
```
VOYAGE_API_KEY=your-api-key-here
```

### 3. Run the Web App

```bash
python3 -m streamlit run app_search.py
```

Open: http://localhost:8501

## 💡 Features

### Web Interface (app_search.py)

**Search Modes:**
1. **Search by text** - Free-text semantic search
   - Enter any query (e.g., "data scientist", "machine learning")
   - Returns most similar job titles
   - Shows KZIS occupation matches with similarity scores

2. **Browse job titles** - Dropdown with all 186k titles
   - Select specific job title
   - View KZIS matches
   - See similar job titles

**Display:**
- Job title
- Number of active positions
- Top 3 KZIS occupation matches with similarity scores (0.00-1.00)
- Similar job titles with scores

### FAISS Search (faiss_search.py)

```python
from faiss_search import FAISSSearch

searcher = FAISSSearch()

# Find similar jobs
results = searcher.search_similar_jobs("Data Scientist", k=10)

# Search with custom text
results = searcher.search_with_text("machine learning engineer", "job", k=10)
```

## 📋 Database Schema

### Table: `job_ads`
Main job offers table with all details:
- `id`, `external_id`, `url`, `title`
- `description`, `requirements`, `responsibilities`
- `job_category_general`, `job_category_narrow`
- `salary_min`, `salary_max`, `location`
- `posted_date`, `company_id`, etc.

### Table: `job_title_embeddings` (no embedding column)
Metadata for job titles (embeddings in FAISS):
- `id`, `job_title`
- `created_at`

### Table: `job_kzis_matches`
Job-to-KZIS occupation mappings:
- `job_title` - Job offer title
- `rank` - 1, 2, or 3 (best to worst match)
- `kzis_occupation_id` - KZIS occupation ID
- `kzis_occupation_name` - KZIS occupation name
- `similarity_score` - Cosine similarity (rounded to 2 decimals)

### Table: `opisy_zawodow` (in KZIS DB)
KZIS occupation descriptions:
- `id`, `kod`, `nazwa`
- `synteza`, `zadania_zawodowe`, `dodatkowe_zadania`

### Table: `occupation_embeddings` (in KZIS DB, no embedding column)
Metadata for KZIS occupations (embeddings in FAISS):
- `id`, `occupation_id`, `nazwa`

## 🔧 Technical Details

### Embeddings
- **Model**: VoyageAI `voyage-4-large`
- **Dimensions**: 1024
- **Type**: float32
- **Similarity metric**: Cosine similarity
- **Storage**: FAISS IndexFlatIP (normalized vectors)

### Batch Processing
- **Method**: VoyageAI Batch API
- **Savings**: 33% cost reduction vs. standard API
- **Completion time**: ~5-10 minutes per 100k titles

### Performance
- **FAISS search**: Sub-second for similarity queries
- **Database queries**: Fast with proper indexes
- **Web app**: Responsive, caches FAISS indexes

## 📈 Example Queries

### SQL Queries

**Find top KZIS matches for a job:**
```sql
SELECT kzis_occupation_name, similarity_score
FROM job_kzis_matches
WHERE job_title = 'Data Scientist'
ORDER BY rank;
```

**Most common KZIS occupations:**
```sql
SELECT kzis_occupation_name, COUNT(*) as matches
FROM job_kzis_matches
WHERE rank = 1
GROUP BY kzis_occupation_name
ORDER BY matches DESC
LIMIT 10;
```

**Average similarity by rank:**
```sql
SELECT rank, AVG(similarity_score) as avg_similarity
FROM job_kzis_matches
GROUP BY rank;
```

### Python API

**Search with FAISS:**
```python
import faiss
import pickle
import numpy as np

# Load index
index = faiss.read_index("faiss_indexes/job_titles.index")
with open("faiss_indexes/job_titles_metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

# Search
query_idx = metadata['titles'].index("Data Scientist")
query_vec = index.reconstruct(query_idx).reshape(1, -1)
faiss.normalize_L2(query_vec)

distances, indices = index.search(query_vec, 10)

for dist, idx in zip(distances[0], indices[0]):
    print(f"{metadata['titles'][idx]}: {dist:.3f}")
```

## 🎨 UI Design Philosophy

**Minimalist principles:**
- Clean, spacious layout
- Large, touch-friendly elements
- No icons or unnecessary decoration
- Clear typography hierarchy
- Subtle hover effects
- Professional color scheme

**User experience:**
- Two simple search modes
- Immediate results
- Clear similarity scores
- Easy to scan results

## 📊 Statistics

### Coverage
- **Job titles with embeddings**: 100% (186,030 / 186,030)
- **KZIS with embeddings**: 100% (2,945 / 2,945)
- **Job-KZIS mappings**: 100% (186,030 × 3 = 558,090)

### File Sizes
- **jobs_database.db**: 2.3 GB (without embeddings)
- **kzis_occupations_descriptions.db**: 7.5 MB
- **FAISS indexes**: ~750 MB total
- **Total system**: ~3.1 GB

### Performance Metrics
- **Embedding generation**: ~100 jobs/second (batch API)
- **Similarity calculation**: ~98 jobs/second
- **FAISS search**: <100ms for top-k queries
- **Database queries**: <500ms typical

## 🔄 Workflow

### Initial Setup (Already Completed)
1. ✅ Load `pracujpl_2025.parquet` → SQLite
2. ✅ Extract 186,030 unique job titles
3. ✅ Generate embeddings via VoyageAI Batch API
4. ✅ Load KZIS occupations
5. ✅ Generate KZIS embeddings
6. ✅ Calculate job-to-KZIS similarities
7. ✅ Migrate embeddings to FAISS
8. ✅ Build web interface

### Daily Usage
1. Open web app: `python3 -m streamlit run app_search.py`
2. Search for jobs or browse titles
3. View KZIS matches and similar positions

## 🛠️ Maintenance

### Update Embeddings
When new job data arrives:
```bash
# Generate embeddings for new titles
python3 process_jobs_embeddings_batch.py

# Match to KZIS
python3 match_jobs_to_kzis.py

# Migrate to FAISS
python3 migrate_to_faiss.py
```

### Backup
Important files to backup:
- `jobs_database.db`
- `kzis_occupations_descriptions.db`
- `faiss_indexes/` folder

## 💰 Costs

**VoyageAI Batch API:**
- Job titles: 186,030 titles × ~10 tokens = ~1.86M tokens
- KZIS: 2,945 occupations × ~10 tokens = ~29K tokens
- **Total**: ~1.89M tokens
- **With batch discount**: 33% savings

## 🔐 Environment Variables

Required in `.env`:
```bash
VOYAGE_API_KEY=pa-your-key-here
```

## 📚 References

- [VoyageAI Batch API](https://docs.voyageai.com/docs/batch-inference)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Streamlit Docs](https://docs.streamlit.io)

## ✅ Checklist

- [x] Job embeddings generated (186,030)
- [x] KZIS embeddings generated (2,945)
- [x] Job-KZIS matching complete (558,090 matches)
- [x] FAISS migration complete
- [x] Web app deployed locally
- [x] Documentation complete

---

**Created**: 2026-02-06  
**Status**: ✅ Production Ready  
**Version**: 1.0
