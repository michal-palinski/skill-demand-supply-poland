# Job Embeddings System - Documentation

System do generowania i wyszukiwania embeddingów dla tytułów zawodów z danych pracuj.pl

## 📊 Podsumowanie

- **Źródło danych**: `pracujpl_2025.parquet`
- **Liczba ogłoszeń**: 749,569
- **Unikalne tytuły**: 186,030
- **Wygenerowane embeddingi**: 100,000 (53.8% pokrycia)
- **Model**: VoyageAI `voyage-4-large`
- **Wymiary embeddingów**: 1024
- **Metoda**: Batch API (33% taniej niż standardowe API)

## 🗂️ Struktura plików

```
├── jobs_database.db              # SQLite database (2.1 GB)
│   ├── job_ads                   # Tabela z ogłoszeniami
│   └── job_title_embeddings      # Tabela z embeddingami
├── batch_input.jsonl             # Input do VoyageAI Batch API (4.38 MB)
├── batch_output.jsonl            # Output z embeddingami (1.31 GB)
├── process_jobs_embeddings_batch.py  # Główny skrypt do generowania
└── query_job_embeddings.py       # Skrypt do wyszukiwania podobieństw
```

## 🚀 Jak używać

### 1. Generowanie embeddingów (już wykonane)

```bash
python3 process_jobs_embeddings_batch.py
```

Ten skrypt:
1. Ładuje dane z parquet do SQLite
2. Ekstraktuje unikalne tytuły zawodów
3. Tworzy batch input JSONL
4. Przesyła do VoyageAI Batch API
5. Monitoruje postęp
6. Pobiera wyniki i zapisuje do bazy

### 2. Wyszukiwanie podobnych zawodów

```bash
python3 query_job_embeddings.py
```

Lub użyj funkcji w swoim kodzie:

```python
from query_job_embeddings import find_similar_jobs, search_with_custom_query

# Znajdź podobne zawody do istniejącego tytułu
similar = find_similar_jobs("Data Scientist", "jobs_database.db", top_k=10)

# Wyszukaj używając własnego zapytania
api_key = os.getenv("VOYAGE_API_KEY")
results = search_with_custom_query(
    "inżynier uczenia maszynowego", 
    "jobs_database.db", 
    api_key, 
    top_k=10
)
```

## 📋 Struktura bazy danych

### Tabela: `job_ads`
Zawiera wszystkie ogłoszenia z pliku parquet:
- `id`, `external_id`, `url`, `title`, `description`
- `requirements`, `responsibilities`, `benefits`
- `job_category_general`, `job_category_narrow`
- `salary_min`, `salary_max`, `salary_currency`
- `location`, `posted_date`, `company_id`
- i wiele innych pól...

### Tabela: `job_title_embeddings`
Zawiera embeddingi dla unikalnych tytułów:
- `id` - PRIMARY KEY
- `job_title` - TEXT (UNIQUE)
- `embedding` - BLOB (1024 float32 values)
- `created_at` - TIMESTAMP

## 🔍 Przykłady zapytań SQL

### Pobranie embeddingu dla konkretnego zawodu
```sql
SELECT job_title, embedding 
FROM job_title_embeddings 
WHERE job_title = 'Data Scientist';
```

### Top 10 najbardziej popularnych tytułów (z embeddingami)
```sql
SELECT 
    e.job_title, 
    COUNT(a.id) as count
FROM job_title_embeddings e
JOIN job_ads a ON e.job_title = a.title
GROUP BY e.job_title
ORDER BY count DESC
LIMIT 10;
```

### Tytuły bez embeddingów
```sql
SELECT DISTINCT title 
FROM job_ads 
WHERE title NOT IN (
    SELECT job_title FROM job_title_embeddings
)
ORDER BY title;
```

## 🔧 Techniczne detale

### VoyageAI Batch API
- **Endpoint**: `/v1/embeddings`
- **Model**: `voyage-4-large`
- **Input type**: `document`
- **Output dimension**: 1024
- **Batch size**: 128 tytułów na request
- **Total requests**: 782
- **Completion window**: 12 godzin
- **Actual completion time**: ~7 minut

### Parametry embeddingów
- **Typ**: float32
- **Rozmiar**: 1024 wymiary × 4 bajty = 4096 bajtów na embedding
- **Całkowity rozmiar**: 100,000 × 4096 = ~400 MB embeddingów

### Podobieństwo
Używamy **cosine similarity** do mierzenia podobieństwa między embeddingami:

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Wartości:
- `1.0` = identyczne
- `0.9+` = bardzo podobne
- `0.7-0.9` = podobne
- `< 0.7` = mało podobne

## 📈 Dalsze kroki

### Przetworzenie pozostałych 86,030 tytułów
Obecnie przetworzono 100,000 z 186,030 tytułów (limit batch API). Aby przetworzyć resztę:

```python
# Modyfikuj w process_jobs_embeddings_batch.py:
unique_titles = unique_titles[100000:186030]  # Następne 86,030
```

Lub podziel na dodatkowe batche po 100k każdy.

### Integracja z wyszukiwarką
Możesz zintegrować embeddingi z:
- **Elasticsearch** z KNN search
- **FAISS** dla szybkiego wyszukiwania podobieństw
- **Pinecone** lub **Weaviate** (vector databases)
- **pgvector** dla PostgreSQL

### Clustering i analiza
Użyj embeddingów do:
- Grupowania podobnych zawodów (K-means, DBSCAN)
- Wizualizacji (t-SNE, UMAP)
- Rekomendacji zawodów
- Analizy trendów na rynku pracy

## 🔐 Zmienne środowiskowe

Wymagane w `.env`:
```
VOYAGE_API_KEY=pa-your-api-key-here
```

## 💰 Koszty

Przy użyciu Batch API (33% taniej):
- 100,000 tytułów × średnio ~10 tokenów = ~1M tokenów
- Koszt z Batch API: znacznie niższy niż standardowe API
- Całkowity czas: ~7 minut

## 📚 Referencje

- [VoyageAI Batch API Documentation](https://docs.voyageai.com/docs/batch-inference)
- [VoyageAI Embeddings Models](https://docs.voyageai.com/docs/embeddings)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

## ⚠️ Uwagi

1. **Limit batch**: Max 100K inputów na batch
2. **Token limit**: Max 120K tokenów dla voyage-4-large na request
3. **Rate limiting**: Batch API automatycznie zarządza rate limits
4. **Plik output**: 1.3 GB - zachowaj go do debugowania lub usuń po zapisaniu do DB
5. **Embeddingi są znormalizowane**: Idealny do cosine similarity

---

**Autor**: Automatycznie wygenerowane  
**Data**: 2026-02-05  
**Wersja**: 1.0
