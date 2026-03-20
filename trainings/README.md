# BUR Training Services Scraper

This script scrapes training services from the Baza Usług Rozwojowych (BUR) API and saves them to PostgreSQL.

## Features

- Authenticates with BUR API using username and API key
- Fetches all training services with pagination
- Saves 65+ fields per service including:
  - Basic info (title, status, dates)
  - Categories and types
  - Pricing information (stored in grosze/cents)
  - Qualifications
  - Service provider details
  - Contact information
  - Full raw JSON data
- Uses upsert logic (INSERT ... ON CONFLICT) to handle updates
- Creates necessary database tables and indexes automatically

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (one level above `trainings/`) with your credentials:

```env
# BUR API Credentials
BUR_USERNAME=your_username
BUR_API_KEY=your_api_key

# PostgreSQL Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trainings_pl
DB_USER=postgres
DB_PASSWORD=your_password
```

See `env.example` for reference.

### 3. Prepare Database

Make sure PostgreSQL is running and the database exists:

```sql
CREATE DATABASE trainings_pl;
```

The script will automatically create the `bur_services` table on first run.

## Usage

Run the scraper:

```bash
python trainings/scraper_trainings_bur.py
```

The script will:
1. Connect to PostgreSQL
2. Create/verify the database table
3. Login to BUR API
4. Fetch all service pages
5. Save each service to the database
6. Display progress with ✓ checkmarks

## Database Schema

The `bur_services` table includes:

- **Basic fields**: id, numer, tytul, status, dates
- **Categories**: id_kategorii_uslugi, id_podkategorii_uslugi, etc.
- **Pricing**: cena_brutto_za_usluge, cena_netto_za_uczestnika, etc. (in grosze)
- **Content**: cel_biznesowy, program_uslugi, grupa_docelowa, etc.
- **Complex data as JSONB**: adres, dostawca_uslug, osoba_kontaktowa, etc.
- **Full data**: raw_data (JSONB with complete API response)
- **Metadata**: scraped_at, updated_at

### Indexes

- `status` - for filtering by service status
- `id_kategorii_uslugi` - for filtering by category
- `data_rozpoczecia_uslugi` - for date range queries

## API Documentation

The BUR API schema is available at:
https://uslugirozwojowe.parp.gov.pl/api/schemat.json

Interactive Swagger UI:
https://uslugirozwojowe.parp.gov.pl/api/#/

## Notes

- Prices are stored in grosze (cents) - divide by 100 to get PLN
- The script uses `ON CONFLICT` to update existing records
- SSL verification is disabled for the HTTP client (`verify=False`)
- All timestamps are stored with timezone info
- Complex nested objects are stored as JSONB for flexible querying

## Streamlit — zakładka Trainings (BUR × województwo × ESCO L1/L2)

1. **Parquet szkoleń** (np. `data/bur_trainings_0126.parquet`): kolumny `id`, `adres` (JSON BUR → `nazwaWojewodztwa`).
2. **Mapowanie KaLM → ESCO** (osobno): `data/bur_to_esco_kalm_top1.sqlite` z tabelą `bur_esco_kalm_top1`
   (`esco_conceptUri`, `bur_bur_ids_json`, `similarity` / `keep`). Zbudujesz: `python bur_esco_parquet_to_sqlite.py`.
3. Z katalogu głównego projektu (SQLite jest wykrywany automatycznie, jeśli leży w `trainings/data/`):
   ```bash
   python precompute_trainings_regional_cache.py --parquet trainings/data/bur_trainings_0126.parquet
   # jawnie:
   python precompute_trainings_regional_cache.py \
     --parquet trainings/data/bur_trainings_0126.parquet \
     --esco-sqlite trainings/data/bur_to_esco_kalm_top1.sqlite
   ```
   Zapis: `app_deploy/trainings_regional_cache.json` (agregaty, bez rekordów jednostkowych w aplikacji).
4. Deploy: `python prepare_deploy.py` — skopiuje cache do `deploy/`, jeśli plik istnieje.

