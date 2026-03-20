import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import json

# =========================
# LOAD ENV
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trainings_pl")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

OUT_DIR = BASE_DIR / "trainings" / "data"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 50_000))

# =========================
# SQL - Export all trainings data
# Excluding raw_data to speed up export (it's huge!)
# =========================
QUERY = """
SELECT 
    id,
    numer,
    tytul,
    status,
    data_rozpoczecia_uslugi,
    data_zakonczenia_uslugi,
    data_zakonczenia_rekrutacji,
    id_kategorii_uslugi,
    id_podkategorii_uslugi,
    id_rodzaju_uslugi,
    id_podrodzaju_uslugi,
    id_formy_swiadczenia,
    id_kwalifikacji_kkz,
    id_kwalifikacji_zrk,
    cel_biznesowy,
    cel_edukacyjny,
    efekt_uslugi,
    efekty_uczenia_sie,
    program_uslugi,
    grupa_docelowa,
    warunki_uczestnictwa,
    warunki_techniczne,
    informacje_dodatkowe,
    cena_brutto_za_usluge,
    cena_netto_za_usluge,
    cena_brutto_za_uczestnika,
    cena_netto_za_uczestnika,
    cena_brutto_za_godzine,
    cena_netto_za_godzine,
    czy_cena_dotyczy_calej_uslugi,
    liczba_godzin,
    minimalna_liczba_uczestnikow,
    maksymalna_liczba_uczestnikow,
    adres,
    dostawca_uslug,
    osoba_kontaktowa,
    ocena,
    lista_projektow,
    warunki_logistyczne,
    formy_dofinansowania,
    kwalifikacje_kuz,
    inne_kwalifikacje,
FROM public.bur_services
ORDER BY id
"""

# =========================
# DB ENGINE
# =========================
print(f"🔌 Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}...")
DB_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
engine = create_engine(DB_URL)

# =========================
# EXTRACT → PARQUET
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

# Get total count for progress bar
print("📊 Counting total records...")
with engine.connect() as conn:
    count_query = text("""
        SELECT COUNT(*) as total
        FROM public.bur_services
    """)
    total_rows = conn.execute(count_query).scalar()
    print(f"✅ Found {total_rows:,} training records")

if total_rows == 0:
    print("⚠️  No data found in bur_services table!")
    print("💡 Run 'python trainings/scraper_trainings_bur.py' to populate the database")
    exit(0)

# Export with progress bar using LIMIT/OFFSET for true chunking
print(f"📥 Exporting trainings to {OUT_DIR}...")
print(f"⏳ Processing {CHUNK_SIZE:,} rows at a time...")

part = 0
offset = 0
part_files = []

with engine.connect() as conn:
    with tqdm(total=total_rows, desc="Exporting", unit="rows", unit_scale=True) as pbar:
        while offset < total_rows:
            # Build query with LIMIT and OFFSET for true chunking
            chunk_query = f"{QUERY} LIMIT {CHUNK_SIZE} OFFSET {offset}"
            
            # Read this chunk
            df = pd.read_sql(text(chunk_query), con=conn)
            
            if len(df) == 0:
                break
            
            # Convert JSONB columns to strings (they cause issues with PyArrow)
            jsonb_columns = [
                'adres', 'dostawca_uslug', 'osoba_kontaktowa', 'ocena',
                'lista_projektow', 'warunki_logistyczne', 'formy_dofinansowania',
                'kwalifikacje_kuz', 'inne_kwalifikacje'
            ]
            
            for col in jsonb_columns:
                if col in df.columns:
                    # Convert to JSON string, handling all types including None/NaN/arrays
                    def safe_json_dumps(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return None
                        try:
                            return json.dumps(x)
                        except (TypeError, ValueError):
                            return str(x)
                    
                    df[col] = df[col].apply(safe_json_dumps)
            
            # Determine output path
            if total_rows <= CHUNK_SIZE:
                # Single file for small datasets
                out_path = OUT_DIR / "trainings_all.parquet"
            else:
                # Multiple parts for large datasets
                out_path = OUT_DIR / f"trainings_part_{part:05d}.parquet"
                part_files.append(out_path)
            
            # Write to parquet
            df.to_parquet(
                out_path,
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            
            # Update progress
            pbar.update(len(df))
            pbar.set_postfix({
                "part": part, 
                "chunk": len(df), 
                "offset": f"{offset:,}"
            })
            
            part += 1
            offset += len(df)

# Create a combined file if we have multiple parts
if part > 1 and part_files:
    print(f"\n📦 Combining {part} parts into single file...")
    combined_df = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    combined_path = OUT_DIR / "trainings_all.parquet"
    combined_df.to_parquet(
        combined_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    file_size = combined_path.stat().st_size / 1024**2
    print(f"✅ Combined file: {combined_path} ({file_size:.2f} MB)")
    
    # Optionally remove part files
    print(f"\n🗑️  Remove part files? (keeping: trainings_all.parquet)")
    response = input("   Remove parts? [y/N]: ").strip().lower()
    if response == 'y':
        for f in part_files:
            f.unlink()
        print(f"✅ Removed {len(part_files)} part files")

print(f"\n✅ Done! Exported {total_rows:,} records")
if part == 1:
    file_size = (OUT_DIR / "trainings_all.parquet").stat().st_size / 1024**2
    print(f"📁 File: {OUT_DIR / 'trainings_all.parquet'} ({file_size:.2f} MB)")
else:
    print(f"📁 Location: {OUT_DIR}/")

