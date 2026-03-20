"""
Extract code dictionaries with labels from BUR services data
"""
import psycopg2
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trainings_pl")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def main():
    print("🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("✅ Connected\n")
    
    # Query to get unique ID combinations with their raw_data
    query = """
    SELECT DISTINCT
        id_kategorii_uslugi,
        id_podkategorii_uslugi,
        id_rodzaju_uslugi,
        id_podrodzaju_uslugi,
        id_formy_swiadczenia,
        raw_data
    FROM public.bur_services
    WHERE raw_data IS NOT NULL
    LIMIT 1000
    """
    
    print("📥 Fetching sample records with raw_data...")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    print(f"✅ Got {len(rows)} records\n")
    
    # Build dictionaries from raw_data
    kategoria = {}
    podkategoria = {}
    rodzaj = {}
    podrodzaj = {}
    forma = {}
    
    for row in rows:
        if row[5]:  # raw_data exists
            try:
                raw = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                
                # Extract category info if available
                if row[0] and 'kategoriaUslugi' in raw:
                    kategoria[row[0]] = raw['kategoriaUslugi']
                if row[1] and 'podkategoriaUslugi' in raw:
                    podkategoria[row[1]] = raw['podkategoriaUslugi']
                if row[2] and 'rodzajUslugi' in raw:
                    rodzaj[row[2]] = raw['rodzajUslugi']
                if row[3] and 'podrodzajUslugi' in raw:
                    podrodzaj[row[3]] = raw['podrodzajUslugi']
                if row[4] and 'formaSwiadczenia' in raw:
                    forma[row[4]] = raw['formaSwiadczenia']
                    
            except Exception as e:
                print(f"⚠️  Error parsing row: {e}")
                continue
    
    # Create final dictionaries
    dictionaries = {
        "id_kategorii_uslugi": [{"id": k, "nazwa": v} for k, v in sorted(kategoria.items())],
        "id_podkategorii_uslugi": [{"id": k, "nazwa": v} for k, v in sorted(podkategoria.items())],
        "id_rodzaju_uslugi": [{"id": k, "nazwa": v} for k, v in sorted(rodzaj.items())],
        "id_podrodzaju_uslugi": [{"id": k, "nazwa": v} for k, v in sorted(podrodzaj.items())],
        "id_formy_swiadczenia": [{"id": k, "nazwa": v} for k, v in sorted(forma.items())],
    }
    
    # Print summary
    print("📊 Extracted dictionaries:")
    for field_name, values in dictionaries.items():
        print(f"\n✅ {field_name}: {len(values)} entries")
        for item in values[:5]:
            print(f"   {item['id']}: {item['nazwa']}")
        if len(values) > 5:
            print(f"   ... and {len(values) - 5} more")
    
    # Save to JSON
    output_file = BASE_DIR / "trainings" / "bur_code_dictionaries.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved dictionaries to {output_file}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()

