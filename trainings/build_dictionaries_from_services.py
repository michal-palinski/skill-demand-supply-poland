"""
Build code dictionaries by extracting labels from actual service records
"""
import httpx
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BUR_USERNAME = os.getenv("BUR_USERNAME")
BUR_API_KEY = os.getenv("BUR_API_KEY")
BASE = "https://uslugirozwojowe.parp.gov.pl/api"

def main():
    print("🔐 Logging in to BUR API...")
    with httpx.Client(verify=False, timeout=30.0) as client:
        r = client.post(
            f"{BASE}/autoryzacja/logowanie",
            json={
                "nazwaUzytkownika": BUR_USERNAME,
                "kluczAutoryzacyjny": BUR_API_KEY,
            }
        )
        r.raise_for_status()
        token = r.json()["token"]
        print("✅ Logged in\n")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Fetch services and extract category info
        print("📥 Fetching service records to extract labels...")
        
        kategoria = {}
        podkategoria = {}
        rodzaj = {}
        podrodzaj = {}
        forma = {}
        
        # Fetch multiple pages to get good coverage
        for page in range(1, 100):  # Fetch first 100 pages
            try:
                r = client.get(
                    f"{BASE}/usluga",
                    params={"strona": page},
                    headers=headers,
                )
                
                if r.status_code != 200:
                    break
                
                data = r.json()
                items = data.get("lista", [])
                
                if not items:
                    break
                
                # Extract labels from each service
                for service in items:
                    # Category
                    if service.get('idKategoriiUslugi') and service.get('kategoriaUslugi'):
                        kategoria[service['idKategoriiUslugi']] = service['kategoriaUslugi']
                    
                    # Subcategory
                    if service.get('idPodkategoriiUslugi') and service.get('podkategoriaUslugi'):
                        podkategoria[service['idPodkategoriiUslugi']] = service['podkategoriaUslugi']
                    
                    # Type
                    if service.get('idRodzajuUslugi') and service.get('rodzajUslugi'):
                        rodzaj[service['idRodzajuUslugi']] = service['rodzajUslugi']
                    
                    # Subtype
                    if service.get('idPodrodzajuUslugi') and service.get('podrodzajUslugi'):
                        podrodzaj[service['idPodrodzajuUslugi']] = service['podrodzajUslugi']
                    
                    # Form
                    if service.get('idFormySwiadczenia') and service.get('formaSwiadczenia'):
                        forma[service['idFormySwiadczenia']] = service['formaSwiadczenia']
                
                print(f"   Page {page}: {len(items)} services | "
                      f"Categories: {len(kategoria)}, "
                      f"Subcategories: {len(podkategoria)}, "
                      f"Types: {len(rodzaj)}, "
                      f"Subtypes: {len(podrodzaj)}, "
                      f"Forms: {len(forma)}")
                
                # Check if we have all the codes we need
                # From bur_id_codes.json we know the counts
                if (len(kategoria) >= 12 and len(podkategoria) >= 93 and 
                    len(rodzaj) >= 2 and len(podrodzaj) >= 7 and len(forma) >= 6):
                    print(f"\n✅ Got all expected codes! Stopping at page {page}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Error on page {page}: {e}")
                break
        
        # Create final dictionaries
        dictionaries = {
            "id_kategorii_uslugi": [{"id": k, "label": v} for k, v in sorted(kategoria.items())],
            "id_podkategorii_uslugi": [{"id": k, "label": v} for k, v in sorted(podkategoria.items())],
            "id_rodzaju_uslugi": [{"id": k, "label": v} for k, v in sorted(rodzaj.items())],
            "id_podrodzaju_uslugi": [{"id": k, "label": v} for k, v in sorted(podrodzaj.items())],
            "id_formy_swiadczenia": [{"id": k, "label": v} for k, v in sorted(forma.items())],
        }
        
        # Print results
        print("\n" + "="*60)
        print("EXTRACTED DICTIONARIES")
        print("="*60)
        
        for field_name, values in dictionaries.items():
            print(f"\n✅ {field_name}: {len(values)} entries")
            for item in values[:10]:
                print(f"   {item['id']}: {item['label']}")
            if len(values) > 10:
                print(f"   ... and {len(values) - 10} more")
        
        # Save to JSON
        output_file = BASE_DIR / "trainings" / "bur_code_labels.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dictionaries, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved dictionaries to {output_file}")

if __name__ == "__main__":
    main()

