"""
Fetch BUR code dictionary labels by trying various naming conventions
"""
import httpx
import os
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BUR_USERNAME = os.getenv("BUR_USERNAME")
BUR_API_KEY = os.getenv("BUR_API_KEY")
BASE = "https://uslugirozwojowe.parp.gov.pl/api"

def try_fetch_dict(client, headers, dict_name, dict_type='prosty'):
    """Try to fetch a dictionary with given name"""
    try:
        url = f"{BASE}/slownik/{dict_type}/{dict_name}"
        r = client.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            # Extract list if wrapped
            if isinstance(data, dict) and 'lista' in data:
                return data['lista']
            return data
        return None
    except:
        return None

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
        
        # Try various naming conventions
        dictionaries_to_try = {
            "id_kategorii_uslugi": [
                "kategoria-uslugi", "kategoriaUslugi", "kategoria",
                "kategoria_uslugi", "kategoriaUslug"
            ],
            "id_podkategorii_uslugi": [
                "podkategoria-uslugi", "podkategoriaUslugi", "podkategoria",
                "podkategoria_uslugi"
            ],
            "id_rodzaju_uslugi": [
                "rodzaj-uslugi", "rodzajUslugi", "rodzaj",
                "rodzaj_uslugi", "rodzajUslug"
            ],
            "id_podrodzaju_uslugi": [
                "podrodzaj-uslugi", "podrodzajUslugi", "podrodzaj",
                "podrodzaj_uslugi"
            ],
            "id_formy_swiadczenia": [
                "forma-swiadczenia", "formaSwiadczenia", "forma",
                "forma_swiadczenia"
            ],
        }
        
        result = {}
        
        for field, name_variants in dictionaries_to_try.items():
            print(f"📥 Trying to fetch {field}...")
            found = False
            
            for name in name_variants:
                # Try prosty
                data = try_fetch_dict(client, headers, name, 'prosty')
                if data:
                    result[field] = data
                    print(f"   ✅ Found via /slownik/prosty/{name}")
                    print(f"      {len(data)} entries")
                    if isinstance(data, list) and data:
                        print(f"      Sample: {data[0]}")
                    found = True
                    break
                
                # Try zlozony
                data = try_fetch_dict(client, headers, name, 'zlozony')
                if data:
                    result[field] = data
                    print(f"   ✅ Found via /slownik/zlozony/{name}")
                    print(f"      {len(data)} entries")
                    if isinstance(data, list) and data:
                        print(f"      Sample: {data[0]}")
                    found = True
                    break
            
            if not found:
                print(f"   ❌ Not found with any naming variant")
                result[field] = None
            print()
    
    # Save results
    output_file = BASE_DIR / "trainings" / "bur_label_dictionaries.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for field, data in result.items():
        if data:
            count = len(data) if isinstance(data, list) else 'N/A'
            print(f"  {field}: ✅ {count} entries")
        else:
            print(f"  {field}: ❌ Not found")

if __name__ == "__main__":
    main()

