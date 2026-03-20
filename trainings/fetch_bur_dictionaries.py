"""
Fetch code dictionaries from BUR API for various ID fields
"""
import httpx
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BUR_USERNAME = os.getenv("BUR_USERNAME")
BUR_API_KEY = os.getenv("BUR_API_KEY")
BASE = "https://uslugirozwojowe.parp.gov.pl/api"

def main():
    # Login to get token
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
        print("✅ Logged in successfully\n")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Dictionary names to fetch using /slownik/prosty/{slownik}
        # Try both prosty (simple) and zlozony (complex) endpoints
        dictionaries_to_fetch = [
            ("id_kategorii_uslugi", "kategoria-uslugi"),
            ("id_podkategorii_uslugi", "podkategoria-uslugi"),
            ("id_rodzaju_uslugi", "rodzaj-uslugi"),
            ("id_podrodzaju_uslugi", "podrodzaj-uslugi"),
            ("id_formy_swiadczenia", "forma-swiadczenia"),
        ]
        
        dictionaries = {}
        
        for dict_name, slownik_name in dictionaries_to_fetch:
            print(f"📥 Fetching {dict_name} ({slownik_name})...")
            
            # Try both /slownik/prosty and /slownik/zlozony
            for endpoint_type in ['prosty', 'zlozony']:
                if dict_name in dictionaries and dictionaries[dict_name]:
                    break  # Already found
                    
                try:
                    url = f"{BASE}/slownik/{endpoint_type}/{slownik_name}"
                    print(f"   Trying /{endpoint_type}/{slownik_name}...")
                    r = client.get(url, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                    
                    # Check if it's a list or has a 'lista' key
                    if isinstance(data, list):
                        dictionaries[dict_name] = data
                    elif isinstance(data, dict) and 'lista' in data:
                        dictionaries[dict_name] = data['lista']
                    else:
                        dictionaries[dict_name] = data
                    
                    count = len(dictionaries[dict_name]) if isinstance(dictionaries[dict_name], list) else '?'
                    print(f"   ✅ Found {count} entries via /{endpoint_type}/")
                    
                    # Show first few entries
                    if isinstance(dictionaries[dict_name], list) and dictionaries[dict_name]:
                        print(f"   Sample: {dictionaries[dict_name][0]}")
                    
                    break  # Success, move to next dictionary
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        print(f"   ⚠️  Not found via /{endpoint_type}/")
                    else:
                        print(f"   ❌ HTTP Error {e.response.status_code}: {e.response.text[:100]}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            # If neither endpoint worked, mark as None
            if dict_name not in dictionaries:
                dictionaries[dict_name] = None
                print(f"   ❌ Failed to fetch {dict_name}")
            
            print()
    
    # Save to JSON file
    output_file = BASE_DIR / "trainings" / "bur_dictionaries.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionaries, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved dictionaries to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dict_name, data in dictionaries.items():
        if data is not None:
            count = len(data) if isinstance(data, list) else 'N/A'
            print(f"  {dict_name}: {count} entries")
        else:
            print(f"  {dict_name}: Failed to fetch")

if __name__ == "__main__":
    main()

