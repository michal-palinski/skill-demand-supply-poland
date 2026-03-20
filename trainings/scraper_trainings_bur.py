import httpx
import asyncio
from pathlib import Path
import os
import asyncpg
import json
from datetime import datetime
from dotenv import load_dotenv
import time
from tqdm import tqdm
import sys

# path to project root (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BUR_USERNAME = os.getenv("BUR_USERNAME")
BUR_API_KEY = os.getenv("BUR_API_KEY")
BASE = "https://uslugirozwojowe.parp.gov.pl/api"

# PostgreSQL credentials
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trainings_pl")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

async def create_table(conn):
    """Create the services table if it doesn't exist"""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS bur_services (
            id INTEGER PRIMARY KEY,
            numer VARCHAR(100),
            tytul TEXT,
            status VARCHAR(50),
            data_rozpoczecia_uslugi TIMESTAMP,
            data_zakonczenia_uslugi TIMESTAMP,
            data_zakonczenia_rekrutacji TIMESTAMP,
            
            -- Categories and types
            id_kategorii_uslugi INTEGER,
            id_podkategorii_uslugi INTEGER,
            id_rodzaju_uslugi INTEGER,
            id_podrodzaju_uslugi INTEGER,
            id_formy_swiadczenia INTEGER,
            
            -- Qualifications
            id_kwalifikacji_kkz INTEGER,
            id_kwalifikacji_zrk INTEGER,
            
            -- Content fields
            cel_biznesowy TEXT,
            cel_edukacyjny TEXT,
            efekt_uslugi TEXT,
            efekty_uczenia_sie TEXT,
            program_uslugi TEXT,
            grupa_docelowa TEXT,
            warunki_uczestnictwa TEXT,
            warunki_techniczne TEXT,
            informacje_dodatkowe TEXT,
            
            -- Pricing (stored in grosze/cents)
            cena_brutto_za_usluge BIGINT,
            cena_netto_za_usluge BIGINT,
            cena_brutto_za_uczestnika BIGINT,
            cena_netto_za_uczestnika BIGINT,
            cena_brutto_za_godzine BIGINT,
            cena_netto_za_godzine BIGINT,
            czy_cena_dotyczy_calej_uslugi BOOLEAN,
            
            -- Capacity
            liczba_godzin INTEGER,
            minimalna_liczba_uczestnikow INTEGER,
            maksymalna_liczba_uczestnikow INTEGER,
            
            -- Complex fields stored as JSONB
            adres JSONB,
            dostawca_uslug JSONB,
            osoba_kontaktowa JSONB,
            ocena JSONB,
            lista_projektow JSONB,
            warunki_logistyczne JSONB,
            formy_dofinansowania JSONB,
            kwalifikacje_kuz JSONB,
            inne_kwalifikacje JSONB,
            
            -- Full raw data
            raw_data JSONB,
            
            -- Metadata
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_bur_services_status ON bur_services(status);
        CREATE INDEX IF NOT EXISTS idx_bur_services_kategoria ON bur_services(id_kategorii_uslugi);
        CREATE INDEX IF NOT EXISTS idx_bur_services_data_rozpoczecia ON bur_services(data_rozpoczecia_uslugi);
    """)
    print("✓ Table created/verified")


def parse_datetime(date_str):
    """Parse ISO8601 date string to datetime object, converting to UTC and removing timezone"""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        # Convert to UTC and make timezone-naive for PostgreSQL TIMESTAMP (without time zone)
        if dt.tzinfo is not None:
            dt = dt.astimezone(None).replace(tzinfo=None)  # Convert to local time and remove tzinfo
        return dt
    except (ValueError, TypeError):
        return None


async def insert_service(conn, service):
    """Insert or update a service in the database"""
    await conn.execute("""
        INSERT INTO bur_services (
            id, numer, tytul, status,
            data_rozpoczecia_uslugi, data_zakonczenia_uslugi, data_zakonczenia_rekrutacji,
            id_kategorii_uslugi, id_podkategorii_uslugi, id_rodzaju_uslugi,
            id_podrodzaju_uslugi, id_formy_swiadczenia,
            id_kwalifikacji_kkz, id_kwalifikacji_zrk,
            cel_biznesowy, cel_edukacyjny, efekt_uslugi, efekty_uczenia_sie,
            program_uslugi, grupa_docelowa, warunki_uczestnictwa, warunki_techniczne,
            informacje_dodatkowe,
            cena_brutto_za_usluge, cena_netto_za_usluge,
            cena_brutto_za_uczestnika, cena_netto_za_uczestnika,
            cena_brutto_za_godzine, cena_netto_za_godzine,
            czy_cena_dotyczy_calej_uslugi,
            liczba_godzin, minimalna_liczba_uczestnikow, maksymalna_liczba_uczestnikow,
            adres, dostawca_uslug, osoba_kontaktowa, ocena,
            lista_projektow, warunki_logistyczne, formy_dofinansowania,
            kwalifikacje_kuz, inne_kwalifikacje,
            raw_data, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
            $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26,
            $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38,
            $39, $40, $41, $42, $43, CURRENT_TIMESTAMP
        )
        ON CONFLICT (id) DO UPDATE SET
            numer = EXCLUDED.numer,
            tytul = EXCLUDED.tytul,
            status = EXCLUDED.status,
            data_rozpoczecia_uslugi = EXCLUDED.data_rozpoczecia_uslugi,
            data_zakonczenia_uslugi = EXCLUDED.data_zakonczenia_uslugi,
            data_zakonczenia_rekrutacji = EXCLUDED.data_zakonczenia_rekrutacji,
            id_kategorii_uslugi = EXCLUDED.id_kategorii_uslugi,
            id_podkategorii_uslugi = EXCLUDED.id_podkategorii_uslugi,
            id_rodzaju_uslugi = EXCLUDED.id_rodzaju_uslugi,
            id_podrodzaju_uslugi = EXCLUDED.id_podrodzaju_uslugi,
            id_formy_swiadczenia = EXCLUDED.id_formy_swiadczenia,
            id_kwalifikacji_kkz = EXCLUDED.id_kwalifikacji_kkz,
            id_kwalifikacji_zrk = EXCLUDED.id_kwalifikacji_zrk,
            cel_biznesowy = EXCLUDED.cel_biznesowy,
            cel_edukacyjny = EXCLUDED.cel_edukacyjny,
            efekt_uslugi = EXCLUDED.efekt_uslugi,
            efekty_uczenia_sie = EXCLUDED.efekty_uczenia_sie,
            program_uslugi = EXCLUDED.program_uslugi,
            grupa_docelowa = EXCLUDED.grupa_docelowa,
            warunki_uczestnictwa = EXCLUDED.warunki_uczestnictwa,
            warunki_techniczne = EXCLUDED.warunki_techniczne,
            informacje_dodatkowe = EXCLUDED.informacje_dodatkowe,
            cena_brutto_za_usluge = EXCLUDED.cena_brutto_za_usluge,
            cena_netto_za_usluge = EXCLUDED.cena_netto_za_usluge,
            cena_brutto_za_uczestnika = EXCLUDED.cena_brutto_za_uczestnika,
            cena_netto_za_uczestnika = EXCLUDED.cena_netto_za_uczestnika,
            cena_brutto_za_godzine = EXCLUDED.cena_brutto_za_godzine,
            cena_netto_za_godzine = EXCLUDED.cena_netto_za_godzine,
            czy_cena_dotyczy_calej_uslugi = EXCLUDED.czy_cena_dotyczy_calej_uslugi,
            liczba_godzin = EXCLUDED.liczba_godzin,
            minimalna_liczba_uczestnikow = EXCLUDED.minimalna_liczba_uczestnikow,
            maksymalna_liczba_uczestnikow = EXCLUDED.maksymalna_liczba_uczestnikow,
            adres = EXCLUDED.adres,
            dostawca_uslug = EXCLUDED.dostawca_uslug,
            osoba_kontaktowa = EXCLUDED.osoba_kontaktowa,
            ocena = EXCLUDED.ocena,
            lista_projektow = EXCLUDED.lista_projektow,
            warunki_logistyczne = EXCLUDED.warunki_logistyczne,
            formy_dofinansowania = EXCLUDED.formy_dofinansowania,
            kwalifikacje_kuz = EXCLUDED.kwalifikacje_kuz,
            inne_kwalifikacje = EXCLUDED.inne_kwalifikacje,
            raw_data = EXCLUDED.raw_data,
            updated_at = CURRENT_TIMESTAMP
    """,
        service['id'],
        service.get('numer'),
        service.get('tytul'),
        service.get('status'),
        parse_datetime(service.get('dataRozpoczeciaUslugi')),
        parse_datetime(service.get('dataZakonczeniaUslugi')),
        parse_datetime(service.get('dataZakonczeniaRekrutacji')),
        service.get('idKategoriiUslugi'),
        service.get('idPodkategoriiUslugi'),
        service.get('idRodzajuUslugi'),
        service.get('idPodrodzajuUslugi'),
        service.get('idFormySwiadczenia'),
        service.get('idKwalifikacjiKkz'),
        service.get('idKwalifikacjiZrk'),
        service.get('celBiznesowy'),
        service.get('celEdukacyjny'),
        service.get('efektUslugi'),
        service.get('efektyUczeniaSie'),
        service.get('programUslugi'),
        service.get('grupaDocelowa'),
        service.get('warunkiUczestnictwa'),
        service.get('warunkiTechniczne'),
        service.get('informacjeDodatkowe'),
        service.get('cenaBruttoZaUsluge'),
        service.get('cenaNettoZaUsluge'),
        service.get('cenaBruttoZaUczestnika'),
        service.get('cenaNettoZaUczestnika'),
        service.get('cenaBruttoZaGodzine'),
        service.get('cenaNettoZaGodzine'),
        service.get('czyCenaDotyczyCalejUslugi'),
        service.get('liczbaGodzin'),
        service.get('minimalnaLiczbaUczestnikow'),
        service.get('maksymalnaLiczbaUczestnikow'),
        json.dumps(service.get('adres')),
        json.dumps(service.get('dostawcaUslug')),
        json.dumps(service.get('osobaKontaktowa')),
        json.dumps(service.get('ocena')),
        json.dumps(service.get('listaProjektow', [])),
        json.dumps(service.get('warunkiLogistyczne', [])),
        json.dumps(service.get('formyDofinansowania', [])),
        json.dumps(service.get('kwalifikacjeKuz', [])),
        json.dumps(service.get('inneKwalifikacje', [])),
        json.dumps(service),
    )


async def login_to_api(client, max_retries=3):
    """Login to BUR API and return auth token with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"🔐 Attempting login (attempt {attempt + 1}/{max_retries})...")
            r = await client.post(
                f"{BASE}/autoryzacja/logowanie",
                json={
                    "nazwaUzytkownika": BUR_USERNAME,
                    "kluczAutoryzacyjny": BUR_API_KEY,
                },
                timeout=60.0  # Longer timeout for login
            )
            print(f"📡 API Response - Status: {r.status_code}")
            print(f"📡 Headers: {dict(r.headers)}")
            
            if r.status_code != 200:
                print(f"📡 Response Body: {r.text[:500]}")  # First 500 chars
            
            r.raise_for_status()
            return r.json()["token"]
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"📡 Full Response: {e.response.text[:1000]}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Login failed after {max_retries} attempts")
                raise
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
            print(f"❌ Network Error: {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Login failed after {max_retries} attempts")
                raise


async def main():
    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    try:
        print("✓ Connected to PostgreSQL")
        
        # Create table
        await create_table(conn)
        
        # Fetch and save services
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            # Initial login
            token = await login_to_api(client)
            print("✓ Logged in to BUR API")

            headers = {"Authorization": f"Bearer {token}"}
            
            # Check if we're in retry mode
            failed_pages_file = BASE_DIR / "failed_pages.json"
            retry_mode = "--retry-failed" in sys.argv
            pages_to_scrape = None
            
            if retry_mode and failed_pages_file.exists():
                try:
                    failed_pages_data = json.loads(failed_pages_file.read_text())
                    pages_to_scrape = sorted([p['page'] for p in failed_pages_data])
                    print(f"🔄 RETRY MODE: Will retry {len(pages_to_scrape)} failed pages")
                    print(f"   Pages: {pages_to_scrape[:10]}" + (' ...' if len(pages_to_scrape) > 10 else ''))
                except Exception as e:
                    print(f"⚠️  Could not load failed pages: {e}")
                    retry_mode = False
            elif retry_mode:
                print("⚠️  No failed pages found to retry")
                retry_mode = False
            
            # Try to resume from checkpoint (only in normal mode)
            checkpoint_file = BASE_DIR / "scraper_checkpoint.txt"
            start_page = 1
            if not retry_mode and checkpoint_file.exists():
                try:
                    start_page = int(checkpoint_file.read_text().strip())
                    print(f"📍 Resuming from page {start_page} (found checkpoint)")
                except:
                    pass
            
            page = start_page
            total_saved = 0
            errors = []
            retry_count = 0
            max_retries = 3
            completed_successfully = False  # Flag to track normal completion
            
            # Create progress bar
            if retry_mode:
                pbar = tqdm(total=len(pages_to_scrape), desc="Retrying failed pages", unit=" pages", colour="yellow")
                page_iterator = iter(pages_to_scrape)
            else:
                pbar = tqdm(desc="Scraping services", unit=" services", colour="green")
                page_iterator = None
            
            try:
                while True:
                    # Get next page to process
                    if retry_mode:
                        try:
                            page = next(page_iterator)
                        except StopIteration:
                            completed_successfully = True
                            break
                    elif page_iterator is None:
                        pass  # Normal mode, page is already set
                    try:
                        # Add delay between requests (respectful to the API)
                        if page > 1:
                            await asyncio.sleep(1)  # 1 second delay between pages
                        
                        r = await client.get(
                            f"{BASE}/usluga",
                            params={"strona": page},
                            headers=headers,
                        )
                        
                        # Log response details for debugging
                        if r.status_code != 200:
                            pbar.write(f"📡 Page {page} - Status: {r.status_code}")
                            pbar.write(f"📡 Headers: {dict(r.headers)}")
                            pbar.write(f"📡 Response: {r.text[:500]}")
                        
                        # Check for rate limiting
                        if r.status_code == 429:
                            retry_count += 1
                            if retry_count > max_retries:
                                error_msg = "Rate limit exceeded after max retries"
                                errors.append({"page": page, "error": error_msg})
                                pbar.write(f"⚠️  {error_msg}")
                                break
                            
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            pbar.write(f"⏳ Rate limited. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        r.raise_for_status()
                        data = r.json()
                        retry_count = 0  # Reset retry count on success

                        items = data.get("lista", [])
                        if not items:
                            # Reached the end - mark as successfully completed
                            completed_successfully = True
                            break

                        # Save each service to database
                        for service in items:
                            try:
                                await insert_service(conn, service)
                                total_saved += 1
                                pbar.update(1)
                                pbar.set_postfix({
                                    "page": page,
                                    "errors": len(errors)
                                })
                            except Exception as e:
                                error_msg = f"Service ID {service.get('id', 'unknown')}: {str(e)[:50]}"
                                errors.append({"service_id": service.get('id'), "error": str(e)})
                                pbar.write(f"❌ Error: {error_msg}")
                        
                        # Save checkpoint after each successful page (only in normal mode)
                        if not retry_mode:
                            page += 1
                            checkpoint_file.write_text(str(page))
                        
                        # Update progress bar for retry mode
                        if retry_mode:
                            pbar.update(1)
                        
                        # Optional: limit for testing
                        # if page > 2:
                        #     break
                    
                    except httpx.HTTPStatusError as e:
                        # Log detailed error information
                        pbar.write(f"❌ HTTP Error on page {page}: {e.response.status_code}")
                        pbar.write(f"📡 Response: {e.response.text[:500]}")
                        
                        if e.response.status_code == 429:
                            retry_count += 1
                            if retry_count > max_retries:
                                error_msg = "Rate limit exceeded after max retries"
                                errors.append({"page": page, "error": error_msg})
                                pbar.write(f"⚠️  {error_msg}")
                                break
                            wait_time = 2 ** retry_count
                            pbar.write(f"⏳ Rate limited. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        elif e.response.status_code == 401:
                            # Token expired - refresh it
                            pbar.write(f"🔄 Token expired on page {page}, refreshing...")
                            try:
                                token = await login_to_api(client)
                                headers = {"Authorization": f"Bearer {token}"}
                                pbar.write(f"✓ Token refreshed successfully")
                                # Retry the same page
                                continue
                            except Exception as login_error:
                                error_msg = f"Failed to refresh token: {login_error}"
                                errors.append({"page": page, "error": error_msg})
                                pbar.write(f"❌ {error_msg}")
                                raise
                        elif e.response.status_code >= 500:
                            # Server error (500-599) - retry with backoff
                            error_type = {
                                500: "Internal Server Error",
                                502: "Bad Gateway",
                                503: "Service Unavailable",
                                504: "Gateway Timeout"
                            }.get(e.response.status_code, "Server Error")
                            
                            retry_count += 1
                            if retry_count > max_retries:
                                error_msg = f"HTTP {e.response.status_code} ({error_type}) on page {page} - skipping after {max_retries} retries"
                                errors.append({"page": page, "error": error_msg})
                                pbar.write(f"⚠️  {error_msg}")
                                # Save checkpoint and skip to next page
                                page += 1
                                checkpoint_file.write_text(str(page))
                                retry_count = 0
                                continue
                            wait_time = 2 ** retry_count
                            pbar.write(f"⚠️  HTTP {e.response.status_code} ({error_type}) on page {page}. Retry {retry_count}/{max_retries} in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Client error (4xx other than 401/429) - skip page
                            error_msg = f"HTTP {e.response.status_code} on page {page} - skipping"
                            errors.append({"page": page, "error": error_msg})
                            pbar.write(f"❌ {error_msg}")
                            # Save checkpoint and skip to next page
                            page += 1
                            checkpoint_file.write_text(str(page))
                            continue
                    
                    except Exception as e:
                        error_msg = f"Page {page}: {str(e)[:100]}"
                        errors.append({"page": page, "error": str(e)})
                        pbar.write(f"❌ Error: {error_msg}")
                        # Continue to next page instead of crashing
                        page += 1
                        continue
            
            finally:
                pbar.close()
                # Clean up checkpoint ONLY on successful completion (not on crash)
                if checkpoint_file.exists() and completed_successfully:
                    checkpoint_file.unlink()
                    print("🗑️  Checkpoint cleared (scraping completed successfully)")
                elif checkpoint_file.exists():
                    print(f"💾 Checkpoint saved at page {page} (can resume later)")
            
            # Handle failed pages
            failed_pages_file = BASE_DIR / "failed_pages.json"
            failed_pages = [err for err in errors if 'page' in err]
            
            if retry_mode:
                # In retry mode, remove successfully processed pages from failed list
                if failed_pages_file.exists():
                    try:
                        all_failed = json.loads(failed_pages_file.read_text())
                        # Remove pages that didn't error this time
                        new_failed_page_nums = {p['page'] for p in failed_pages}
                        successfully_retried = [p for p in pages_to_scrape if p not in new_failed_page_nums]
                        
                        # Keep only pages that still fail
                        all_failed = [p for p in all_failed if p['page'] not in successfully_retried]
                        
                        # Add new failures
                        existing_page_nums = {p['page'] for p in all_failed}
                        for new_fail in failed_pages:
                            if new_fail['page'] not in existing_page_nums:
                                all_failed.append(new_fail)
                        
                        if all_failed:
                            failed_pages_file.write_text(json.dumps(all_failed, indent=2))
                            print(f"✅ Successfully retried {len(successfully_retried)} pages")
                            print(f"❌ Still failing: {len(all_failed)} pages")
                        else:
                            failed_pages_file.unlink()
                            print(f"🎉 All failed pages successfully retried!")
                    except Exception as e:
                        print(f"⚠️  Error updating failed pages: {e}")
            elif failed_pages:
                # Normal mode - save new failed pages
                existing_failed = []
                if failed_pages_file.exists():
                    try:
                        existing_failed = json.loads(failed_pages_file.read_text())
                    except:
                        pass
                
                # Merge with new failed pages (avoid duplicates)
                existing_page_nums = {p['page'] for p in existing_failed}
                new_failed = [p for p in failed_pages if p['page'] not in existing_page_nums]
                all_failed = existing_failed + new_failed
                
                # Save to file
                failed_pages_file.write_text(json.dumps(all_failed, indent=2))
                print(f"💾 Saved {len(new_failed)} new failed pages to {failed_pages_file.name}")
                print(f"   Total failed pages on record: {len(all_failed)}")
            
            # Summary
            print(f"\n{'='*60}")
            print(f"✓ Scraping complete!")
            print(f"  Total services saved: {total_saved}")
            print(f"  Total pages processed: {page - 1}")
            print(f"  Total errors: {len(errors)}")
            print(f"  Failed pages: {len(failed_pages)}")
            
            if errors:
                print(f"\n⚠️  Errors encountered:")
                for i, err in enumerate(errors[:10], 1):  # Show first 10 errors
                    if 'service_id' in err:
                        print(f"  {i}. Service {err['service_id']}: {err['error'][:80]}")
                    else:
                        print(f"  {i}. Page {err.get('page', '?')}: {err['error'][:80]}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
                
                if failed_pages:
                    print(f"\n💡 To retry failed pages, run:")
                    print(f"   python trainings/scraper_trainings_bur.py --retry-failed")
            print(f"{'='*60}")
            
    finally:
        await conn.close()
        print("✓ Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())

