import asyncio
import re
import pandas as pd
from playwright.async_api import async_playwright

URLS = {
    "pracuj_lang_uk": "https://www.pracuj.pl/praca?lang=uk",
    "pracuj_ua_true": "https://www.pracuj.pl/praca?ua=true&lang=uk",
}

TARGET_SECTIONS = [
    "Poziom stanowiska",
    "Rodzaj umowy",
    "Wymiar pracy",
    "Tryb pracy",
]


def parse_item(raw_text):
    """Parse 'label text (count)' into (label, count)."""
    raw_text = raw_text.strip()
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    if not lines:
        return None, None
    combined = ' '.join(lines)
    match = re.search(r'\((\d[\d\s\u00a0]*)\)\s*$', combined)
    if match:
        count_str = re.sub(r'[\s\u00a0]', '', match.group(1))
        try:
            count = int(count_str)
        except ValueError:
            count = None
        label = combined[:match.start()].strip()
        return label, count
    return combined, None


def parse_accordion_raw_text(section_name, full_text):
    """
    Parse raw accordion text such as:
    'Wymiar pracyczęść etatu(874)dodatkowa / tymczasowa(339)pełny etat(4630)'
    into a list of 'label (count)' strings.
    """
    text = full_text.strip()
    if text.startswith(section_name):
        text = text[len(section_name):]

    pattern = r'([^\(\)]+?)\((\d[\d\s\u00a0]*)\)'
    items = []
    seen = set()
    for m in re.finditer(pattern, text):
        label = m.group(1).strip()
        count_str = re.sub(r'[\s\u00a0]', '', m.group(2))
        try:
            count = int(count_str)
        except ValueError:
            count = None
        if label and label not in seen:
            seen.add(label)
            items.append(f"{label} ({count})" if count is not None else label)
    return items


async def accept_cookies(page):
    for selector in [
        '[data-test="button-submitCookie"]',
        'button:has-text("Akceptuję")',
        'button:has-text("Akceptuj")',
        '#onetrust-accept-btn-handler',
    ]:
        try:
            btn = page.locator(selector)
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(1500)
                return
        except Exception:
            pass


async def close_any_popup(page):
    """Dismiss any popup blocking the page."""
    for selector in [
        '[class*="popup"] button[class*="close"]',
        '[class*="modal"] button[class*="close"]',
        '[class*="popup_p"]',
        'button[aria-label="Close"]',
        'button[aria-label="Zamknij"]',
    ]:
        try:
            el = page.locator(selector)
            if await el.count() > 0:
                await el.first.click(timeout=3000)
                await page.wait_for_timeout(500)
        except Exception:
            pass
    # Press Escape as a catch-all
    try:
        await page.keyboard.press('Escape')
        await page.wait_for_timeout(300)
    except Exception:
        pass


async def scrape_filters(page, url):
    print(f"\nScraping: {url}")
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(4000)
    await accept_cookies(page)
    await page.wait_for_timeout(2000)
    await close_any_popup(page)
    await page.wait_for_timeout(500)

    # --- Strategy: extract data from accordion elements (raw text) ---
    # Works for both URL structures because:
    #   - URL1: standard filter panel also uses accordion-like elements
    #   - URL2: accordion class elements with raw concatenated text
    section_data = await page.evaluate("""
        (targetSections) => {
            const result = {};

            // PRIMARY: Look for elements whose raw text starts with a section name
            // These can be accordion divs or any container
            const candidates = document.querySelectorAll(
                '[class*="accordion"], [class*="filter"], [class*="Filter"], section, fieldset, [data-test*="filter"]'
            );

            for (const el of candidates) {
                const txt = el.textContent?.trim();
                if (!txt) continue;
                for (const section of targetSections) {
                    if (result[section]) continue;
                    if (!txt.startsWith(section)) continue;
                    // Must contain at least one count in parentheses
                    if (!txt.match(/\\(\\d+\\)/)) continue;
                    result[section] = { raw: txt.substring(0, 2000) };
                    break;
                }
            }

            // SECONDARY: For any still missing, try standard label-based approach
            for (const section of targetSections) {
                if (result[section]) continue;
                const allEls = document.querySelectorAll('h2,h3,h4,h5,legend,span,div,p');
                for (const el of allEls) {
                    const txt = el.textContent?.trim();
                    if (txt !== section) continue;
                    let container = el;
                    for (let i = 0; i < 6; i++) {
                        container = container.parentElement;
                        if (!container) break;
                        const items = container.querySelectorAll('label, li');
                        if (items.length > 1) break;
                    }
                    if (!container) continue;
                    const items = [];
                    for (const c of container.querySelectorAll('label, li, [role="option"]')) {
                        const t = c.textContent?.trim();
                        if (t && t !== section && t.length > 1 && !targetSections.includes(t)) {
                            if (!items.includes(t)) items.push(t);
                        }
                    }
                    if (items.length > 0) {
                        result[section] = { structured: items };
                        break;
                    }
                }
            }
            return result;
        }
    """, TARGET_SECTIONS)

    # Parse raw text results
    parsed = {}
    for section_name in TARGET_SECTIONS:
        val = section_data.get(section_name)
        if not val:
            continue
        if 'raw' in val:
            items = parse_accordion_raw_text(section_name, val['raw'])
            if items:
                parsed[section_name] = items
        elif 'structured' in val:
            parsed[section_name] = val['structured']

    print(f"  Sections found: {list(parsed.keys())}")
    for k, v in parsed.items():
        print(f"    {k}: {len(v)} items")
    return parsed


async def main():
    all_data = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="pl-PL",
            viewport={"width": 1400, "height": 900},
        )

        for url_key, url in URLS.items():
            page = await context.new_page()
            section_data = await scrape_filters(page, url)
            all_data[url_key] = section_data
            await page.close()

        await browser.close()

    # Build Excel — one sheet per URL
    output_path = "pracuj_filters.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for url_key, section_data in all_data.items():
            rows = []
            for section_name in TARGET_SECTIONS:
                items = section_data.get(section_name, [])
                for raw_item in items:
                    label, count = parse_item(raw_item)
                    if label:
                        rows.append({
                            "Kategoria": section_name,
                            "Opcja": label,
                            "Liczba ofert": count,
                        })

            sheet_name = url_key[:31]
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Kategoria", "Opcja", "Liczba ofert"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"\nSheet '{sheet_name}': {len(df)} rows")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
