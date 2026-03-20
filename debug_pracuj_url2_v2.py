import asyncio
from playwright.async_api import async_playwright


async def main():
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

        page = await context.new_page()
        url = "https://www.pracuj.pl/praca?ua=true&lang=uk"
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(4000)

        # Accept cookies
        for selector in [
            '[data-test="button-submitCookie"]',
            'button:has-text("Akceptuję")',
            '#onetrust-accept-btn-handler',
        ]:
            try:
                btn = page.locator(selector)
                if await btn.count() > 0:
                    await btn.first.click()
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                pass
        await page.wait_for_timeout(2000)

        # Check what elements with 'accordion' class exist
        result = await page.evaluate("""
            () => {
                const info = {};

                // Check accordion elements
                const accordions = document.querySelectorAll('[class*="accordion"]');
                info.accordion_count = accordions.length;
                info.accordion_samples = Array.from(accordions).slice(0, 10).map(el => ({
                    cls: el.className,
                    text_preview: el.textContent?.trim()?.substring(0, 100)
                }));

                // Check with-label-background elements
                const wlb = document.querySelectorAll('[class*="with-label-background"]');
                info.wlb_count = wlb.length;
                info.wlb_samples = Array.from(wlb).slice(0, 10).map(el => ({
                    cls: el.className,
                    text_preview: el.textContent?.trim()?.substring(0, 100)
                }));

                // Check for filter-related data-test attributes
                const dataTests = document.querySelectorAll('[data-test]');
                info.data_tests = Array.from(dataTests)
                    .filter(el => {
                        const dt = el.getAttribute('data-test');
                        return dt && (dt.includes('filter') || dt.includes('Filter'));
                    })
                    .slice(0, 20)
                    .map(el => ({
                        'data-test': el.getAttribute('data-test'),
                        cls: el.className,
                        text: el.textContent?.trim()?.substring(0, 80)
                    }));

                // Scroll down to trigger lazy loading
                return info;
            }
        """)

        print(f"Accordion elements: {result['accordion_count']}")
        for s in result['accordion_samples']:
            print(f"  [{s['cls'][:50]}] {s['text_preview']}")

        print(f"\nWith-label-background elements: {result['wlb_count']}")
        for s in result['wlb_samples']:
            print(f"  [{s['cls'][:50]}] {s['text_preview']}")

        print(f"\nFilter data-test elements: {len(result['data_tests'])}")
        for s in result['data_tests']:
            print(f"  data-test={s['data-test']} [{s['cls'][:40]}] {s['text']}")

        # Try scrolling and wait for filters to appear
        print("\n-- Scrolling page to trigger lazy load --")
        await page.evaluate("window.scrollTo(0, 500)")
        await page.wait_for_timeout(2000)

        # Try clicking on Poziom stanowiska
        print("\n-- Trying to click on Poziom stanowiska --")
        try:
            els = await page.query_selector_all('[class*="with-label-background"]')
            for el in els:
                txt = (await el.inner_text()).strip()
                print(f"  Found with-label-background text: '{txt[:60]}'")
                if 'Poziom stanowiska' in txt:
                    print(f"  -> Clicking!")
                    await el.click()
                    await page.wait_for_timeout(1500)
                    break
        except Exception as e:
            print(f"  Error: {e}")

        # After click, get filter data
        result2 = await page.evaluate("""
            (targetSections) => {
                const result = {};
                // Get ALL elements and look for section content
                for (const section of targetSections) {
                    // Find accordion divs containing this section
                    for (const el of document.querySelectorAll('[class*="accordion"], [class*="filter"], [class*="Filter"]')) {
                        const txt = el.textContent?.trim();
                        if (txt && txt.startsWith(section)) {
                            const labels = el.querySelectorAll('label, li, span, [class*="item"]');
                            const items = [];
                            for (const l of labels) {
                                const t = l.textContent?.trim();
                                if (t && t !== section && t.length > 1) {
                                    items.push(t);
                                }
                            }
                            if (items.length > 0) {
                                result[section] = { type: 'structured', items, raw: txt.substring(0, 200) };
                            } else {
                                result[section] = { type: 'raw', raw: txt.substring(0, 200) };
                            }
                        }
                    }
                }
                return result;
            }
        """, ["Poziom stanowiska", "Rodzaj umowy", "Wymiar pracy", "Tryb pracy"])

        print("\n-- After click, filter data --")
        for section, data in result2.items():
            print(f"  {section}: {data}")

        await browser.close()


asyncio.run(main())
