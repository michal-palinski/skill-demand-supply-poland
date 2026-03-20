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
        print(f"Loading: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

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

        # Extract all text that might be section headers
        result = await page.evaluate("""
            () => {
                // Find all h2, h3, h4, legend elements and their text
                const headers = [];
                for (const el of document.querySelectorAll('h1,h2,h3,h4,h5,legend,span[class*="title"],div[class*="title"]')) {
                    const txt = el.textContent?.trim();
                    if (txt && txt.length < 80) {
                        headers.push({tag: el.tagName, cls: el.className, text: txt});
                    }
                }
                return headers.slice(0, 100);
            }
        """)

        print("\nAll headers found on page:")
        for h in result:
            print(f"  <{h['tag']} class='{h['cls'][:50]}'> {h['text']}")

        # Also check if page redirected
        print(f"\nFinal URL: {page.url}")
        print(f"Page title: {await page.title()}")

        # Check for filter-related text
        filter_texts = await page.evaluate("""
            () => {
                const keywords = ['stanowiska', 'umowy', 'pracy', 'tryb', 'poziom', 'wymiar'];
                const found = [];
                for (const el of document.querySelectorAll('*')) {
                    const txt = el.textContent?.trim();
                    if (!txt || txt.length > 100) continue;
                    if (keywords.some(k => txt.toLowerCase().includes(k))) {
                        found.push({tag: el.tagName, text: txt, cls: el.className?.substring?.(0, 40) || ''});
                    }
                }
                // Deduplicate
                const seen = new Set();
                return found.filter(f => {
                    if (seen.has(f.text)) return false;
                    seen.add(f.text);
                    return true;
                }).slice(0, 50);
            }
        """)

        print("\nElements with filter-related keywords:")
        for f in filter_texts:
            print(f"  <{f['tag']} cls='{f['cls']}'> {f['text']}")

        await browser.close()


asyncio.run(main())
