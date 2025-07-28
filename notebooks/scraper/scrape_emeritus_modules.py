
import asyncio
from playwright.async_api import async_playwright
import pandas as pd

async def scrape_emeritus_program():
    async with async_playwright() as p:
        user_data_dir = "playwright_user_data"

        browser = await p.chromium.launch_persistent_context(
            user_data_dir,
            headless=False
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()

        await page.goto("https://classroom.emeritus.org/courses/12959/pages/program-outline-professional-certificate-in-machine-learning-and-artificial-intelligence?module_item_id=2328817")
        await page.wait_for_timeout(5000)

        # TODO: Update selectors after inspecting the actual page
        modules = await page.query_selector_all("h3, h4, ul")

        extracted = []
        current_module = {"Module": None, "Learning Outcomes": "", "Key Activities": ""}

        for el in modules:
            tag = await el.evaluate("el => el.tagName")
            text = (await el.inner_text()).strip()

            if tag == "H3":
                if current_module["Module"]:
                    extracted.append(current_module)
                current_module = {"Module": text, "Learning Outcomes": "", "Key Activities": ""}

            elif tag == "H4" and "Learning Outcomes" in text:
                sibling = await el.evaluate_handle("el => el.nextElementSibling")
                if sibling:
                    current_module["Learning Outcomes"] = (await sibling.inner_text()).strip()

            elif tag == "H4" and "Key Activities" in text:
                sibling = await el.evaluate_handle("el => el.nextElementSibling")
                if sibling:
                    current_module["Key Activities"] = (await sibling.inner_text()).strip()

        if current_module["Module"]:
            extracted.append(current_module)

        await browser.close()

        df = pd.DataFrame(extracted)
        df.to_csv("modules_scraped.csv", index=False)
        print("Scraping complete. Data saved to modules_scraped.csv")

if __name__ == "__main__":
    asyncio.run(scrape_emeritus_program())
