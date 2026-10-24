"""Screenshots of the website for the report -> report/images/.

Needs: Neo4j running, the API on :8000 and the site on :5173 (see README).
  python report/take_screenshots.py
"""
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "images"
SITE = "http://localhost:5173/#"
WAIT = 1500


def page_shot(page, url, name, full=False, before=None):
    page.goto(url)
    page.wait_for_timeout(WAIT)
    if before:
        before(page)
        page.wait_for_timeout(WAIT)
    page.screenshot(path=OUT / name, full_page=full)
    print("wrote", name)


def open_first_sources(page):
    page.get_by_role("button", name="Sources").first.click()


def run_symptoms(page):
    for s in ["Khát nhiều", "Tiểu nhiều", "Sụt cân", "Nhìn mờ"]:
        page.get_by_role("button", name=s, exact=True).click()
    page.get_by_role("button", name="Check 4 symptoms").click()


def search_typing(page):
    page.get_by_role("combobox").fill("insu")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
        page_shot(page, f"{SITE}/", "web_home.png")
        page_shot(page, f"{SITE}/", "web_search.png", before=search_typing)
        page_shot(page, f"{SITE}/entity/Metformin", "web_entity.png", full=True, before=open_first_sources)
        page_shot(page, f"{SITE}/types/{quote('Đái tháo đường thai kỳ')}", "web_types.png")
        page_shot(page, f"{SITE}/symptoms", "web_symptoms.png", full=True, before=run_symptoms)
        page_shot(page, f"{SITE}/paths?from={quote('Thừa cân, béo phì')}&to={quote('Thận')}", "web_paths.png")
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        page_shot(mobile, f"{SITE}/entity/HbA1c", "web_mobile.png")
        browser.close()


if __name__ == "__main__":
    main()
