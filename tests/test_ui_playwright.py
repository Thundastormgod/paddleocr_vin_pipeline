"""
Live-browser end-to-end tests of the CURRENT Streamlit UI.

The previous version of this file targeted the pre-restructure TAB-based
UI (locator('[role="tab"]') with "Recognition"/"Batch Evaluation" tabs
that no longer exist) and was written as a bare ``async def`` that pytest
cannot execute natively - so it never truly ran against the current app.
The current UI is a sidebar radio with five pages (measured via
streamlit AppTest, 2026-08-19).

These tests are e2e-marked (excluded by default via addopts -m 'not e2e')
and skip with explicit, actionable reasons when a capability is missing:

    pip install playwright && playwright install chromium
    streamlit run src/vin_ocr/web/app.py --server.port 8501
    pytest -m e2e tests/test_ui_playwright.py
"""

import urllib.error
import urllib.request

import pytest

sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright not installed; pip install playwright && playwright install chromium",
)

pytestmark = pytest.mark.e2e

APP_URL = "http://localhost:8501"

#: The sidebar pages of the current UI, verified headlessly via AppTest.
EXPECTED_PAGES = [
    "📁 Data Management",
    "🎯 Training",
    "🔍 Inference",
    "📊 Results Dashboard",
    "🔧 System Health",
]


def _server_is_up() -> bool:
    try:
        with urllib.request.urlopen(APP_URL, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(scope="module")
def page():
    if not _server_is_up():
        pytest.skip(
            f"Streamlit server not running at {APP_URL} - start it with: "
            f"streamlit run src/vin_ocr/web/app.py --server.port 8501"
        )
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except sync_api.Error as exc:
            pytest.skip(
                f"chromium not available ({exc}); run: playwright install chromium"
            )
        page = browser.new_page()
        page.goto(APP_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_timeout(2_000)  # Streamlit hydration
        yield page
        browser.close()


class TestCurrentUILive:
    def test_app_loads(self, page):
        assert "VIN" in page.title() or page.locator("body").inner_text().strip()

    def test_sidebar_has_the_five_current_pages(self, page):
        sidebar_text = page.locator('[data-testid="stSidebar"]').inner_text()
        for label in EXPECTED_PAGES:
            # emoji rendering can vary across fonts; assert on the words
            assert label.split(" ", 1)[1] in sidebar_text, label

    @pytest.mark.parametrize("label", [p.split(" ", 1)[1] for p in EXPECTED_PAGES])
    def test_each_page_renders_without_streamlit_exception(self, page, label):
        page.get_by_text(label, exact=False).first.click()
        page.wait_for_timeout(2_500)
        # Streamlit renders uncaught exceptions in a dedicated element
        assert page.locator('[data-testid="stException"]').count() == 0, (
            f"page {label!r} rendered a Streamlit exception"
        )
        assert page.locator("body").inner_text().strip()
