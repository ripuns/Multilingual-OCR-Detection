"""End-to-end UI test (Playwright driving an installed Edge/Chrome) against the mock API.

Setup:  pip install playwright   (no browser download needed; uses the installed Edge)
Run:    python webapp/ui_tests/mock_server.py &   then   python webapp/ui_tests/ui_test.py
Set SHOTS_DIR to choose where screenshots go. Needs experiments/runs/tamil_pilot/images/sample_0.png
(local-only data) for the Tamil step.
"""
import os
import sys
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHOTS = os.environ.get("SHOTS_DIR", os.path.join(HERE, "shots"))
os.makedirs(SHOTS, exist_ok=True)
URL = os.environ.get("UI_URL", "http://127.0.0.1:8765/")
TAMIL_IMG = os.path.join(REPO, "experiments", "runs", "tamil_pilot", "images", "sample_0.png")  # local-only data
problems = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1320, "height": 900}, color_scheme="light")
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    page.goto(URL)
    page.wait_for_selector("#status-text")
    page.wait_for_function("document.getElementById('status-text').textContent.includes('ready')")
    check(page.inner_text("#status-text") == "Models ready", "health pill shows 'Models ready'")
    check(page.is_disabled("#run-btn"), "Extract button disabled before an image is chosen")
    page.screenshot(path=f"{SHOTS}/01_empty_light.png")

    # sample -> preview
    page.click("#sample-btn")
    page.wait_for_selector("#preview:not([hidden])")
    check(page.is_enabled("#run-btn"), "Extract enabled after sample load")
    check(page.is_hidden("#dropzone"), "dropzone hidden once an image is chosen")
    page.wait_for_function("document.getElementById('preview-meta').textContent.includes('×')")
    print("preview meta:", page.inner_text("#preview-meta"))

    # auto run: capture loading state
    page.click("#run-btn")
    page.wait_for_selector("#state-loading:not([hidden])")
    page.wait_for_timeout(700)
    page.screenshot(path=f"{SHOTS}/02_loading.png")
    check("three language models" in page.inner_text("#loading-title"), "auto loading title mentions three models")
    page.wait_for_selector("#state-result:not([hidden])", timeout=15000)
    page.screenshot(path=f"{SHOTS}/03_result_simple_light.png")
    chip = page.inner_text("#detect-chip")
    check("Detected script: English" in chip, f"detected chip shown ({chip!r})")
    text = page.inner_text("#text-out")
    check("microcontroller:" in text and "Smoke sensor connected" in text, "recognized text rendered")
    check(page.locator("#text-out .ln").count() >= 10, "text rendered as separate lines")
    check(page.locator("#text-out .low").count() >= 1, "low-confidence line is marked")
    check(page.locator("#simple-notes .notice").count() == 1, "low-confidence notice shown")
    check(page.locator("#simple-notes .rerun button").count() == 2, "re-run buttons offered for the other 2 languages")

    # developer view
    page.click("#mode-dev")
    page.wait_for_selector("#view-dev:not([hidden])")
    page.wait_for_function("document.querySelectorAll('#overlay rect').length > 0")
    check(page.locator("#overlay rect").count() == 16, "16 boxes drawn on the image overlay")
    check(page.locator("#regions-body tr").count() == 16, "16 region rows")
    check(page.locator(".bar").count() == 3, "3 evidence bars in auto mode")
    page.hover("#regions-body tr:nth-child(4)")
    check(page.locator("#overlay rect.hl").count() == 1, "hovering a row highlights its box")
    page.screenshot(path=f"{SHOTS}/04_result_dev_light.png", full_page=True)
    page.click("#mode-simple")

    # copy (grant permissions) + download
    ctx.grant_permissions(["clipboard-read", "clipboard-write"])
    page.click("#copy-btn")
    page.wait_for_selector("#toast.show")
    clip = page.evaluate("navigator.clipboard.readText()")
    check("microcontroller:" in clip and clip.count("\n") >= 8, "Copy puts multi-line text on the clipboard")
    with page.expect_download() as dl:
        page.click("#download-btn")
    check(dl.value.suggested_filename == "lekhak-text.txt", "Download .txt works")

    # re-run as another language
    page.click("#simple-notes .rerun button:nth-of-type(1)")
    page.wait_for_selector("#state-loading:not([hidden])")
    check("English" in page.inner_text("#loading-title") or "Tamil" in page.inner_text("#loading-title"), "manual re-run loading title names the language")
    page.wait_for_selector("#state-result:not([hidden])", timeout=15000)
    check(page.inner_text("#detect-chip").startswith("Language:"), "manual run shows 'Language: …' chip")
    check(page.locator("#simple-notes .rerun").count() == 0, "no re-run row in manual mode")

    # dark theme
    page.click("#theme-btn")
    check(page.evaluate("document.documentElement.dataset.theme") == "dark", "theme toggles to dark")
    page.wait_for_timeout(500)
    page.screenshot(path=f"{SHOTS}/05_result_simple_dark.png")
    page.click("#mode-dev")
    page.screenshot(path=f"{SHOTS}/06_result_dev_dark.png", full_page=True)

    # research tab
    page.click("#tab-research")
    page.wait_for_selector("#research-content .claim")
    check(page.locator("#research-content .bar2").count() >= 5, "research bars rendered")
    check(page.locator("#research-content table.data").count() >= 3, "research tables rendered")
    page.wait_for_timeout(600)
    page.screenshot(path=f"{SHOTS}/07_research_dark.png", full_page=True)
    page.click("#theme-btn")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{SHOTS}/08_research_light.png", full_page=True)

    # tamil upload via file chooser, manual Tamil run
    page.click("#tab-recognize")
    page.click("#remove-btn")
    check(page.is_visible("#dropzone"), "Remove returns to the dropzone")
    page.set_input_files("#file", TAMIL_IMG)
    page.wait_for_selector("#preview:not([hidden])")
    page.check("input[name=lang][value=tamil]", force=True)
    check("Tamil model" in page.inner_text("#lang-hint"), "language hint updates")
    page.click("#run-btn")
    page.wait_for_selector("#state-result:not([hidden])", timeout=15000)
    page.click("#mode-simple")
    page.screenshot(path=f"{SHOTS}/09_tamil_simple_light.png")
    check("வணக்கம்" in page.inner_text("#text-out"), "Tamil text rendered")
    check(page.get_attribute("#text-out", "lang") == "ta", "text block lang=ta for correct font shaping")

    # cancel
    page.click("#run-btn")
    page.wait_for_selector("#state-loading:not([hidden])")
    page.click("#cancel-btn")
    page.wait_for_selector("#state-empty:not([hidden])")
    check(True, "Cancel returns to the empty state")

    pre_injection_errors = list(errors)
    # error state (server error)
    page.route("**/api/ocr", lambda r: r.fulfill(status=500, content_type="application/json", body='{"detail":"Pipeline error: boom"}'))
    page.click("#run-btn")
    page.wait_for_selector("#state-error:not([hidden])")
    check("boom" in page.inner_text("#error-msg"), "server error message shown")
    page.screenshot(path=f"{SHOTS}/10_error.png")
    # network failure
    page.unroute("**/api/ocr")
    page.route("**/api/ocr", lambda r: r.abort())
    page.click("#retry-btn")
    page.wait_for_selector("#state-error:not([hidden])")
    check("Can't reach the server" in page.inner_text("#error-title"), "network failure message shown")

    # mobile
    m = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme="light")
    mp = m.new_page()
    mp.goto(URL)
    mp.click("#sample-btn")
    mp.wait_for_selector("#preview:not([hidden])")
    mp.click("#run-btn")
    mp.wait_for_selector("#state-result:not([hidden])", timeout=15000)
    mp.screenshot(path=f"{SHOTS}/11_mobile_result.png", full_page=True)
    overflow = mp.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    check(not overflow, "no horizontal overflow on a 390px-wide screen")

    check(not pre_injection_errors, "no console errors/warnings before the injected failures: " + "; ".join(pre_injection_errors[:5]))
    browser.close()

print("\nPROBLEMS:", problems if problems else "none")
sys.exit(1 if problems else 0)
