"""Real browser smoke and explicitly labeled simulated demonstration video."""

import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "artifacts/demo"
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        browser = playwright.chromium.launch(
            executable_path=str(chrome) if chrome.exists() else None,
            headless=True,
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1080},
            record_video_dir=str(output),
            record_video_size={"width": 1440, "height": 1080},
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        page.get_by_label("Anunciar resultados por voz").uncheck()
        page.get_by_label("Modo de execução").select_option("simulation")
        page.get_by_label("Linha desejada").fill("971D-10")
        page.get_by_role("button", name="Iniciar análise").click()
        expect(page.locator("#result-text")).to_contain_text(
            "Linha solicitada confirmada", timeout=15000
        )
        page.screenshot(path=str(output / "simulation-desktop.png"), full_page=True)
        expect(page.locator("#result-text")).to_contain_text("É outra linha", timeout=15000)
        expect(page.locator("#events")).to_contain_text("#3", timeout=15000)
        expect(page.locator("#result-text")).to_contain_text("Não foi possível", timeout=15000)
        page.get_by_role("button", name="Encerrar análise").click()
        expect(page.locator("#start")).to_be_enabled()
        video = page.video
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "simulation-mobile.png"), full_page=True)
        bounds = page.locator("body").bounding_box()
        overflow = bool(bounds and bounds["width"] > 390)
        context.close()
        if video:
            video.save_as(str(output / "simulation-browser.webm"))
        browser.close()
    result = {
        "scope": "simulation_ui_only",
        "browser": "Chromium",
        "page_errors": errors,
        "mobile_horizontal_overflow": overflow,
        "match_non_match_uncertain": True,
        "audio": "not verified acoustically; disabled in headless recording",
    }
    (ROOT / "reports/browser-smoke.json").write_text(json.dumps(result, indent=2))
    if errors or overflow:
        raise SystemExit("Browser smoke failed; inspect reports/browser-smoke.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
