#!/usr/bin/env python3
"""Local Chromium probe for the MVP 1 synthetic clock and live farm UI.

This probe deliberately writes to a new remediation evidence directory; the
original MVP 1 browser baselines in ``artifacts/validation/browser`` are kept
untouched. Run with ``python3 validation/playwright_mvp1_remediation.py``.
Requires the project Python environment, Playwright Python package, and its
Chromium browser (``python3 -m playwright install chromium`` if needed).
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
OUT = ROOT / "artifacts" / "validation" / "mvp1-remediation" / "browser"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 4107
BOUNDARIES = (0, 1, 23, 24, 47, 48, 72)


def api_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(BASE_URL + path, timeout=3) as response:
        return json.loads(response.read())


def local_server_available() -> bool:
    try:
        api_json("/api/state")
        return True
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def start_server_if_needed() -> tuple[subprocess.Popen[bytes] | None, bool]:
    if local_server_available():
        return None, False
    proc = subprocess.Popen(
        ["python3", "-m", "farm_sim.server"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"local server exited during startup with code {proc.returncode}")
        if local_server_available():
            return proc, True
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("local server did not become ready at 127.0.0.1:8000")


def wait_clock(page: Any, absolute_tick: int, timeout: int = 8000) -> None:
    day, within_day = divmod(absolute_tick, 24)
    expected = f"DAY {day} · TICK {within_day:03d}/024"
    page.wait_for_function(
        "expected => document.querySelector('#clock-time')?.textContent.trim() === expected",
        arg=expected, timeout=timeout,
    )


def clock_snapshot(page: Any, absolute_tick: int) -> dict[str, Any]:
    state = api_json("/api/state")
    clock = state.get("clock", {})
    day, within_day = divmod(absolute_tick, 24)
    expected_ui = f"DAY {day} · TICK {within_day:03d}/024"
    page.wait_for_function(
        "expected => document.querySelector('#clock-time')?.textContent.trim() === expected",
        arg=expected_ui, timeout=8000,
    )
    actual_ui = page.locator("#clock-time").inner_text().strip()
    return {
        "requested_absolute_tick": absolute_tick,
        "api_tick": state.get("tick"),
        "clock": clock,
        "ui_label": actual_ui,
        "expected_ui_label": expected_ui,
        "matches": (
            state.get("tick") == absolute_tick
            and clock.get("tick") == absolute_tick
            and clock.get("absolute_tick") == absolute_tick
            and clock.get("synthetic_day_index") == day
            and clock.get("tick_within_day") == within_day
            and clock.get("ticks_per_virtual_day") == 24
            and actual_ui == expected_ui
        ),
    }


def main() -> int:
    results: dict[str, Any] = {
        "base_url": BASE_URL,
        "local_only": True,
        "started_server": False,
        "server_management": "Use an existing local server if available; otherwise start and stop one for this run.",
        "ticks_per_virtual_day": 24,
        "time_unit": "abstract_simulation_tick (synthetic virtual days; not biological time)",
        "seed": SEED,
        "console_errors": [], "page_errors": [], "failed_requests": [],
        "checks": {}, "screenshots": [],
    }
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc, results["started_server"] = start_server_if_needed()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright Python package missing; install with `python3 -m pip install playwright` then `python3 -m playwright install chromium`.") from exc
        results["playwright_version"] = importlib.metadata.version("playwright")
        results["python_version"] = platform.python_version()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            results["chromium_version"] = browser.version
            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            page.on("console", lambda msg: results["console_errors"].append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda error: results["page_errors"].append(str(error)))
            page.on("requestfailed", lambda req: results["failed_requests"].append({"url": req.url, "error": req.failure}))
            page.goto(BASE_URL + "/", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_selector("#animal-selector [data-animal]", timeout=10000)
            page.wait_for_function("document.querySelector('#pasture-map').toDataURL().length > 1000", timeout=5000)

            # A load initializes the seed control from the API; user edits must
            # remain untouched by periodic state polling, then reset must apply it.
            loaded_state = api_json("/api/state")
            load_input_seed = page.locator("#seed-input").input_value()
            loaded_input_matches_api = load_input_seed == str(loaded_state["seed"])
            edited_seed = int(loaded_state["seed"]) + 1
            page.locator("#seed-input").fill(str(edited_seed))
            time.sleep(1.7)  # Exceed the paused UI poll interval.
            edited_seed_retained = page.locator("#seed-input").input_value() == str(edited_seed)
            api_unchanged_during_edit = api_json("/api/state")["seed"] == loaded_state["seed"]
            page.locator("#reset-world").click()
            page.wait_for_function(f"document.querySelector('#seed-value').textContent.trim() === '{edited_seed}'")
            page.wait_for_function("fetch('/api/state').then(r => r.json()).then(s => s.seed === Number(document.querySelector('#seed-input').value) && s.tick === 0)")
            reset_input_matches_api = page.locator("#seed-input").input_value() == str(edited_seed) == str(api_json("/api/state")["seed"])
            results["checks"]["seed_input"] = {
                "api_seed_on_load": loaded_state["seed"],
                "input_seed_on_load": load_input_seed,
                "load_input_matches_api": loaded_input_matches_api,
                "edited_seed": edited_seed,
                "retained_after_poll": edited_seed_retained,
                "api_unchanged_until_reset": api_unchanged_during_edit,
                "reset_input_matches_api": reset_input_matches_api,
            }

            # Return to a fixed seed for all remaining repeated measurements.
            page.locator("#seed-input").fill(str(SEED))
            page.locator("#reset-world").click()
            page.wait_for_function(f"document.querySelector('#seed-value').textContent.trim() === '{SEED}'")
            page.wait_for_function("document.querySelector('#clock-state').textContent.trim() === 'PAUSED'")
            state0 = api_json("/api/state")
            animal_ids = [cow["id"] for cow in state0["animals"]]
            results["checks"]["initial"] = {
                "http_status": 200,
                "title": page.title(),
                "headline": page.locator("#page-title").inner_text().strip(),
                "headline_identifies_synthetic": "synthetic" in page.locator("#page-title").inner_text().lower(),
                "animal_ids": animal_ids,
                "api_animal_count": len(animal_ids),
                "selector_count": page.locator("#animal-selector [data-animal]").count(),
                "canvas": page.locator("#pasture-map").evaluate("c => ({width:c.width,height:c.height,dataUrlLength:c.toDataURL().length})"),
                "clock": clock_snapshot(page, 0),
            }

            # Check individual selection and key live inspector values against API.
            inspection_checks = []
            for animal_id in animal_ids:
                page.locator(f'#animal-selector [data-animal="{animal_id}"]').click()
                api_animal = page.request.get(BASE_URL + f"/api/animals/{animal_id}").json()
                number = int(animal_id.rsplit("-", 1)[-1])
                shown_name = page.locator("#animal-name").inner_text().strip()
                shown_action = page.locator("#animal-action").inner_text().strip()
                action_labels = {"graze": "grazing", "drink": "drinking", "walk": "walking", "rest": "resting", "ruminate": "ruminating", "herd_interact": "herd interaction"}
                drive_values = {}
                for row in page.locator("#drive-list .drive-row").all():
                    label = row.locator("span").inner_text().strip().lower().replace(" ", "_")
                    shown = float(row.locator("b").inner_text().strip())
                    drive_values[label] = {"shown": shown, "api": round(api_animal["physiology"][label], 2)}
                inspection_checks.append({
                    "id": animal_id,
                    "selected": page.locator(f'#animal-selector [data-animal="{animal_id}"]').get_attribute("aria-pressed") == "true",
                    "name_matches": shown_name == f"Cow {number:02d}",
                    "action_matches": shown_action.lower() == action_labels[api_animal["action"]],
                    "decision_probability_rows": page.locator("#probability-list .prob-row").count(),
                    "drives": drive_values,
                    "drives_match": all(v["shown"] == v["api"] for v in drive_values.values()),
                })
            results["checks"]["inspection_panels"] = inspection_checks
            page.locator('#animal-selector [data-animal="cow-1"]').click()
            page.screenshot(path=str(OUT / "desktop-initial.png"), full_page=True)
            results["screenshots"].append("desktop-initial.png")

            # Exercise controls via their UI, verifying API state reflects them.
            control_checks: dict[str, Any] = {}
            page.locator("#step-once").click()
            page.wait_for_function("document.querySelector('#clock-time').textContent.includes('TICK 001/024')")
            control_checks["step_once"] = clock_snapshot(page, 1)
            page.locator("#toggle-run").click()
            page.wait_for_function("document.querySelector('#clock-state').textContent.trim() === 'RUNNING'")
            run_state = api_json("/api/state")
            control_checks["resume"] = {"ui_state": page.locator("#clock-state").inner_text().strip(), "api_paused": run_state["clock"]["paused"]}
            page.locator("#toggle-run").click()
            page.wait_for_function("document.querySelector('#clock-state').textContent.trim() === 'PAUSED'")
            pause_state = api_json("/api/state")
            control_checks["pause"] = {"ui_state": page.locator("#clock-state").inner_text().strip(), "api_paused": pause_state["clock"]["paused"]}
            page.locator("#speed-select").select_option("2")
            page.wait_for_function("document.querySelector('#speed-select').value === '2'")
            speed_state = api_json("/api/state")
            control_checks["speed"] = {"ui_value": page.locator("#speed-select").input_value(), "api_speed": speed_state["clock"]["speed"]}
            page.locator("#speed-select").select_option("1")
            results["checks"]["controls"] = control_checks

            # Advance to all requested boundaries using deterministic single ticks.
            # At each point verify API metadata and the independently rendered UI label.
            boundary_records = [clock_snapshot(page, 1)]
            for target_tick in BOUNDARIES[2:]:
                current_tick = api_json("/api/state")["tick"]
                for _ in range(target_tick - current_tick):
                    page.request.post(BASE_URL + "/api/step", data={})
                boundary_records.append(clock_snapshot(page, target_tick))
            results["checks"]["clock_boundaries"] = boundary_records
            results["checks"]["three_day_run"] = {
                "final_tick": api_json("/api/state")["tick"],
                "abstract_days_completed": api_json("/api/state")["clock"]["synthetic_day_index"],
                "api_clock": api_json("/api/state")["clock"],
            }
            page.screenshot(path=str(OUT / "desktop-after-72-ticks.png"), full_page=True)
            results["screenshots"].append("desktop-after-72-ticks.png")

            # Determinism check: compare complete state after 72 steps under same seed.
            run_a = api_json("/api/state")
            events_a = page.request.get(BASE_URL + "/api/events?after=-1").json()["events"]
            page.request.post(BASE_URL + "/api/reset", data={"seed": SEED})
            for _ in range(72):
                page.request.post(BASE_URL + "/api/step", data={})
            run_b = api_json("/api/state")
            events_b = page.request.get(BASE_URL + "/api/events?after=-1").json()["events"]
            results["checks"]["same_seed_replay"] = {
                "equal_animals": run_a["animals"] == run_b["animals"],
                "equal_resources": run_a["forage"] == run_b["forage"] and run_a["water"] == run_b["water"],
                "equal_events": events_a == events_b,
                "tick_a": run_a["tick"], "tick_b": run_b["tick"],
            }

            # Narrow viewport visual smoke test; use a fresh reset to capture a clean UI.
            page.request.post(BASE_URL + "/api/reset", data={"seed": SEED})
            page.set_viewport_size({"width": 390, "height": 844})
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("#animal-selector [data-animal]")
            page.wait_for_function("document.querySelector('#pasture-map').toDataURL().length > 1000")
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(150)
            page.screenshot(path=str(OUT / "mobile-initial.png"), full_page=True)
            results["screenshots"].append("mobile-initial.png")
            results["checks"]["mobile"] = {
                "viewport": {"width": 390, "height": 844},
                "selector_count": page.locator("#animal-selector [data-animal]").count(),
                "map_bbox": page.locator("#pasture-map").bounding_box(),
                "inspector_visible": page.locator(".inspector").is_visible(),
                "horizontal_overflow": page.evaluate("document.documentElement.scrollWidth > window.innerWidth"),
                "clock_label": page.locator("#clock-time").inner_text().strip(),
                "layout_audit": page.evaluate("""() => {
                  const selectors = ['.brand', '.top-status', '#page-title', '.intro-copy', '.clock-card', '.toolbar', '#seed-input', '.map-card', '#animal-selector', '.inspector', '.event-section'];
                  const rect = (el) => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height}; };
                  const visible = selectors.map(selector => { const el = document.querySelector(selector); const item={selector, rect: rect(el), clippedX: el.scrollWidth > el.clientWidth + 1, clippedY: el.scrollHeight > el.clientHeight + 1}; if(selector==='#seed-input'){item.value=el.value;item.clientWidth=el.clientWidth;item.scrollWidth=el.scrollWidth;} return item; });
                  const intersections = [];
                  for (let i=0;i<visible.length;i++) for (let j=i+1;j<visible.length;j++) {
                    const a=visible[i], b=visible[j];
                    if(a.selector==='#seed-input'||b.selector==='#seed-input') continue; // input is intentionally nested in toolbar.
                    // Components in separate columns/rows should not geometrically overlap.
                    if (a.rect.x < b.rect.right-1 && b.rect.x < a.rect.right-1 && a.rect.y < b.rect.bottom-1 && b.rect.y < a.rect.bottom-1) intersections.push([a.selector,b.selector]);
                  }
                  const viewport = {width: innerWidth, height: innerHeight, documentWidth: document.documentElement.scrollWidth};
                  const outOfBounds = visible.filter(x => x.rect.x < -1 || x.rect.right > innerWidth + 1).map(x => x.selector);
                  return {viewport, elements: visible, intersections, outOfBounds, noClippedText: visible.every(x => !x.clippedX && !x.clippedY), noComponentOverlap: intersections.length === 0, allWithinViewport: outOfBounds.length === 0};
                }"""),
            }
            browser.close()

        c = results["checks"]
        result_flags = [
            c["initial"]["api_animal_count"] == 5,
            c["initial"]["selector_count"] == 5,
            c["initial"]["canvas"]["dataUrlLength"] > 1000,
            c["initial"]["headline_identifies_synthetic"],
            c["seed_input"]["load_input_matches_api"] and c["seed_input"]["retained_after_poll"] and c["seed_input"]["api_unchanged_until_reset"] and c["seed_input"]["reset_input_matches_api"],
            c["initial"]["clock"]["matches"],
            len(c["inspection_panels"]) == 5 and all(x["selected"] and x["name_matches"] and x["action_matches"] and x["drives_match"] for x in c["inspection_panels"]),
            c["controls"]["step_once"]["matches"],
            c["controls"]["resume"]["ui_state"] == "RUNNING" and c["controls"]["resume"]["api_paused"] is False,
            c["controls"]["pause"]["ui_state"] == "PAUSED" and c["controls"]["pause"]["api_paused"] is True,
            c["controls"]["speed"]["ui_value"] == "2" and c["controls"]["speed"]["api_speed"] == 2,
            all(record["matches"] for record in c["clock_boundaries"]),
            c["three_day_run"]["final_tick"] == 72 and c["three_day_run"]["abstract_days_completed"] == 3,
            all(c["same_seed_replay"].values()),
            c["mobile"]["selector_count"] == 5 and c["mobile"]["inspector_visible"] and not c["mobile"]["horizontal_overflow"] and c["mobile"]["layout_audit"]["noClippedText"] and c["mobile"]["layout_audit"]["noComponentOverlap"] and c["mobile"]["layout_audit"]["allWithinViewport"],
            not results["console_errors"] and not results["page_errors"] and not results["failed_requests"],
        ]
        results["verdict"] = "PASS" if all(result_flags) else "FAIL"
        results["verdict_checks"] = result_flags
    except Exception as exc:
        results["verdict"] = "BLOCKED" if "Playwright Python package missing" in str(exc) or "server" in str(exc).lower() else "FAIL"
        results["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        (OUT / "playwright-remediation-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"verdict": results["verdict"], "started_server": results["started_server"], "checks": results.get("checks", {}), "error": results.get("error"), "console_errors": results["console_errors"], "page_errors": results["page_errors"], "failed_requests": results["failed_requests"], "screenshots": results["screenshots"]}, indent=2))
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
