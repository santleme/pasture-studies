#!/usr/bin/env python3
"""Reproducible, local-only browser acceptance probe for Pasture Studies MVP 1."""
from __future__ import annotations

import json
import importlib.metadata
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000/"
OUT = ROOT / "artifacts" / "validation" / "browser"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    results = {
        "base_url": BASE_URL,
        "started_server": False,
        "ticks_per_virtual_day": None,
        "virtual_day_semantics": "Not defined by MVP source; 300 logical ticks recorded without labeling them as days.",
        "console_errors": [],
        "page_errors": [],
        "failed_requests": [],
        "checks": {},
        "screenshots": [],
    }
    results["playwright_version"] = importlib.metadata.version("playwright")
    results["python_version"] = __import__("platform").python_version()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        results["chromium_version"] = browser.version
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.on("console", lambda msg: results["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda error: results["page_errors"].append(str(error)))
        page.on("requestfailed", lambda req: results["failed_requests"].append({"url": req.url, "error": req.failure}))
        # Normalize any pre-existing local server state to a reproducible paused seed.
        page.request.post(BASE_URL.rstrip("/") + "/api/control", data={"action": "pause"})
        page.request.post(BASE_URL.rstrip("/") + "/api/reset", data={"seed": 7})
        # The page intentionally polls while open, so networkidle is not a stable
        # lifecycle state; wait for document parsing and the app's initial paint.
        response = page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_selector("#animal-selector [data-animal]", timeout=10000)
        page.wait_for_function("document.querySelector('#pasture-map').width > 0", timeout=5000)
        results["checks"]["http_status"] = response.status if response else None
        results["checks"]["page_title"] = page.title()
        results["checks"]["canvas"] = page.locator("#pasture-map").evaluate("c => ({width:c.width,height:c.height,rect:(()=>{const r=c.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})(),dataUrlLength:c.toDataURL().length})")
        state = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()
        animal_ids = [a["id"] for a in state["animals"]]
        results["checks"]["animal_ids"] = animal_ids
        results["checks"]["animal_count_api"] = len(animal_ids)
        results["checks"]["selector_count"] = page.locator("#animal-selector [data-animal]").count()
        results["checks"]["map_bbox"] = page.locator("#pasture-map").bounding_box()
        results["checks"]["inspector_bbox"] = page.locator(".inspector").bounding_box()
        before_step_state = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()
        page.screenshot(path=str(OUT / "initial.png"), full_page=True)
        results["screenshots"].append("initial.png")

        # Selecting each individual should synchronize tab selection and inspector identity.
        selections = []
        for i, animal_id in enumerate(animal_ids, 1):
            page.locator(f'#animal-selector [data-animal="{animal_id}"]').click()
            actual = page.locator("#animal-name").inner_text()
            pressed = page.locator(f'#animal-selector [data-animal="{animal_id}"]').get_attribute("aria-pressed")
            selections.append({"animal_id": animal_id, "panel_name": actual, "aria_pressed": pressed})
        results["checks"]["selection_checks"] = selections

        # One UI step validates clock, decision log, and live decision/feature provenance.
        page.locator("#step-once").click()
        page.wait_for_function("document.querySelector('#clock-time').textContent.trim() === 'TICK 001'")
        page.wait_for_function("document.querySelectorAll('#event-list .event-row').length > 0")
        page.locator('#animal-selector [data-animal="cow-1"]').click()
        api_animal = page.request.get(BASE_URL.rstrip("/") + "/api/animals/cow-1").json()
        after_step_state = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()
        events_after_step = page.request.get(BASE_URL.rstrip("/") + "/api/events?after=-1").json()["events"]
        panel = {
            "name": page.locator("#animal-name").inner_text(),
            "action": page.locator("#animal-action").inner_text(),
            "decision_context": page.locator("#decision-context").inner_text(),
            "feature_label": page.locator("#feature-vector-label").inner_text(),
            "probability_rows": page.locator("#probability-list .prob-row").count(),
            "chosen_rows": page.locator("#probability-list .prob-row.chosen").count(),
            "displayed_probabilities": {
                row.locator("span").inner_text().strip().lower().replace(" ", "_"): float(row.locator("b").inner_text().strip().rstrip("%"))
                for row in page.locator("#probability-list .prob-row").all()
            },
            "chosen_action_label": page.locator("#probability-list .prob-row.chosen span").inner_text().strip().lower(),
            "state": {"tick": api_animal.get("decision_tick"), "action": api_animal.get("action"), "decision_probabilities": api_animal.get("decision_probabilities"), "decision_features": api_animal.get("decision_features"), "observation_tick": api_animal.get("observation", {}).get("tick")},
            "movement_events": [e for e in events_after_step if e.get("kind") == "movement"],
            "moved_animals": sum((old["x"], old["y"]) != (new["x"], new["y"]) for old, new in zip(before_step_state["animals"], after_step_state["animals"])),
        }
        results["checks"]["post_step_inspector"] = panel
        results["checks"]["event_rows_after_step"] = page.locator("#event-list .event-row").count()
        display_probabilities_match_api = all(
            panel["displayed_probabilities"][action] == round(probability * 100, 1)
            for action, probability in zip(("graze", "drink", "walk", "rest", "ruminate", "herd_interact"), panel["state"]["decision_probabilities"])
        )
        chosen_action_matches_api = panel["chosen_action_label"].replace(" ", "_") == panel["state"]["action"]
        results["checks"]["inspector_matches_api"] = {
            "action_matches": panel["action"].strip().lower() == {"graze": "grazing", "drink": "drinking", "walk": "walking", "rest": "resting", "ruminate": "ruminating", "herd_interact": "herd interaction"}[panel["state"]["action"]],
            "displayed_probabilities_match_api_rounded_0_1_percent": display_probabilities_match_api,
            "chosen_probability_row_matches_api_action": chosen_action_matches_api,
            "chosen_action_is_probability_argmax": panel["state"]["action"] == ("graze", "drink", "walk", "rest", "ruminate", "herd_interact")[max(range(6), key=lambda i: panel["state"]["decision_probabilities"][i])],
        }
        page.screenshot(path=str(OUT / "after-first-step.png"), full_page=True)
        results["screenshots"].append("after-first-step.png")

        # Pause/resume through the UI and deterministic reset behavior.
        page.locator("#toggle-run").click()
        page.wait_for_function("document.querySelector('#clock-state').textContent.includes('RUNNING')")
        page.locator("#toggle-run").click()
        page.wait_for_function("document.querySelector('#clock-state').textContent.includes('PAUSED')")
        page.locator("#seed-input").fill("19")
        page.locator("#reset-world").click()
        page.wait_for_function("document.querySelector('#seed-value').textContent.trim() === '19' && document.querySelector('#clock-time').textContent.trim() === 'TICK 000'")
        state_after_reset = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()
        first_replay_step = page.request.post(BASE_URL.rstrip("/") + "/api/step", data={}).json()
        first_replay_animals = first_replay_step["state"]["animals"]
        page.request.post(BASE_URL.rstrip("/") + "/api/reset", data={"seed": 19})
        second_replay_step = page.request.post(BASE_URL.rstrip("/") + "/api/step", data={}).json()
        results["checks"]["pause_resume"] = page.locator("#clock-state").inner_text()
        results["checks"]["reset_seed"] = state_after_reset["seed"]
        results["checks"]["same_seed_trajectory_replay_equal"] = first_replay_animals == second_replay_step["state"]["animals"]
        page.request.post(BASE_URL.rstrip("/") + "/api/reset", data={"seed": 19})
        # Controlled 300-tick sustained run; app defines no virtual-day length.
        start_tick = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()["tick"]
        started = time.monotonic()
        for _ in range(300):
            reply = page.request.post(BASE_URL.rstrip("/") + "/api/step", data={}).json()
            if not reply.get("ok"):
                raise RuntimeError(f"step failed: {reply}")
        elapsed = time.monotonic() - started
        final_state = page.request.get(BASE_URL.rstrip("/") + "/api/state").json()
        results["checks"]["sustained_run"] = {"start_tick": start_tick, "steps": 300, "final_tick": final_state["tick"], "elapsed_seconds": round(elapsed, 3), "events_retained": final_state["event_retention"]["retained_count"], "all_animals_have_decisions": all(a["decision_tick"] == final_state["tick"] - 1 for a in final_state["animals"]), "all_probabilities_length_6": all(len(a["decision_probabilities"] or []) == 6 for a in final_state["animals"])}
        page.wait_for_function("document.querySelector('#clock-time').textContent.trim() === 'TICK 300'", timeout=5000)
        page.screenshot(path=str(OUT / "after-300-ticks.png"), full_page=True)
        results["screenshots"].append("after-300-ticks.png")
        results["checks"]["final_panel_tick"] = page.locator("#clock-time").inner_text()
        results["checks"]["final_event_rows"] = page.locator("#event-list .event-row").count()
        browser.close()
    results["verdict"] = "PASS" if (
        results["checks"]["http_status"] == 200
        and results["checks"]["animal_count_api"] == 5
        and results["checks"]["selector_count"] == 5
        and all(s["aria_pressed"] == "true" and s["panel_name"] == f"Cow {i:02d}" for i, s in enumerate(results["checks"]["selection_checks"], 1))
        and results["checks"]["post_step_inspector"]["state"]["tick"] == 0
        and results["checks"]["post_step_inspector"]["state"]["observation_tick"] == 1
        and results["checks"]["post_step_inspector"]["probability_rows"] == 6
        and all(results["checks"]["inspector_matches_api"].values())
        and results["checks"]["same_seed_trajectory_replay_equal"]
        and results["checks"]["post_step_inspector"]["moved_animals"] > 0
        and results["checks"]["sustained_run"]["final_tick"] == 300
        and results["checks"]["sustained_run"]["all_animals_have_decisions"]
        and not results["console_errors"] and not results["page_errors"] and not results["failed_requests"]
    ) else "FAIL"
    (OUT / "playwright-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"verdict": results["verdict"], "checks": results["checks"], "console_errors": results["console_errors"], "page_errors": results["page_errors"], "failed_requests": results["failed_requests"], "screenshots": results["screenshots"]}, indent=2))
    if results["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
