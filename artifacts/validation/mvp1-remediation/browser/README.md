# MVP 1 remediation browser evidence

Local-only Chromium acceptance probe for the synthetic clock, five-cow rendering, live inspection panels, controls, deterministic replay, and narrow viewport. It does not modify application source or overwrite `artifacts/validation/browser/` baselines.

## Run

```sh
# If needed, create an isolated test environment (outside the repository):
python3 -m venv /tmp/mvp1-browser-venv
/tmp/mvp1-browser-venv/bin/pip install playwright
/tmp/mvp1-browser-venv/bin/python -m playwright install chromium

# Start the app in another terminal if it is not already running:
.venv/bin/python -m farm_sim.server

# Run the probe:
/tmp/mvp1-browser-venv/bin/python validation/playwright_mvp1_remediation.py
```

The probe uses only `127.0.0.1:8000`, uses seed `4107`, and writes its JSON/screenshots into this directory. It reuses an existing server rather than terminating an externally owned process; when no server is available it attempts to start and stop one itself. In the recorded run `started_server` is `false`: the parent agent restarted the app to load the updated server implementation before this probe ran.

## Recorded run

- Verdict: **PASS**
- At browser load the seed input matched API seed `4107`; an edit to `4108` survived a polling cycle without changing the world, then reset applied the edited value and synchronized input/API. Probe restored seed `4107` before remaining checks.
- Headline explicitly identifies the pasture day as synthetic.
- Synthetic clock/API/UI agree at absolute ticks 0, 1, 23, 24, 47, 48, and 72 (3 abstract 24-tick days).
- Five cows rendered on the canvas and selectable; all five inspector names/actions/rounded drive values matched API values.
- UI pause/resume, single-step, and speed selection matched API state.
- Same-seed 72-tick replay produced identical animal/resource/event state.
- At 390px, browser layout audit found no horizontal overflow, clipped text/input, component overlap, or components out of horizontal bounds; seed `4107` remains fully visible.
- No console errors, page errors, or failed requests.
- Screenshots: `desktop-initial.png`, `desktop-after-72-ticks.png`, `mobile-initial.png`.

The 390px screenshot was inspected; the map, five selectors, synthetic clock, inspection card, and event log remain laid out without clipping or overlap. These are software/UI checks, not biological claims.
