# MVP 1 browser validation

## Scope and verdict

**Browser acceptance: PASS** for the local desktop UI behaviors exercised below. This is not the overall MVP validation verdict and does not validate policy quality, training, biology, or multi-day behavior.

The app source does not define a virtual-day duration or day-boundary semantics (the clock is a discrete integer tick; physiology and rewards update per tick). Therefore the browser run records **300 logical ticks**, but does not call them three days. No scientifically meaningful day interpretation was invented for this test.

## Reproduction

Tested with Python 3.12.3, Playwright 1.64.0, and headless Chromium 156.0.8078.4. Uses only loopback HTTP; no remote calls.

```sh
# In terminal 1, if the server is not already running:
.venv/bin/python -m farm_sim.server

# In terminal 2:
.venv/bin/python validation/playwright_mvp1.py
```

The test normalizes the service to paused seed 7, exercises UI controls, compares the inspector's selected action and rounded probabilities to the API decision, repeats a one-step seed-19 trajectory for determinism, takes the app through 300 deterministic API step requests, then waits for the browser's polling UI to display tick 300. This captured run connected to an already-running local server (`started_server: false`); the startup instructions above reproduce it from a stopped state. JSON evidence is in `playwright-results.json`; the executable scenario is `validation/playwright_mvp1.py`.

## Evidence

- Page HTTP 200; the rendered 2D canvas measured 860×615 CSS pixels (860×615 backing pixels) and returned a nonempty canvas data URL.
- API and selector each contained exactly cow-1 through cow-5; each selection updated the matching individual inspection panel and `aria-pressed` state.
- UI Step once advanced tick 0→1 and produced 15 event entries (decision/movement/action events); all five cows changed position and emitted movement events. The selected cow showed six action probabilities, one selected-action row, decision tick 0, decision features, and live observation tick 1. API decision/action/probabilities matched the panel.
- Pause/resume toggled state; reset seed 19 returned tick 0. Replaying one step from seed 19 produced an identical serialized animal state.
- 300 steps completed; tick 300, 3,040 retained event entries, five animals with decisions at tick 299, and six probabilities per animal. The browser subsequently showed tick 300 and populated event rows (elapsed time varies by host/runtime).
- No JavaScript console errors, uncaught page errors, or failed browser requests.
- Screenshots (`initial.png`, `after-first-step.png`, `after-300-ticks.png`) were visually inspected. At 1440px width, the two-column pasture/inspector layout and all five markers are visible; no clipped controls, overlapping panels, or missing map content observed. Full-page screenshots are 1440px wide.

## Limitations

This lane does not compare trained/untrained/baseline models, estimate action/reward distributions, run causal interventions, or assess unseen-policy generalization. Those need a controlled simulation evaluation separate from browser acceptance. The page heading says “A day on the pasture,” but no explicit simulated day duration exists in code; this mismatch is reported rather than interpreted as an actual day.
