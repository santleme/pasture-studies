# Pasture Studies: local cattle-farm simulation

Pasture Studies is a small research toy: a 2D grid farm with five individual agents, per-animal event memory, a shared locally trained neural policy, and a discrete simulation clock. It has no hosted model or inference-service dependency.

> **Research boundary:** physiology, rewards, cognition, and behavior in this software are synthetic toy assumptions, not biologically validated cattle models. Do not use this project for husbandry, veterinary, welfare, or farm-operation decisions.

## Run locally

Requires Python 3 and internet access only to install NumPy. From the repository root, create a clean environment and install the sole runtime dependency:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Train five seeded policy checkpoints locally (10,000 episodes total), save them under `artifacts/policies/`, and write the evaluation report:

```sh
.venv/bin/python -m farm_sim.training --output artifacts/policies
```

The training command exits unsuccessfully if its predeclared learning or paired-memory counterfactual checks fail. `default-policy` is the last-known-good promoted checkpoint: the seed-11 policy replaces it only after the full suite passes. If a later run completes but fails those checks, its report records the failed criteria and the promoted default checkpoint and sidecar remain unchanged. Start the local server:

```sh
.venv/bin/python -m farm_sim.server
```

Open <http://127.0.0.1:8000/>. The server binds to loopback only and requires the validated local `artifacts/policies/default-policy.npz` checkpoint and its JSON sidecar. It fails visibly if either is missing or incompatible; it does not silently fall back to a heuristic.

Run the test suite:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Use the simulation

- **Resume / Pause** starts and stops the clock; **Step once** advances exactly one logical tick. Time is synthetic: 24 abstract ticks make one virtual day; no real-hours or biological-time mapping is intended.
- Choose `½×`, `1×`, `2×`, or `4×` to change wall-clock pacing; this does not change the logical transition step or seeded logical trajectory. Each action occupies one abstract tick; each tick uses one pre-transition perception snapshot per animal, resolves actions in stable ID order, applies one physiology/reward update per animal, then commits memory/events and the clock.
- Set a scenario seed and choose **Reset scenario** to replay a seeded initial farm.
- Select one of the five cattle on the map or in the animal strip to inspect its toy state, observation, retrieved events, encoded feature vector, action probabilities, reward, and recent events.
- The decision log lists tick-stamped decisions and transition events.

Actions are grazing, drinking, walking, resting, rumination, and herd interaction. The learned policy chooses an action; separate code updates toy physiology and reward, resolves movement, advances time, and renders the browser view. See [Architecture and experiment design](docs/architecture.md), [Reference analysis](docs/reference-analysis.md), and the [training report](artifacts/policies/training-report.json).

## Results and limitations

The 97% versus 14% accuracy result and five passing paired memory checks above describe the **original narrow fixed-task baseline only**; they do not establish rollout utility or causal cognition. The current **59-test** full suite passes, including the direct clock-speed invariance check and paired three-day replays across 100 seeds using the shipped default policy. Playwright/Chromium browser validation is available and passes. Current locked evaluation found the encoder-generated policy selected `HERD_INTERACT` in all 216,000 recorded decisions per family (648,000 total across three families), while the four-case policy remained almost entirely `WALK`; memory/perception counterfactual influence failed, and learned utility remains inconclusive because its frozen precision target was not met. A separate provenance audit found the pre-lock attestation did not bind runtime simulation/policy modules; see the [MVP 1 remediation report](validation/MVP1_REMEDIATION_REPORT.md) for current evidence and gate-by-gate results. None of these synthetic outcomes validate cattle behavior or biology.

The implementation is an independent project inspired by Stanford's Generative Agents source. It does not extend that codebase. See the reference analysis for the pinned source commit, source-based findings, and adaptation decisions.
