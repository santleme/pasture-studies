# Pasture Studies: local cattle-farm simulation

Pasture Studies is a small research toy: a 2D grid farm with five individual agents, per-animal event memory, a shared locally trained neural policy, and a discrete simulation clock. It has no hosted model or inference-service dependency.

> **Research boundary:** physiology, rewards, cognition, and behavior in this software are synthetic toy assumptions, not biologically validated cattle models. Do not use this project for husbandry, veterinary, welfare, or farm-operation decisions.

## Run locally

Requires Python 3 and internet access only to install NumPy. From the repository root, create a clean environment and install the sole runtime dependency:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Local execution is for development, lightweight tests, and explicit smoke checks only. Do not run full training or mass trajectory generation on a workstation. Full training, trajectory generation, and extensive benchmarks must use the approved Kaggle execution path described in [Kaggle training](docs/kaggle-training.md); experimental modes remain fail-closed unless separately authorized. To run the explicit one-update diagnostic smoke, install the optional training dependencies in a separate environment and use the command below; this is not research training:

```sh
python3 -m venv .venv-smoke
.venv-smoke/bin/python -m pip install -r requirements-training.txt
.venv-smoke/bin/python -m validation.train_torch_mvp1 --smoke-update \
  --algorithm a2c --device cpu --output /tmp/pasture-smoke
```

For a lightweight simulator development run, use a short smoke scenario rather than the full 10,000-episode policy-training pipeline.

Start the local server:

```sh
.venv/bin/python -m farm_sim.server
```

Open <http://127.0.0.1:8000/>. The server binds to loopback only and requires the validated local `artifacts/policies/default-policy.npz` checkpoint and its JSON sidecar. It fails visibly if either is missing or incompatible; it does not silently fall back to a heuristic.

Run the test suite:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Training and benchmark execution boundary

Do not launch the full local `farm_sim.training` pipeline or Stable-Baselines3 PPO training command from this checkout. Full training (including DAgger/PPO/A2C), mass trajectory generation, and extensive benchmarks are Kaggle-only. Local mode is limited to development, lightweight tests, and explicit smoke diagnostics. The Kaggle release workflow also fails closed for experimental/full-validation execution until independently trusted authorization is available; a Kaggle requirement does not itself grant authorization. See [Kaggle training](docs/kaggle-training.md) for the current approved bounded workflow and restrictions.

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
