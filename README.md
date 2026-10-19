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
.venv/bin/python -m scripts.run_standard_tests unittest
.venv/bin/python -m scripts.run_standard_tests pytest
```

Both default runners execute the static test-safety gate before collection or
execution. Default tests use frozen fixtures/mocks for training, serialization,
and dataset contracts; they do not update policy weights or generate DAgger
trajectories. Tests that deliberately perform policy or optimizer training are
isolated under `tests/opt_in_training/` and are excluded from default discovery.
They can only be requested through the dedicated runner, which requires both
`PASTURE_RUN_OPT_IN_TRAINING_TESTS=1` and a verified Kaggle runtime:

```sh
PASTURE_RUN_OPT_IN_TRAINING_TESTS=1 .venv/bin/python -m scripts.run_opt_in_training_tests unittest
PASTURE_RUN_OPT_IN_TRAINING_TESTS=1 .venv/bin/python -m scripts.run_opt_in_training_tests pytest
```

## Training and benchmark execution boundary

The simulator can also be exposed as a Gymnasium task without changing its runtime dependencies or transition logic. Install the separate optional stack and run a reproducible one-cow pilot followed by a joint five-cow PPO run; the command evaluates both against seeded random actions and writes models plus a comparison report under `artifacts/sb3-ppo/`:

```sh
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r requirements-rl.txt
.venv-rl/bin/python -m scripts.train_sb3_ppo --seed 11 --timesteps 50000 --evaluation-episodes 20
```

The one-animal mode controls cow-1 and holds the other four at `REST`; joint mode controls all five with one shared PPO policy and mean per-cow simulator reward. The 72-tick boundary is an artificial Gymnasium truncation, not a biological terminal state. This is an optional research comparison, does not replace the shipped NumPy checkpoint, and must not be interpreted as cattle validation. PettingZoo/TorchRL are not needed for this single-agent/joint-action interface. See the [consolidated RL pilot report](validation/SB3_CONSOLIDATED_PILOT_REPORT.md) for results, behavioral limitations, and a gated expansion proposal.

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
