# Gymnasium / Stable-Baselines3 local pilot

**Disposition: integration PASS; useful-learning gate FAIL.** This is an exploratory local pilot, not a change to the locked MVP1 evidence or a promotion of a checkpoint. No Kaggle training was run.

## Setup

- Existing simulator retained; no production simulation/physiology code changed.
- Optional package environment: `.venv/bin/python -m pip install -r requirements-sb3-pilot.txt` (Gymnasium 1.2.3, Stable-Baselines3 2.7.1; PyTorch 2.14.0 CPU).
- New `validation/sb3_pilot.py` wraps the existing `Simulation`. Episode horizon is 72 ticks and the external validation wrapper regrows consumed resource patches after three ticks; existing world mechanics remain unchanged in production.
- Single-agent phase trains cow-1 while four need-controller animals provide scripted context. Five-agent phase uses a centralized PPO MLP with 5-component `MultiDiscrete(6)` actions; all observations are collected before the existing sequential stable-ID commit. This is not PettingZoo and does not assert simultaneous commit semantics.
- In the wrapper only, the already-prototyped marginal-contact adjustment removes the +0.1 HERD_INTERACT bonus when social need does not improve. No locked reward source was edited.
- Each trial uses a single training seed, a small timestep cap, and held-out evaluation seeds 101–103. These are screening results, not powered statistical claims.

## Results (behavior metrics independent of reward)

| Pilot | Trained PPO | Need-controller baseline | Random baseline | Outcome |
|---|---|---|---|---|
| Cow-1, 7,200 steps | 50.7 resource successes/episode (all five cows combined); controlled cow final hunger 1.00, thirst 1.00 | 53.0 successes; controlled hunger 0.094, thirst 1.00 | 38.0 successes; controlled hunger/thirst 1.00 | PPO fails to match heuristic maintenance. It eliminated HERD_INTERACT after marginal correction, but still did not maintain the controlled cow's hunger. |
| Five-agent joint, 3,600 steps | 12.0 successes; mean world hunger 0.807, thirst 1.00 | 41.7 successes; mean world hunger 0.490, thirst 0.897 | 16.3 successes; mean world hunger 0.826, thirst 1.00 | Joint PPO collapses to REST/GRAZE and is worse than the heuristic. |

Action counts over the 3 evaluation episodes:
- One-cow PPO: GRAZE 205, DRINK 61, WALK 458, REST 356, RUMINATE 0, HERD_INTERACT 0.
- Five-agent PPO: GRAZE 216, REST 864, all other actions 0.

The water need saturating at 1.00 even for the scripted comparator reveals a remaining task/supply issue: this wrapper is useful for adapter integration but is not a validated sustainable objective. Aggregate event counts do not by themselves establish useful individual behavior.

## Reproduce

```sh
.venv/bin/python -m pytest -q tests/test_sb3_pilot.py
.venv/bin/python -m validation.sb3_pilot --agents 1 --steps 7200 --seed 1
.venv/bin/python -m validation.sb3_pilot --agents 5 --steps 3600 --seed 1
```

## Decision

The stack adapter and Gymnasium checker work, but the learning criterion failed at both scales; therefore do **not** promote PPO or start an expensive cohort/Kaggle run. Keep this isolated prototype as an executable benchmark. Next work should first define a viable task with adequate forage **and water** supply and an externally measurable per-agent success criterion, then repeat with multiple training seeds and held-out worlds. If the feasible single-agent PPO baseline passes that gate, test shared-policy IPPO/MAPPO in TorchRL or a correctly specified PettingZoo Parallel environment. The existing five-agent wrapper is only a centralized PPO bridge, not evidence that the project's multi-agent training is solved.
