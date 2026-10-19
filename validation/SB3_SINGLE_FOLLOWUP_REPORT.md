# SB3 single-controlled-animal follow-up

**Disposition: integration PASS; useful-learning gate FAIL.** This follow-up makes
the existing one-cow screen reproducible and applies a behavior-only gate. It
does not change the simulator, physiology, reward, locked evidence, or training
budget, and it does not promote a checkpoint. No Kaggle or expanded training
was run while producing this follow-up.

## Method

`validation/sb3_single_agent_followup.py` reuses `validation/sb3_pilot.py` with
one PPO-controlled cow and four `NeedController` context cows. The maximum and
default training budget is the existing **7,200 steps**; held-out evaluation
seeds are fixed at **101, 102, 103** (training seed 1). The comparison includes
PPO, the supplied heuristic (`need_controller`), and the existing random
control. The gate uses behavioral resource-success and controlled-cow final
hunger/thirst metrics—not PPO reward—and requires PPO to be no worse than the
heuristic on all three. It is a small screening gate, not a powered claim.

## Existing screen result carried forward

The current pilot report records PPO at 50.7 aggregate resource successes per
episode and controlled-cow final hunger/thirst of 1.00/1.00; the heuristic
records 53.0 successes and 0.094/1.00. Random control records 38.0 successes
and 1.00/1.00. PPO therefore **fails** the resource-success and controlled
hunger checks (and only ties thirst). This remains an exploratory fail; no new
training result is claimed here.

The wrapper's water need saturates at 1.00 even for the heuristic, so these
metrics do not establish that the task is sustainable. Define a feasible
water/forage scenario before treating this as a learning objective.

## Reproduction

Install the optional stack described by `requirements-sb3-pilot.txt`, then run:

```sh
python -m validation.sb3_single_agent_followup --steps 7200
python -m pytest -q tests/test_sb3_single_agent_followup.py
```

The script rejects budgets above 7,200 steps. Its output records the algorithm,
budget, seed, evaluation seeds, three behavior comparisons, and PASS/FAIL gate.
