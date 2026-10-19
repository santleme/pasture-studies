"""Reproduce the existing one-controlled-animal SB3 screening gate.

This deliberately reuses the original pilot and its 7,200-step training cap;
it adds no simulator or reward changes and is not a higher-budget experiment.
"""
from __future__ import annotations

import argparse
import json
from farm_sim.execution_policy import require_kaggle_for_full_workload

DEFAULT_SEED = 1
DEFAULT_TRAINING_STEPS = 7_200
MAX_TRAINING_STEPS = 7_200
EVALUATION_SEEDS = (101, 102, 103)


def learning_gate(result: dict) -> dict:
    """Compare PPO to the supplied heuristic using behavior, not reward."""
    trained = result["trained"]
    heuristic = result["baselines"]["need_controller"]
    checks = {
        "resource_successes_at_least_heuristic": (
            trained["mean_resource_successes"] >= heuristic["mean_resource_successes"]
        ),
        "controlled_hunger_no_worse_than_heuristic": (
            trained["mean_controlled_hunger"] <= heuristic["mean_controlled_hunger"]
        ),
        "controlled_thirst_no_worse_than_heuristic": (
            trained["mean_controlled_thirst"] <= heuristic["mean_controlled_thirst"]
        ),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def run_followup(*, seed: int = DEFAULT_SEED, training_steps: int = DEFAULT_TRAINING_STEPS) -> dict:
    if seed != DEFAULT_SEED:
        raise ValueError(f"seed must be the fixed follow-up seed {DEFAULT_SEED}")
    if training_steps < 1 or training_steps > MAX_TRAINING_STEPS:
        raise ValueError(f"training_steps must be between 1 and {MAX_TRAINING_STEPS}")

    # Keep the optional RL stack lazy so the gate logic/tests do not require
    # Gymnasium, SB3, or Torch just to import this module.
    from validation.sb3_pilot import train_and_evaluate

    result = train_and_evaluate(agents=1, total_timesteps=training_steps, seed=seed)
    if result["evaluation_seeds"] != list(EVALUATION_SEEDS):
        raise RuntimeError("pilot evaluation seeds do not match the fixed follow-up seeds")
    return {
        **result,
        "phase": "single-controlled-animal",
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "learning_gate": learning_gate(result),
        "training_budget_cap": MAX_TRAINING_STEPS,
        "metrics_basis": "behavior metrics independent of training reward",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--smoke", action="store_true", help="fixed 128-step local development smoke")
    args = parser.parse_args()
    try:
        require_kaggle_for_full_workload("SB3 single-agent follow-up", smoke=args.smoke)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.smoke:
        if args.steps != DEFAULT_TRAINING_STEPS:
            parser.error("--smoke uses fixed limits; omit --steps")
        args.steps = 128
    print(json.dumps(run_followup(training_steps=args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
