"""Bounded, validator-gated multi-seed RL/BC research runner.

This runner changes only validation wrapper configuration. It writes explicit
per-seed behavioral outcomes; reward is retained as a diagnostic but is never
used as the success gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.renewable_rl import (
    BehaviorCloner,
    collect_teacher_data,
    evaluate_policy,
    train_dagger,
    train_ppo,
)
from farm_sim.execution_policy import require_kaggle_for_full_workload


FORAGE_SITES = 8
WATER_SITES = 6
REGROWTH_TICKS = 1
EVALUATION_SEEDS = (111, 123, 145, 167, 189)


def run_phase(*, agents: int, train_seeds: tuple[int, ...], steps: int,
              bc_episodes: int, output: Path, smoke: bool = False) -> dict:
    require_kaggle_for_full_workload("autoresearch RL training and evaluation", smoke=smoke)
    if smoke and (len(train_seeds) != 1 or steps != 128 or bc_episodes != 1):
        raise ValueError("local autoresearch smoke is fixed to one seed, 128 PPO steps, and one BC episode")
    if agents not in (1, 5):
        raise ValueError("agents must be 1 or 5")
    if not train_seeds or not (0 < steps <= 15_000) or bc_episodes <= 0:
        raise ValueError("provide training seeds, steps in [1, 15000], and positive BC episodes")
    modes = {"teacher": evaluate_policy(
        None, agents=agents, seeds=EVALUATION_SEEDS,
        forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
        regrowth_ticks=REGROWTH_TICKS, mode="teacher", context="rest",
    ), "random": evaluate_policy(
        None, agents=agents, seeds=EVALUATION_SEEDS,
        forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
        regrowth_ticks=REGROWTH_TICKS, mode="random", context="rest",
    )}
    observations_per_animal = 30
    bc_rows = []
    for seed in train_seeds:
        x, y = collect_teacher_data(
            agents=agents, episodes=bc_episodes, seed=10_000 + seed,
            forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
            regrowth_ticks=REGROWTH_TICKS, context="rest",
        )
        cloner = BehaviorCloner(observations_per_animal, 6, seed, agents=agents).fit(x, y)
        bc_rows.append({
            "training_seed": seed,
            "training_samples": int(len(y)),
            "evaluation": evaluate_policy(
                cloner, agents=agents, seeds=EVALUATION_SEEDS,
                forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
                regrowth_ticks=REGROWTH_TICKS, mode="behavior_cloning", context="rest",
            ),
        })
    modes["behavior_cloning"] = {"training_runs": bc_rows}

    dagger_runs = []
    for seed in train_seeds:
        cloner, collection = train_dagger(
            agents=agents, seed=seed, forage_sites=FORAGE_SITES,
            water_sites=WATER_SITES, initial_episodes=bc_episodes,
            rounds=1 if smoke else 6,
            episodes_per_round=1 if smoke else max(10, bc_episodes // 2),
            regrowth_ticks=REGROWTH_TICKS,
        )
        dagger_runs.append({
            "training_seed": seed,
            **collection,
            "evaluation": evaluate_policy(
                cloner, agents=agents, seeds=EVALUATION_SEEDS,
                forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
                regrowth_ticks=REGROWTH_TICKS, mode="dagger", context="rest",
            ),
        })
    modes["dagger"] = {"training_runs": dagger_runs}

    ppo_variants = (
        ("ppo_legacy_observation_reward", False, "legacy"),
        ("ppo_global_observation_balanced_events", True, "balanced_events"),
        ("ppo_global_observation_state_cost", True, "state_cost"),
    )
    for name, global_observations, reward_mode in ppo_variants:
        training_runs = []
        for seed in train_seeds:
            model = train_ppo(
                agents=agents, seed=seed, total_timesteps=steps,
                forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
                regrowth_ticks=REGROWTH_TICKS,
                global_observations=global_observations,
                reward_mode=reward_mode, context="rest",
            )
            evaluation = evaluate_policy(
                model, agents=agents, seeds=EVALUATION_SEEDS,
                forage_sites=FORAGE_SITES, water_sites=WATER_SITES,
                regrowth_ticks=REGROWTH_TICKS, mode=name, context="rest",
                global_observations=global_observations,
                reward_mode=reward_mode,
            )
            training_runs.append({"training_seed": seed, "evaluation": evaluation})
        modes[name] = {"training_runs": training_runs}

    result = {
        "schema_version": 1,
        "smoke_mode": bool(smoke),
        "interpretation": "local smoke is plumbing evidence only; do not use as training, learning, or benchmark evidence" if smoke else "bounded validator-gated research output",
        "agents": agents,
        "scenario": {
            "forage_sites": FORAGE_SITES,
            "water_sites": WATER_SITES,
            "regrowth_ticks": REGROWTH_TICKS,
            "context": "rest for uncontrolled animals",
            "historic_scenario_modified": False,
        },
        "training_seeds": list(train_seeds),
        "training_steps_per_ppo_seed": steps,
        "bc_teacher_episodes_per_seed": bc_episodes,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "metrics": ["per-animal forage successes", "per-animal water successes",
                    "per-animal final hunger", "per-animal final thirst",
                    "per-animal action frequencies", "reward diagnostic only"],
        "modes": modes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, choices=(1, 5), default=1)
    parser.add_argument("--train-seeds", default="1")
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--bc-episodes", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="fixed 128-step/one-episode local engineering smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_kaggle_for_full_workload("autoresearch RL training and evaluation", smoke=args.smoke)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.smoke:
        if (args.train_seeds, args.steps, args.bc_episodes) != ("1", 7200, 30):
            parser.error("--smoke uses fixed reduced limits; omit seed and workload overrides")
        args.steps, args.bc_episodes = 128, 1
    seeds = tuple(int(value) for value in args.train_seeds.split(",") if value)
    result = run_phase(
        agents=args.agents, train_seeds=seeds, steps=args.steps,
        bc_episodes=args.bc_episodes, output=args.output, smoke=args.smoke,
    )
    print(json.dumps({"agents": result["agents"], "training_seeds": result["training_seeds"],
                      "output": str(args.output),
                      "controlled_metric_means": {
                          name: mode.get("controlled_mean")
                          for name, mode in result["modes"].items()
                          if "controlled_mean" in mode
                      }}, indent=2))


if __name__ == "__main__":
    main()
