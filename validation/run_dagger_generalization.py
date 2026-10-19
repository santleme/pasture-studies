"""Reproduce paired fixed-vs-domain-randomized DAgger generalization."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from farm_sim.execution_policy import require_kaggle_for_full_workload

from validation.dagger_generalization import (
    TEST_DOMAINS, TRAIN_PROFILE, evaluate_domain, save_json,
    train_generalized_dagger,
)


class SeededRandom:
    def __init__(self, agents, seed):
        self.agents = agents
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):
        del observation, deterministic
        actions = self.rng.integers(0, 6, size=self.agents)
        return (int(actions[0]) if self.agents == 1 else actions), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, choices=(1, 5), default=5)
    parser.add_argument("--train-seeds", default="1,2,3,4,5")
    parser.add_argument("--eval-seeds", default="301,307,313,317,331")
    parser.add_argument("--initial-episodes", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--episodes-per-round", type=int, default=15)
    parser.add_argument("--observation", choices=("base", "domain", "history_domain"), default="base")
    parser.add_argument("--architecture", choices=("mlp", "wide", "history", "joint"), default="mlp")
    parser.add_argument("--smoke", action="store_true", help="fixed one-seed/one-round local engineering smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_kaggle_for_full_workload("DAgger generalization study", smoke=args.smoke)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.smoke:
        if (args.train_seeds, args.eval_seeds, args.initial_episodes, args.rounds,
                args.episodes_per_round) != ("1,2,3,4,5", "301,307,313,317,331", 30, 6, 15):
            parser.error("--smoke uses fixed reduced limits; omit seed and workload overrides")
        args.train_seeds, args.eval_seeds = "1", "301"
        args.initial_episodes, args.rounds, args.episodes_per_round = 1, 1, 1
    train_seeds = tuple(int(x) for x in args.train_seeds.split(","))
    eval_seeds = tuple(int(x) for x in args.eval_seeds.split(","))
    if not args.smoke and (len(train_seeds) != 5 or len(eval_seeds) != 5):
        parser.error("the declared gate requires five train and five held-out seeds")

    paired = []
    for seed in train_seeds:
        one = {"training_seed": seed, "candidates": {}}
        for name, profile, architecture in (
            ("fixed_reference", "nominal", args.architecture),
            ("domain_randomized", "randomized_train", args.architecture),
        ):
            observation = args.observation
            model, config = train_generalized_dagger(
                agents=args.agents, seed=seed, observation=observation,
                initial_episodes=args.initial_episodes, rounds=args.rounds,
                episodes_per_round=args.episodes_per_round,
                training_profile=profile, architecture=architecture,
            )
            domains = {}
            for domain in TEST_DOMAINS:
                domains[domain] = evaluate_domain(
                    model, agents=args.agents, seeds=eval_seeds,
                    domain=domain, observation=observation,
                )
            timing_intervention = {}
            if name == "domain_randomized":
                for domain in ("regrowth_shift", "combined_unseen"):
                    timing_intervention[domain] = evaluate_domain(
                        model, agents=args.agents, seeds=eval_seeds,
                        domain=domain, observation=observation,
                        refresh_observation=False,
                    )
                if observation == "history_domain":
                    for domain in ("regrowth_shift", "combined_unseen"):
                        for intervention in ("mask_global_supply",
                                             "mask_previous_observation",
                                             "mask_domain_context"):
                            timing_intervention[f"{domain}:{intervention}"] = evaluate_domain(
                                model, agents=args.agents, seeds=eval_seeds,
                                domain=domain, observation=observation,
                                ablation=intervention,
                            )
            one["candidates"][name] = {
                "training": config, "domains": domains,
                "causal_interventions": timing_intervention,
            }
            print(f"finished train_seed={seed} method={name}", flush=True)
        one["baselines"] = {}
        for label, policy in (("teacher", None), ("random", SeededRandom(args.agents, seed + 99_000))):
            one["baselines"][label] = {
                domain: evaluate_domain(policy, agents=args.agents, seeds=eval_seeds,
                                        domain=domain, observation="base")
                for domain in TEST_DOMAINS
            }
        paired.append(one)
        result = {
            "schema_version": 1,
            "smoke_mode": bool(args.smoke),
            "interpretation": "local smoke is plumbing evidence only; not generalization or benchmark evidence" if args.smoke else "paired DAgger generalization study",
            "agents": args.agents,
            "training_seeds": list(train_seeds),
            "evaluation_seeds": list(eval_seeds),
            "training_config": {"initial_episodes": args.initial_episodes,
                                "dagger_rounds": args.rounds,
                                "episodes_per_round": args.episodes_per_round,
                                "observation": args.observation,
                                "architecture": args.architecture},
            "training_domains": [list(x) for x in TRAIN_PROFILE],
            "test_domains": {name: list(config) for name, config in TEST_DOMAINS.items()},
            "paired_runs": paired,
        }
        save_json(args.output, result)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
