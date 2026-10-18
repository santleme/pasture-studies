#!/usr/bin/env python3
"""Matched-seed exploratory argmax-vs-sampling inference audit for MVP1.

Only inference mode changes. Checkpoints, world seeds, initial states, horizon,
features, toy reward and simulator are fixed. Nothing is trained or promoted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farm_sim.model import Action
from farm_sim.policy import load_checkpoint
from farm_sim.simulation import Simulation, make_world
import farm_sim.simulation as simulation_module

OUT = ROOT / ".omx/research/mvp1-remediation-baseline"
CHECKPOINT = ROOT / "artifacts/policies/cattle-policy-seed-11.npz"
WORLD_SEEDS = (11, 29, 47, 83, 101, 17, 23, 71)
TICKS = 72
BASE_STREAM_SEED = 913_771
ANIMAL_COUNT = 5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CategoricalSamplePolicy:
    """Use independent, reproducible per-animal/per-tick categorical draws."""
    def __init__(self, policy, world_seed):
        self.policy = policy
        self.world_seed = int(world_seed)
        self.call_index = 0
        self.policy_id = f"{policy.policy_id}-categorical-sampling"

    def decide(self, features):
        idx = self.call_index
        self.call_index += 1
        tick, animal_index = divmod(idx, ANIMAL_COUNT)
        if animal_index >= ANIMAL_COUNT:
            raise AssertionError("sample policy called too often within one tick")
        rng = np.random.default_rng(np.random.SeedSequence(
            [BASE_STREAM_SEED, self.world_seed, animal_index, tick]))
        action = self.policy.sample(features, rng)
        probs = self.policy.probabilities(features)
        from farm_sim.model import PolicyDecision
        return PolicyDecision(action, tuple(float(p) for p in probs), self.policy_id)


def valid(action, probs):
    return (action in Action and len(probs) == len(Action) and
            all(np.isfinite(p) and 0 <= p <= 1 for p in probs) and
            abs(sum(probs) - 1.0) <= 1e-6)


def run(seed, policy, sampled):
    sim_policy = CategoricalSamplePolicy(policy, seed) if sampled else policy.copy()
    sim = Simulation(make_world(seed), sim_policy)
    reasons = Counter()
    original_navigate = simulation_module.navigate

    def counted_navigate(*args, **kwargs):
        result = original_navigate(*args, **kwargs)
        reasons[result.reason] += 1
        return result

    simulation_module.navigate = counted_navigate
    output = []
    invalid = 0
    reward_total = 0.0
    run_actions = Counter()
    first_actions = Counter()
    reward_by_tick = []
    drive_sums = Counter()
    try:
        for tick in range(TICKS):
            before = sim.world
            pre_positions = {a.animal_id: [a.position.x, a.position.y] for a in before.animals}
            result = sim.step()
            tick_reward = 0.0
            for animal in sorted(result.world.animals, key=lambda a: a.animal_id):
                action = animal.action
                run_actions[action.value] += 1
                if tick == 0:
                    first_actions[action.value] += 1
                probs = tuple(animal.last_probabilities)
                invalid += int(not valid(action, probs))
                reward_total += animal.last_reward
                tick_reward += animal.last_reward
                for field in ("hunger", "thirst", "fatigue", "social_need"):
                    drive_sums[field] += getattr(animal.physiology, field)
                output.append({"world_seed": seed, "tick": tick, "animal_id": animal.animal_id,
                    "mode": "categorical_sampling" if sampled else "argmax",
                    "features": [float(v) for v in animal.last_features],
                    "probabilities": [float(v) for v in probs], "action": action.value,
                    "position_before": pre_positions[animal.animal_id],
                    "position_after": [animal.position.x, animal.position.y],
                    "reward": float(animal.last_reward),
                    "drives": {field: float(getattr(animal.physiology, field))
                               for field in ("hunger", "thirst", "fatigue", "social_need")}})
            reward_by_tick.append(tick_reward)
    finally:
        simulation_module.navigate = original_navigate

    final = sim.world.animals
    return output, {"world_seed": seed, "mode": "categorical_sampling" if sampled else "argmax",
        "checkpoint_sha256": sha(CHECKPOINT), "ticks": TICKS, "abstract_ticks_per_day": 24,
        "days": TICKS // 24, "decision_count": TICKS * ANIMAL_COUNT,
        "action_counts": dict(run_actions),
        "action_fractions": {action.value: run_actions[action.value] / (TICKS * ANIMAL_COUNT) for action in Action},
        "first_step_action_counts": dict(first_actions),
        "first_step_action_fractions": {action.value: first_actions[action.value] / ANIMAL_COUNT for action in Action},
        "cumulative_toy_reward": reward_total,
        "mean_summed_agent_reward_per_tick": reward_total / TICKS,
        "reward_trajectory_per_tick_all_animals": reward_by_tick,
        "mean_toy_drives_over_all_animal_ticks": {k: v / (TICKS * ANIMAL_COUNT) for k, v in drive_sums.items()},
        "final_mean_toy_drives": {k: sum(getattr(a.physiology, k) for a in final) / ANIMAL_COUNT
                                    for k in ("hunger", "thirst", "fatigue", "social_need")},
        "final_positions": {a.animal_id: [a.position.x, a.position.y] for a in final},
        "navigation_reason_counts": dict(reasons), "invalid_decisions": invalid,
        "sample_calls": sim_policy.call_index if sampled else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template = load_checkpoint(CHECKPOINT)
    outcomes = []
    paired_rows = []
    differences = Counter()
    max_prob_delta_same_features = 0.0
    probabilities_equal_for_same_features = True
    aggregate = {mode: Counter() for mode in ("argmax", "categorical_sampling")}
    for seed in WORLD_SEEDS:
        arg_rows, arg_summary = run(seed, template.copy(), False)
        sample_rows, sample_summary = run(seed, template.copy(), True)
        if len(arg_rows) != len(sample_rows):
            raise AssertionError("paired policy runs produced different decision counts")
        seed_diff = Counter()
        for arg, sample in zip(arg_rows, sample_rows):
            if (arg["world_seed"], arg["tick"], arg["animal_id"]) != (sample["world_seed"], sample["tick"], sample["animal_id"]):
                raise AssertionError("rollout decisions lost paired seed/tick/animal ordering")
            changed = arg["action"] != sample["action"]
            same_features = arg["features"] == sample["features"]
            prob_delta = max(abs(x-y) for x,y in zip(arg["probabilities"], sample["probabilities"]))
            differences["same_feature_pairs"] += int(same_features)
            if same_features:
                max_prob_delta_same_features = max(max_prob_delta_same_features, prob_delta)
                probabilities_equal_for_same_features &= prob_delta <= 1e-12
            differences["decision_pairs"] += 1
            differences["action_different_pairs"] += int(changed)
            seed_diff["decision_pairs"] += 1
            seed_diff["action_different_pairs"] += int(changed)
            paired_rows.append({"world_seed": seed, "tick": arg["tick"], "animal_id": arg["animal_id"],
                "argmax_action": arg["action"], "sampled_action": sample["action"],
                "actions_differ": changed, "argmax_probabilities": arg["probabilities"],
                "sampled_probabilities": sample["probabilities"],
                "features_identical": same_features,
                "max_probability_vector_abs_difference": prob_delta,
                "argmax_position_after": arg["position_after"], "sampled_position_after": sample["position_after"],
                "trajectory_positions_differ": arg["position_after"] != sample["position_after"],
                "argmax_reward": arg["reward"], "sampled_reward": sample["reward"]})
        outcomes.extend((arg_summary, sample_summary))
        for mode, summary in (("argmax", arg_summary), ("categorical_sampling", sample_summary)):
            aggregate[mode]["decisions"] += summary["decision_count"]
            aggregate[mode]["invalid"] += summary["invalid_decisions"]
            aggregate[mode]["reward"] += summary["cumulative_toy_reward"]
            for action, count in summary["action_counts"].items(): aggregate[mode][f"action:{action}"] += count
        differences[f"seed_{seed}_action_different_pairs"] = seed_diff["action_different_pairs"]

    means = {}
    run_n = len(WORLD_SEEDS)
    for mode, values in aggregate.items():
        decisions = values["decisions"]
        means[mode] = {"decision_count": decisions, "invalid_decisions": values["invalid"],
            "mean_run_cumulative_toy_reward": values["reward"] / run_n,
            "action_counts": {a.value: values[f"action:{a.value}"] for a in Action},
            "action_fractions": {a.value: values[f"action:{a.value}"] / decisions for a in Action}}
    report = {"title": "MVP1 trained seed-11 inference-mode diagnostic",
        "generated_by": "validation/diagnose_policy_inference_mvp1.py",
        "command": "python3 validation/diagnose_policy_inference_mvp1.py",
        "scope": {"world_seeds": list(WORLD_SEEDS), "ticks_per_run": TICKS,
            "abstract_ticks_per_day": 24, "animals": ANIMAL_COUNT,
            "changed_factor_only": "inference mode: deterministic argmax vs categorical sample from the exact same policy probabilities",
            "sample_seed_derivation": "NumPy SeedSequence([913771, world_seed, stable cow index in sorted-ID order, tick])",
            "same_checkpoint": True, "checkpoint_sha256": sha(CHECKPOINT),
            "same_seeded_initial_world_per_pair": True, "training_or_checkpoint_mutation": False,
            "reward_or_simulator_mutation": False, "purpose": "exploratory inference audit; not a policy-selection or deployment claim"},
        "per_run_outcomes": outcomes,
        "aggregate_across_matched_runs": means,
        "paired_action_and_probability_differences": {"decision_pairs": differences["decision_pairs"],
            "action_different_pairs": differences["action_different_pairs"],
            "different_action_fraction": differences["action_different_pairs"] / max(1, differences["decision_pairs"]),
            "different_action_pairs_by_seed": {str(seed): differences[f"seed_{seed}_action_different_pairs"] for seed in WORLD_SEEDS},
            "same_feature_pairs": differences["same_feature_pairs"],
            "probability_vectors_equal_when_features_identical": bool(probabilities_equal_for_same_features),
            "max_probability_abs_difference_when_features_identical": max_prob_delta_same_features,
            "max_probability_abs_difference_all_pairs": max((r["max_probability_vector_abs_difference"] for r in paired_rows), default=0.0),
            "trajectory_position_different_pairs": sum(r["trajectory_positions_differ"] for r in paired_rows),
            "downstream_action_rewards_can_differ": True},
        "limitations": ["Categorical sampling changes action realization while preserving the policy distribution; this is a diagnostic contrast, not a trained-policy improvement.",
            "Toy rewards, physiology, and navigation are synthetic software outcomes, not biological measures.",
            "The fixed sample stream is deterministic for this implementation and declared seed derivation."],
        "artifacts": {"paired_decisions_jsonl": "policy-inference-paired-decisions.jsonl"}}
    (OUT / "policy-inference-paired-decisions.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in paired_rows))
    (OUT / "policy-inference-diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str((OUT / "policy-inference-diagnostic.json").relative_to(ROOT)),
        "decision_pairs": differences["decision_pairs"],
        "different_actions": differences["action_different_pairs"],
        "argmax_walk_fraction": means["argmax"]["action_fractions"]["walk"],
        "sample_walk_fraction": means["categorical_sampling"]["action_fractions"]["walk"],
        "max_probability_difference_same_features": report["paired_action_and_probability_differences"]["max_probability_abs_difference_when_features_identical"]}, indent=2))


if __name__ == "__main__":
    main()
