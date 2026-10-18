"""Reproducible REINFORCE training for a fixed four-case memory task.

The task is intentionally synthetic. It tests whether a learned policy can use a
retrieved memory bit; it is not a model of cattle behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import Action, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .policy import CategoricalPolicy, save_checkpoint

TRAINING_SEEDS = (11, 29, 47, 83, 101)
EPISODES_PER_RUN = 2_000
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -0.25
LEARNING_RATE = 0.07
ALGORITHM_DESCRIPTION = "episodic REINFORCE with an action-independent exact expected-reward state baseline"


def task_cases() -> tuple[tuple[np.ndarray, int, str], ...]:
    """Four encoded observations, with exactly one feature changed per pair."""
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    water_positive = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    water_positive[names["thirst"]] = 0.9
    water_positive[names["water_visible"]] = 1.0
    water_positive[names["water_available"]] = 1.0
    water_positive[names["water_distance_norm"]] = 0.1
    water_positive[names["mem_water_success"]] = 1.0
    water_blocked = water_positive.copy()
    water_blocked[names["mem_water_success"]] = 0.0

    forage_positive = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    forage_positive[names["hunger"]] = 0.2
    forage_positive[names["forage_visible"]] = 1.0
    forage_positive[names["forage_available"]] = 1.0
    forage_positive[names["forage_distance_norm"]] = 0.1
    forage_positive[names["mem_forage_success"]] = 1.0
    forage_depleted = forage_positive.copy()
    forage_depleted[names["mem_forage_success"]] = 0.0

    result = ((water_positive, Action.DRINK, "water-memory-positive"),
              (water_blocked, Action.WALK, "water-memory-negative"),
              (forage_positive, Action.GRAZE, "forage-memory-positive"),
              (forage_depleted, Action.WALK, "forage-memory-negative"))
    # Guard the causal-pair design against accidental feature leakage.
    for (left, _, _), (right, _, _) in ((result[0], result[1]), (result[2], result[3])):
        changed = np.flatnonzero(left != right)
        assert len(changed) == 1 and FEATURE_NAMES[int(changed[0])] in ("mem_water_success", "mem_forage_success")
    return tuple((x.copy(), tuple(Action).index(action), name) for x, action, name in result)


def train_policy(seed: int, episodes: int = EPISODES_PER_RUN) -> tuple[CategoricalPolicy, CategoricalPolicy, list[float]]:
    """Train one independently seeded policy with episodic, batched REINFORCE."""
    if episodes <= 0 or episodes % 4:
        raise ValueError("episodes must be a positive multiple of four for balanced sampling")
    policy = CategoricalPolicy.initialize(seed)
    initial = policy.copy()
    policy.policy_id = f"reinforce-seed-{seed}"
    rng = np.random.default_rng(seed + 1_000_003)
    cases = task_cases()
    recent_returns: list[float] = []
    n_in, n_hidden, n_out = len(FEATURE_NAMES), policy.hidden_size, len(Action)

    for epoch in range(episodes // 4):
        # Each batch contains each of the four cases exactly once, in seeded order.
        order = rng.permutation(4)
        traces = []
        rewards = []
        for case_index in order:
            x, target, _ = cases[int(case_index)]
            hidden_pre = x @ policy.w1 + policy.b1
            hidden = np.tanh(hidden_pre)
            logits = hidden @ policy.w2 + policy.b2
            shifted = logits - np.max(logits)
            exp = np.exp(shifted)
            probs = exp / exp.sum()
            chosen = int(rng.choice(n_out, p=probs))
            reward = REWARD_CORRECT if chosen == target else REWARD_INCORRECT
            traces.append((x, hidden, probs, chosen))
            rewards.append(reward)

        recent_returns.append(float(np.mean(rewards)))
        # State-value baseline is the exact current expected toy reward for each
        # case. It is independent of sampled action and reduces REINFORCE variance.
        advantages = np.asarray([
            reward - (REWARD_CORRECT * trace[2][target] + REWARD_INCORRECT * (1.0 - trace[2][target]))
            for trace, reward, case_index in zip(traces, rewards, order)
            for target in [cases[int(case_index)][1]]
        ])
        # REINFORCE gradient of log pi(a|s), with action-independent state baseline.
        gw1 = np.zeros_like(policy.w1); gb1 = np.zeros_like(policy.b1)
        gw2 = np.zeros_like(policy.w2); gb2 = np.zeros_like(policy.b2)
        for (x, hidden, probs, chosen), advantage in zip(traces, advantages):
            dlogits = -probs
            dlogits[chosen] += 1.0
            dlogits *= advantage
            gw2 += np.outer(hidden, dlogits)
            gb2 += dlogits
            dhidden = policy.w2 @ dlogits
            dz = dhidden * (1.0 - hidden * hidden)
            gw1 += np.outer(x, dz)
            gb1 += dz
        factor = LEARNING_RATE / 4.0
        grads = [gw1 * factor, gb1 * factor, gw2 * factor, gb2 * factor]
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if not np.isfinite(norm):
            raise FloatingPointError("non-finite REINFORCE gradient")
        clip = min(1.0, 2.0 / max(norm, 1e-12))
        policy.w1 += grads[0] * clip; policy.b1 += grads[1] * clip
        policy.w2 += grads[2] * clip; policy.b2 += grads[3] * clip
        if not np.isfinite(policy.parameter_vector()).all():
            raise FloatingPointError("non-finite policy parameters")

    policy.episodes = episodes
    return policy, initial, recent_returns


def evaluate_matched(initial: CategoricalPolicy, trained: CategoricalPolicy, seed: int) -> dict[str, Any]:
    """20 balanced stochastic draws; use identical categorical uniforms pre/post."""
    rng = np.random.default_rng(100_000 + seed)
    cases = task_cases()
    initial_correct = trained_correct = 0
    per_case: dict[str, dict[str, int]] = {}
    for x, target, label in cases:
        record = per_case[label] = {"count": 0, "initial_correct": 0, "trained_correct": 0}
        for _ in range(5):
            draw = float(rng.random())
            initial_action = _sample_at(initial.probabilities(x), draw)
            trained_action = _sample_at(trained.probabilities(x), draw)
            initial_hit = int(initial_action == target)
            trained_hit = int(trained_action == target)
            initial_correct += initial_hit; trained_correct += trained_hit
            record["count"] += 1
            record["initial_correct"] += initial_hit
            record["trained_correct"] += trained_hit
    return {"eval_seed": 100_000 + seed, "cases": 20,
            "initial_accuracy": initial_correct / 20, "trained_accuracy": trained_correct / 20,
            "improvement_pp": (trained_correct - initial_correct) / 20 * 100,
            "per_case": per_case}


def _sample_at(probabilities: np.ndarray, draw: float) -> int:
    return min(len(probabilities) - 1, int(np.searchsorted(np.cumsum(probabilities), draw, side="right")))


def counterfactual_report(policy: CategoricalPolicy) -> dict[str, Any]:
    cases = task_cases()
    def report_pair(pos_i: int, neg_i: int, pos_action: Action, neg_action: Action) -> dict[str, Any]:
        pos_x, _, pos_name = cases[pos_i]; neg_x, _, neg_name = cases[neg_i]
        pos = policy.probabilities(pos_x); neg = policy.probabilities(neg_x)
        pi, ni = tuple(Action).index(pos_action), tuple(Action).index(neg_action)
        return {"positive_case": pos_name, "negative_case": neg_name,
                "positive_argmax": tuple(Action)[int(np.argmax(pos))].value,
                "negative_argmax": tuple(Action)[int(np.argmax(neg))].value,
                "positive_probabilities": {a.value: float(pos[i]) for i, a in enumerate(Action)},
                "negative_probabilities": {a.value: float(neg[i]) for i, a in enumerate(Action)},
                "positive_contrast": float(pos[pi] - neg[pi]),
                "negative_contrast": float(neg[ni] - pos[ni]),
                "passes": bool(np.argmax(pos) == pi and np.argmax(neg) == ni and
                               pos[pi] - neg[pi] >= 0.20 and neg[ni] - pos[ni] >= 0.20)}
    return {"water_memory": report_pair(0, 1, Action.DRINK, Action.WALK),
            "forage_memory": report_pair(2, 3, Action.GRAZE, Action.WALK)}


def train_suite(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    runs = []
    all_initial = all_trained = 0
    for seed in TRAINING_SEEDS:
        policy, initial, returns = train_policy(seed, EPISODES_PER_RUN)
        prefix = f"cattle-policy-seed-{seed}"
        policy_path = out / f"{prefix}.npz"
        initial_path = out / f"{prefix}.initial.npz"
        metadata = {"algorithm": ALGORITHM_DESCRIPTION, "learning_rate": LEARNING_RATE,
                    "balanced_episodes": EPISODES_PER_RUN, "reward": {"correct": REWARD_CORRECT, "incorrect": REWARD_INCORRECT},
                    "task": "four-case one-step paired memory task", "feature_schema_version": FEATURE_SCHEMA_VERSION}
        save_checkpoint(policy, policy_path, metadata=metadata)
        initial.episodes = 0
        save_checkpoint(initial, initial_path, metadata={**metadata, "role": "captured pre-training initialization"})
        comparison = evaluate_matched(initial, policy, seed)
        cf = counterfactual_report(policy)
        all_initial += int(comparison["initial_accuracy"] * 20)
        all_trained += int(comparison["trained_accuracy"] * 20)
        runs.append({"seed": seed, "episodes": EPISODES_PER_RUN, "checkpoint": str(policy_path),
                     "initial_checkpoint": str(initial_path), "parameter_delta_l2": float(np.linalg.norm(policy.parameter_vector() - initial.parameter_vector())),
                     "mean_batch_reward_first_100": float(np.mean(returns[:100])),
                     "mean_batch_reward_last_100": float(np.mean(returns[-100:])),
                     "evaluation": comparison, "counterfactuals": cf})
    aggregate_initial, aggregate_trained = all_initial / 100, all_trained / 100
    thresholds_pass = aggregate_trained >= 0.85 and aggregate_trained - aggregate_initial >= 0.25
    all_counterfactuals_pass = all(r["counterfactuals"]["water_memory"]["passes"] and r["counterfactuals"]["forage_memory"]["passes"] for r in runs)
    report = {"policy": "NumPy tanh MLP", "algorithm": ALGORITHM_DESCRIPTION,
              "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
              "training_seeds": list(TRAINING_SEEDS), "episodes_per_run": EPISODES_PER_RUN,
              "aggregate": {"cases": 100, "initial_accuracy": aggregate_initial, "trained_accuracy": aggregate_trained,
                            "improvement_pp": (aggregate_trained - aggregate_initial) * 100,
                            "accuracy_threshold": 0.85, "improvement_threshold_pp": 25,
                            "learning_thresholds_pass": thresholds_pass, "all_counterfactuals_pass": all_counterfactuals_pass},
              "runs": runs,
              "claims_boundary": "Synthetic task performance only; no claim of biological validity or real-cattle cognition."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # Seed 11 is the runtime default only if all predeclared learning checks pass.
    if thresholds_pass and all_counterfactuals_pass:
        source = out / "cattle-policy-seed-11.npz"
        import shutil
        shutil.copy2(source, out / "default-policy.npz")
        shutil.copy2(source.with_suffix(".json"), out / "default-policy.json")
        report["default_checkpoint"] = str(out / "default-policy.npz")
        (out / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/policies", help="directory for local NPZ/JSON checkpoints and metrics")
    args = parser.parse_args()
    report = train_suite(args.output)
    print(json.dumps(report, indent=2))
    if not report["aggregate"]["learning_thresholds_pass"] or not report["aggregate"]["all_counterfactuals_pass"]:
        raise SystemExit("policy training plumbing completed, but predeclared learning criteria did not pass")


if __name__ == "__main__":
    main()
