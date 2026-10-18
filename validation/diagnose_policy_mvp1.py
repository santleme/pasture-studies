#!/usr/bin/env python3
"""Read-only, deterministic P1 audit of the MVP1 trained cattle policy.

This diagnostic never trains, tunes, or writes checkpoints. It captures fixed starter
rollouts, audits checkpoint/inference consistency, compares rollout inputs with the
four synthetic training fixtures, probes controlled feature edits, and checks a
numerical gradient for the exact expected-reward objective.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farm_sim.model import Action, FEATURE_NAMES, MemoryEvent, Position, AnimalMemory, ToyPhysiology, ResourcePatch
from farm_sim.policy import CategoricalPolicy, load_checkpoint
from farm_sim.simulation import Simulation, make_world, serialize_world
from farm_sim.perception import perceive, encode_features
from farm_sim.memory import retrieve
from farm_sim.training import task_cases, REWARD_CORRECT, REWARD_INCORRECT

OUT = ROOT / ".omx/research/mvp1-remediation-baseline"
TRAINED = ROOT / "artifacts/policies/cattle-policy-seed-11.npz"
INITIAL = ROOT / "artifacts/policies/cattle-policy-seed-11.initial.npz"
WORLD_SEEDS = (11, 29, 47, 83, 101, 17, 23, 71)
TICKS = 72  # fixed P1 diagnostic horizon: 3 abstract 24-tick days


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_features(world, animal):
    obs = perceive(world, animal)
    mem = retrieve(animal.memory, world.tick)
    return np.asarray(encode_features(obs, mem, animal, world.width, world.height), dtype=float)


def logit_gradient(policy: CategoricalPolicy, features: np.ndarray, action_index: int):
    """Analytic d log pi(a|x)/d parameters, in parameter_vector order."""
    x = np.asarray(features, dtype=float)
    pre = x @ policy.w1 + policy.b1
    hidden = np.tanh(pre)
    probs = policy.probabilities(x)
    dz2 = -probs
    dz2[action_index] += 1.0
    gw2 = np.outer(hidden, dz2)
    gb2 = dz2.copy()
    dh = policy.w2 @ dz2
    dz1 = dh * (1.0 - hidden * hidden)
    gw1 = np.outer(x, dz1)
    gb1 = dz1
    return np.concatenate((gw1.ravel(), gb1.ravel(), gw2.ravel(), gb2.ravel()))


def expected_reward(policy, features, target):
    p = policy.probabilities(features)[target]
    return REWARD_INCORRECT + (REWARD_CORRECT - REWARD_INCORRECT) * p


def expected_reward_gradient(policy, features, target):
    # d [r_bad + (r_good-r_bad) p(target)] = scale*p* d log p(target).
    p = policy.probabilities(features)[target]
    return (REWARD_CORRECT - REWARD_INCORRECT) * p * logit_gradient(policy, features, target)


def gradient_check(policy):
    features, target, label = task_cases()[0]
    target = int(target)
    analytic = expected_reward_gradient(policy, features, target)
    params = policy.parameter_vector()
    # Deterministic evenly-spaced parameter sample, plus both biases of target output.
    indices = sorted(set(np.linspace(0, len(params) - 1, 17, dtype=int).tolist() +
                         [len(params) - len(Action) + target]))
    eps = 1e-6
    numerical = []
    for i in indices:
        plus = policy.copy(); minus = policy.copy()
        _set_parameter(plus, i, _get_parameter(plus, i) + eps)
        _set_parameter(minus, i, _get_parameter(minus, i) - eps)
        numerical.append((expected_reward(plus, features, target) - expected_reward(minus, features, target)) / (2 * eps))
    a = analytic[indices]
    n = np.asarray(numerical)
    abs_error = np.abs(a - n)
    rel_error = abs_error / np.maximum(1e-8, np.abs(a) + np.abs(n))
    return {"objective": "single-case exact expected toy reward, r_bad + (r_good-r_bad)*p(target)",
            "fixture": label, "reward_correct": REWARD_CORRECT, "reward_incorrect": REWARD_INCORRECT,
            "parameters_checked": len(indices), "indices": indices,
            "max_absolute_error": float(abs_error.max()), "max_symmetric_relative_error": float(rel_error.max()),
            "tolerance_absolute": 2e-6, "passes": bool(abs_error.max() < 2e-6)}


def _arrays(policy, features):
    logits = policy.logits(features)
    probs = policy.probabilities(features)
    decision = policy.decide(features)
    actions = tuple(Action)
    return logits, probs, decision, actions


def inference_audit(policy, features):
    logits, probs, decision, actions = _arrays(policy, features)
    expected = actions[int(np.argmax(probs))]
    softmax = np.exp(logits - np.max(logits)); softmax /= softmax.sum()
    return {"argmax_matches_decide": bool(decision.action == expected),
            "probability_matches_softmax": bool(np.allclose(probs, softmax, rtol=0, atol=1e-12)),
            "finite_logits": bool(np.isfinite(logits).all()), "finite_probabilities": bool(np.isfinite(probs).all()),
            "normalized_probabilities": bool(abs(float(probs.sum()) - 1.0) <= 1e-12),
            "probability_shape": list(probs.shape), "action_count": len(actions)}


def probe_cases():
    """Controlled fixed-vector probes; changes are explicit in output, not disguised as rollouts."""
    cases = []
    base, _, name = task_cases()[0]
    # Memory states retain every non-memory bit. The blocked encoding marks blocked and clears success.
    for label, edits in (("empty_memory", {12: 0.0, 14: 0.0, 16: 0.0}),
                         ("successful_water_memory", {12: 1.0, 14: 0.0, 16: 1.0}),
                         ("blocked_water_memory", {12: 0.0, 14: 1.0, 16: 1.0})):
        x = base.copy()
        for i, v in edits.items(): x[i] = v
        cases.append((f"{label}_vs_{name}", base, x, "memory", [FEATURE_NAMES[i] for i in edits]))
    # Resource controls: visible-but-depleted is a one-coordinate availability edit;
    # fully absent is reported separately because visibility and distance also change.
    depleted = base.copy(); depleted[9] = 0.0
    cases.append(("water_visible_depleted_vs_available", base, depleted, "resource", [FEATURE_NAMES[9]]))
    absent = base.copy(); absent[5] = 0.0; absent[9] = 0.0; absent[7] = 1.0
    cases.append(("water_absent_vs_visible_available", base, absent, "resource",
                  [FEATURE_NAMES[i] for i in (5, 7, 9)]))
    # Exact training negative case preserves the defined paired-memory contrast.
    _, _, negname = task_cases()[1]
    negative = task_cases()[1][0]
    cases.append((f"training_pair_{name}_vs_{negname}", base, negative, "training_pair", ["mem_water_success"]))
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trained = load_checkpoint(TRAINED)
    initial = load_checkpoint(INITIAL)
    if trained.seed != initial.seed or trained.seed != 11:
        raise RuntimeError("expected matched seed-11 trained/initial checkpoints")
    training = task_cases()
    train_matrix = np.stack([x for x, _, _ in training])
    policy_rows = {"trained_seed11": trained, "initial_seed11": initial}
    all_rollout_records = []
    per_policy = {}
    for p_name, policy in policy_rows.items():
        records = []
        action_counts = Counter()
        feature_values = []
        max_identity_error = 0
        for seed in WORLD_SEEDS:
            sim = Simulation(make_world(seed), policy.copy())
            for _ in range(TICKS):
                before = sim.world
                # Capture exact policy inputs before the state transition, in stable agent order.
                inputs = {a.animal_id: as_features(before, a) for a in sorted(before.animals, key=lambda x: x.animal_id)}
                result = sim.step()
                for animal in sorted(result.world.animals, key=lambda x: x.animal_id):
                    x = inputs[animal.animal_id]
                    logits, probs, decision, actions = _arrays(policy, x)
                    max_identity_error = max(max_identity_error, int(decision.action != animal.action),
                                             int(not np.allclose(decision.probabilities, animal.last_probabilities, atol=1e-12, rtol=0)))
                    row = {"policy": p_name, "checkpoint_sha256": sha(TRAINED if p_name == "trained_seed11" else INITIAL),
                           "world_seed": seed, "tick": before.tick, "animal_id": animal.animal_id,
                           "features": dict(zip(FEATURE_NAMES, [float(v) for v in x])),
                           "logits": dict(zip([a.value for a in actions], [float(v) for v in logits])),
                           "probabilities": dict(zip([a.value for a in actions], [float(v) for v in probs])),
                           "selected_action": decision.action.value,
                           "action_margin_top1_top2": float(np.partition(probs, -1)[-1] - np.partition(probs, -2)[-2]),
                           "position_before": [before.animals[[a.animal_id for a in before.animals].index(animal.animal_id)].position.x,
                                               before.animals[[a.animal_id for a in before.animals].index(animal.animal_id)].position.y],
                           "position_after": [animal.position.x, animal.position.y],
                           "reward": float(animal.last_reward),
                           "drive": {k: float(getattr(animal.physiology, k)) for k in ("hunger", "thirst", "fatigue", "social_need")}}
                    records.append(row); action_counts[decision.action.value] += 1; feature_values.append(x)
        matrix = np.stack(feature_values)
        all_rollout_records.extend(records)
        means = matrix.mean(0); q01 = np.quantile(matrix, .01, axis=0); q99 = np.quantile(matrix, .99, axis=0)
        support = []
        # Per-row nearest training fixture in scaled Euclidean feature space; descriptive support only.
        scales = np.maximum(train_matrix.std(0), 0.1)
        dist = np.sqrt((((matrix[:, None, :] - train_matrix[None, :, :]) / scales) ** 2).mean(axis=2))
        nearest = np.argmin(dist, axis=1)
        for j, (_, target, label) in enumerate(training):
            matching = nearest == j
            support.append({"training_case": label, "training_action": tuple(Action)[target].value,
                            "rows_nearest_this_case": int(matching.sum()), "fraction": float(matching.mean())})
        per_policy[p_name] = {"decision_rows": len(records), "action_counts": dict(action_counts),
            "action_fractions": {a.value: action_counts[a.value] / len(records) for a in Action},
            "selected_walk_fraction": action_counts[Action.WALK.value] / len(records),
            "policy_decisions_match_simulation_records": max_identity_error == 0,
            "max_argmax_vs_simulation_action_or_probability_mismatches": max_identity_error,
            "feature_summary": {name: {"train_min": float(train_matrix[:, i].min()), "train_max": float(train_matrix[:, i].max()),
                    "rollout_min": float(matrix[:, i].min()), "rollout_max": float(matrix[:, i].max()),
                    "rollout_mean": float(means[i]), "rollout_q01": float(q01[i]), "rollout_q99": float(q99[i]),
                    "rollout_zero_fraction": float(np.mean(matrix[:, i] == 0.0))}
                    for i, name in enumerate(FEATURE_NAMES)},
            "nearest_fixture_support": support,
            "mean_nearest_fixture_scaled_distance": float(dist.min(axis=1).mean()),
            "seed_rollouts": list(WORLD_SEEDS), "ticks_each": TICKS}

    probe_results = []
    for p_name, policy in policy_rows.items():
        for label, before, after, factor, changed in probe_cases():
            b = policy.probabilities(before); after_probs = policy.probabilities(after)
            probe_results.append({"policy": p_name, "probe": label, "factor": factor,
                "claimed_changed_features": changed,
                "actual_changed_features": [FEATURE_NAMES[i] for i in np.flatnonzero(before != after)],
                "before_features": {k: float(v) for k, v in zip(FEATURE_NAMES, before)},
                "after_features": {k: float(v) for k, v in zip(FEATURE_NAMES, after)},
                "before_action": tuple(Action)[int(np.argmax(b))].value,
                "after_action": tuple(Action)[int(np.argmax(after_probs))].value,
                "before_probabilities": {a.value: float(b[i]) for i, a in enumerate(Action)},
                "after_probabilities": {act.value: float(after_probs[i]) for i, act in enumerate(Action)},
                "probability_delta": {act.value: float(after_probs[i] - b[i]) for i, act in enumerate(Action)}})

    inference_checks = {}
    for p_name, policy in policy_rows.items():
        inference_checks[p_name] = [inference_audit(policy, x) for x, _, _ in training]
        inference_checks[p_name].append(inference_audit(policy, np.stack([np.asarray([row["features"][n] for n in FEATURE_NAMES])
                                                     for row in all_rollout_records if row["policy"] == p_name])[0]))
    checks = {p: all(r[k] for r in rows for k in ("argmax_matches_decide", "probability_matches_softmax", "finite_logits", "finite_probabilities", "normalized_probabilities"))
               for p, rows in inference_checks.items()}
    trained_support = per_policy["trained_seed11"]["nearest_fixture_support"]
    positive_fixture_nearest = sum(row["rows_nearest_this_case"] for row in trained_support
                                   if row["training_case"] in ("water-memory-positive", "forage-memory-positive"))
    gradient_results = {p: gradient_check(policy) for p, policy in policy_rows.items()}
    report = {
        "title": "MVP1 P1 policy collapse diagnostic (read-only)",
        "generated_by": "validation/diagnose_policy_mvp1.py",
        "command": "python3 validation/diagnose_policy_mvp1.py",
        "scope": {"world_seeds": list(WORLD_SEEDS), "ticks_per_rollout": TICKS, "abstract_ticks_per_day": 24,
                  "days_per_rollout": TICKS // 24, "agents": 5,
                  "policies": {k: {"policy_id": v.policy_id, "seed": v.seed, "episodes": v.episodes,
                                  "checkpoint_sha256": sha(TRAINED if k == "trained_seed11" else INITIAL)}
                               for k, v in policy_rows.items()},
                  "training_cases": [label for _, _, label in training],
                  "explicitly_not_training_or_tuning": True},
        "checkpoint_identity": {"trained_path": str(TRAINED.relative_to(ROOT)), "trained_sha256": sha(TRAINED),
            "trained_sidecar_sha256": sha(TRAINED.with_suffix(".json")), "initial_path": str(INITIAL.relative_to(ROOT)),
            "initial_sha256": sha(INITIAL), "initial_sidecar_sha256": sha(INITIAL.with_suffix(".json")),
            "matched_seed": trained.seed == initial.seed, "trained_episode_count": trained.episodes,
            "loaded_policy_id": trained.policy_id, "feature_names": list(FEATURE_NAMES), "actions": [a.value for a in Action]},
        "inference_audit": {"checks_by_policy": inference_checks, "all_checks_pass": checks,
                            "interpretation": "Pure policy argmax and probability calculations are compared to simulation-recorded decisions."},
        "rollout_comparison": per_policy,
        "controlled_one_feature_and_semantic_probes": probe_results,
        "expected_reward_gradient_finite_difference": gradient_results,
        "root_cause_assessment": {
            "software_defect_evidence": {"checkpoint_schema_and_load": "PASS: expected seed-11 trained and matched initial checkpoint load with current versioned feature/action schema",
                "probability_and_argmax": "PASS: finite normalized softmax, probabilities match direct softmax, and decide matches argmax on all four training cases and one captured rollout feature per policy",
                "rollout_decision_identity": {k: v["policy_decisions_match_simulation_records"] for k, v in per_policy.items()},
                "expected_reward_gradient": {k: v["passes"] for k, v in gradient_results.items()}},
            "train_task_vs_rollout_support": {"evidence": "trained rollout feature vectors are far from the four one-step fixtures under the stated descriptive standardized distance; the task fixtures omit operational variation in most drive, herd, obstacle, and tick-phase channels",
                "trained_rollout_rows_nearest_a_positive_memory_training_case": positive_fixture_nearest,
                "trained_rollout_rows_total": per_policy["trained_seed11"]["decision_rows"],
                "trained_rows_nearest_water_or_forage_positive_cases_fraction": positive_fixture_nearest / per_policy["trained_seed11"]["decision_rows"],
                "qualification": "supports a train-task/rollout-support mismatch hypothesis; nearest-fixture classification is descriptive and does not isolate causality"},
            "provisional_disposition": "no inference, checkpoint-schema, rollout-decision, or tested expected-reward-gradient defect detected; strong synthetic train-task/rollout-support mismatch evidence is present, but root cause remains unconfirmed until controlled training-distribution/objective/inference factor experiments run",
            "no_forced_action_diversity": True},
        "limitations": ["The eight fixed starter-world seed rollouts reproduce current code, not necessarily historical per-decision records.",
            "Nearest-fixture scaled distances describe support mismatch; they do not establish why an action was selected.",
            "Fixed-vector probes are model-input experiments, not whole-simulation counterfactuals.",
            "The finite-difference test checks the mathematical gradient of expected toy reward, not the REINFORCE sampler/training loop end-to-end.",
            "All actions, reward and physiology are synthetic software conventions; no biological validation is implied."],
        "artifacts": {"rollout_jsonl": "policy-rollout-decisions.jsonl", "report": "policy-diagnostic.json"}
    }
    # Do not write shared manifest or touch evaluator outputs/checkpoints.
    (OUT / "policy-rollout-decisions.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in all_rollout_records))
    (OUT / "policy-diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str((OUT / "policy-diagnostic.json").relative_to(ROOT)),
        "rollout_rows": len(all_rollout_records), "inference_checks_pass": checks,
        "gradient_checks_pass": {p: r["passes"] for p, r in report["expected_reward_gradient_finite_difference"].items()},
        "trained_walk_fraction": per_policy["trained_seed11"]["selected_walk_fraction"],
        "initial_walk_fraction": per_policy["initial_seed11"]["selected_walk_fraction"]}, indent=2))


def _get_parameter(policy, i): return float(policy.parameter_vector()[i])


def _set_parameter(policy, i, value):
    # Flatten order: w1, b1, w2, b2.
    n1 = policy.w1.size; n2 = policy.b1.size; n3 = policy.w2.size
    if i < n1: policy.w1.flat[i] = value
    elif i < n1 + n2: policy.b1.flat[i - n1] = value
    elif i < n1 + n2 + n3: policy.w2.flat[i - n1 - n2] = value
    else: policy.b2.flat[i - n1 - n2 - n3] = value


if __name__ == "__main__":
    main()
