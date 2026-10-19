#!/usr/bin/env python3
"""Fail-closed validator for exploratory five-cattle learning candidates.

All scores must be from the untouched original simulator reward on a shared,
new world schedule. This validator is not a formal/locked or biological gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

POLICIES = ("trained", "untrained_same_seed", "heuristic_baseline")
REQUIRED_EVENTS = ("forage_success", "water_success")
REQUIRED_DRIVES = ("hunger", "thirst")


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _metric(group: dict[str, Any], name: str) -> float:
    episodes = group.get("episodes")
    counts = group.get("reward_component_counts")
    if type(episodes) is not int or episodes <= 0 or not isinstance(counts, dict):
        raise ValueError("aggregate group has invalid episode or reward-component counts")
    return float(counts[name]) / episodes


def _validate_flags(doc: dict[str, Any], label: str, failures: list[str],
                    candidate_condition: str | None = None) -> None:
    if doc.get("mode") != "autoresearch_comparison":
        failures.append(f"{label}: mode is not autoresearch_comparison")
    for key in ("formal_eligible", "benchmark_eligible", "locked_test_access",
                "locked_outcomes_accessed", "reward_mutations", "reserved_data_read"):
        expected = False
        if doc.get(key) is not expected:
            failures.append(f"{label}: {key} must be false")
    split = doc.get("seed_disjointness")
    if not isinstance(split, dict) or any(split.get(key) is not True for key in (
            "eval_root_distinct_from_training_roots", "world_seeds_unique",
            "common_worlds_disjoint_from_training_worlds")):
        failures.append(f"{label}: training/evaluation seed disjointness is not verified")
    if doc.get("evaluation_root_seed") == 9483999:
        failures.append(f"{label}: previously inspected evaluation worlds cannot be reused")
    if doc.get("evaluation_reward_source") != "original simulator reward":
        failures.append(f"{label}: evaluation must use original simulator reward")
    if type(doc.get("ticks_per_episode")) is not int or doc["ticks_per_episode"] != 72:
        failures.append(f"{label}: evaluation horizon must be 72 ticks")
    if len(doc.get("scenario_families", [])) < 4:
        failures.append(f"{label}: at least four held-out scenario families are required")
    if label == "candidate":
        seeds = doc.get("training_seed_sets")
        candidate_seeds = seeds.get(candidate_condition) if isinstance(seeds, dict) else None
        if not isinstance(candidate_seeds, list) or len(candidate_seeds) < 3 or len(set(candidate_seeds)) != len(candidate_seeds):
            failures.append(f"{label}: selected cohort must contain at least three distinct training seeds")


def validate(candidate: dict[str, Any], references: dict[str, Any],
             candidate_condition: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    conditions = candidate.get("conditions")
    if candidate_condition is None:
        candidate_names = [key for key in conditions if key.startswith("candidate-")] if isinstance(conditions, dict) else []
        if "candidate" in (conditions or {}):
            candidate_condition = "candidate"
        elif len(candidate_names) == 1:
            candidate_condition = candidate_names[0]
    _validate_flags(candidate, "candidate", failures, candidate_condition)
    _validate_flags(references, "references", failures)
    if candidate.get("evaluation_root_seed") != references.get("evaluation_root_seed"):
        failures.append("candidate and references use different evaluation root seeds")
    for key in ("episodes_per_family", "ticks_per_episode", "scenario_families"):
        if candidate.get(key) != references.get(key):
            failures.append(f"candidate and references have different {key}")
    reference_conditions = references.get("conditions")
    if not isinstance(conditions, dict) or candidate_condition not in conditions:
        failures.append("candidate comparison lacks the selected candidate condition")
    if not isinstance(reference_conditions, dict):
        failures.append("reference comparison lacks conditions")
    if failures:
        return {"status": "failed", "passed": False, "summary": "candidate failed preregistered exploratory acceptance checks",
                "failures": failures, "formal_eligible": False, "locked_test_access": False}

    candidate_group = conditions[candidate_condition]
    candidate_worlds = candidate_group.get("common_worlds")
    if not isinstance(candidate_worlds, list) or not candidate_worlds:
        failures.append("candidate common evaluation world schedule is missing")
    candidate_policies = candidate_group.get("policies", {})
    group = candidate_policies.get("trained") if isinstance(candidate_policies, dict) else None
    if not isinstance(group, dict):
        failures.append("candidate trained policy aggregate is missing")
        return {"status": "failed", "passed": False, "summary": "candidate evaluation artifact is incomplete",
                "failures": failures, "formal_eligible": False, "locked_test_access": False}
    try:
        for name in ("mean_original_reward_per_episode", "invalid_actions", "navigation_failure_count"):
            if not _finite(group.get(name)):
                failures.append(f"candidate trained group missing finite {name}")
        for event in REQUIRED_EVENTS:
            if not _finite(group.get("reward_component_counts", {}).get(event)):
                failures.append(f"candidate trained group missing event count {event}")
        drives = group.get("mean_final_drives")
        if not isinstance(drives, dict) or any(not _finite(drives.get(key)) for key in REQUIRED_DRIVES):
            failures.append("candidate trained group missing finite final hunger/thirst")
        candidate_metrics = {
            "original_return_per_episode": float(group["mean_original_reward_per_episode"]),
            "forage_successes_per_episode": _metric(group, "forage_success"),
            "water_successes_per_episode": _metric(group, "water_success"),
            "final_hunger": float(drives["hunger"]),
            "final_thirst": float(drives["thirst"]),
            "invalid_actions": float(group["invalid_actions"]),
            "navigation_failures_per_episode": float(group["navigation_failure_count"]) / int(group["episodes"]),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        failures.append(f"candidate aggregate malformed: {exc}")
        return {"status": "failed", "passed": False, "summary": "candidate evaluation artifact is malformed",
                "failures": failures, "formal_eligible": False, "locked_test_access": False}

    reference_groups = []
    for condition_name, condition in reference_conditions.items():
        if condition_name == candidate_condition or condition_name.startswith("candidate-"):
            continue
        worlds = condition.get("common_worlds") if isinstance(condition, dict) else None
        if worlds != candidate_worlds:
            failures.append(f"reference condition {condition_name} uses different evaluation worlds")
            continue
        policies = condition.get("policies", {})
        for policy_name in POLICIES:
            ref_group = policies.get(policy_name) if isinstance(policies, dict) else None
            if isinstance(ref_group, dict):
                reference_groups.append((condition_name, policy_name, ref_group))
    if not reference_groups:
        failures.append("no trained, untrained, or heuristic reference groups are available")
    ref_metrics: dict[str, list[tuple[float, str]]] = {key: [] for key in candidate_metrics}
    for condition_name, policy_name, ref_group in reference_groups:
        try:
            ref_metrics["original_return_per_episode"].append((float(ref_group["mean_original_reward_per_episode"]), f"{condition_name}/{policy_name}"))
            ref_metrics["forage_successes_per_episode"].append((_metric(ref_group, "forage_success"), f"{condition_name}/{policy_name}"))
            ref_metrics["water_successes_per_episode"].append((_metric(ref_group, "water_success"), f"{condition_name}/{policy_name}"))
            drives = ref_group["mean_final_drives"]
            ref_metrics["final_hunger"].append((float(drives["hunger"]), f"{condition_name}/{policy_name}"))
            ref_metrics["final_thirst"].append((float(drives["thirst"]), f"{condition_name}/{policy_name}"))
            ref_metrics["invalid_actions"].append((float(ref_group["invalid_actions"]), f"{condition_name}/{policy_name}"))
            ref_metrics["navigation_failures_per_episode"].append((float(ref_group["navigation_failure_count"])/int(ref_group["episodes"]), f"{condition_name}/{policy_name}"))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            failures.append(f"reference condition {condition_name}/{policy_name} has incomplete required metrics")
    if failures:
        return {"status": "failed", "passed": False, "summary": "candidate comparison failed provenance or metric completeness checks",
                "failures": failures, "formal_eligible": False, "locked_test_access": False}

    gates: dict[str, Any] = {}
    for key in ("original_return_per_episode", "forage_successes_per_episode", "water_successes_per_episode"):
        best_value, best_label = max(ref_metrics[key], key=lambda pair: pair[0])
        passed = candidate_metrics[key] > best_value
        gates[key] = {"candidate": candidate_metrics[key], "best_reference": best_value,
                      "best_reference_label": best_label, "passed": passed,
                      "comparison": "strictly greater"}
        if not passed:
            failures.append(f"candidate did not strictly exceed best reference for {key}")
    for key in ("final_hunger", "final_thirst"):
        best_value, best_label = min(ref_metrics[key], key=lambda pair: pair[0])
        passed = candidate_metrics[key] <= best_value
        gates[key] = {"candidate": candidate_metrics[key], "best_reference": best_value,
                      "best_reference_label": best_label, "passed": passed,
                      "comparison": "no worse than best reference"}
        if not passed:
            failures.append(f"candidate {key} is worse than best reference")
    gates["invalid_actions"] = {"candidate": candidate_metrics["invalid_actions"], "passed": candidate_metrics["invalid_actions"] == 0}
    if candidate_metrics["invalid_actions"] != 0:
        failures.append("candidate emitted invalid actions")
    best_nav = min(ref_metrics["navigation_failures_per_episode"], key=lambda pair: pair[0])
    gates["navigation_failures_per_episode"] = {"candidate": candidate_metrics["navigation_failures_per_episode"],
        "best_reference": best_nav[0], "best_reference_label": best_nav[1],
        "passed": None, "role": "diagnostic_only; reason-coded navigation correctness is covered by pathfinding regression tests"}
    return {"status": "passed" if not failures else "failed", "passed": not failures,
            "summary": "candidate passed all preregistered exploratory gates" if not failures else "candidate failed one or more preregistered exploratory gates",
            "evaluation_root_seed": candidate["evaluation_root_seed"],
            "training_seeds": candidate["training_seed_sets"][candidate_condition],
            "gate_metrics": gates, "failures": failures,
            "formal_eligible": False, "benchmark_eligible": False, "locked_test_access": False,
            "biological_validation": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidate-condition", default=None,
                        help="exact candidate cohort in the comparison, e.g. candidate-G")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(json.loads(args.comparison.read_text()), json.loads(args.references.read_text()),
                          candidate_condition=args.candidate_condition)
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "passed": False, "summary": "unable to read validation inputs",
                  "failures": [str(exc)], "formal_eligible": False, "locked_test_access": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
