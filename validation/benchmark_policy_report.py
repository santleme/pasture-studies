#!/usr/bin/env python3
"""Aggregate a preregistered held-out policy evaluation into benchmark metrics.

The input must be a validation-split report from evaluate_torch_mvp1.py. This
module summarizes paired trained, same-seed untrained, and heuristic outcomes;
it never accesses worlds or locked-test outcomes itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = ("trained", "untrained_same_seed", "heuristic_baseline")
METRICS = ("total_reward", "invalid_actions", "forage_success", "forage_depleted",
           "water_success", "water_blocked")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    """Validate held-out pairing and report per-family and baseline-relative metrics."""
    if report.get("schema_version") != 1 or report.get("paired_world_seeds") is not True:
        raise ValueError("input must be a schema-v1 paired held-out evaluation report")
    split = report.get("seed_split", {})
    if split.get("evaluation_split") != "validation":
        raise ValueError("policy benchmark requires an explicitly preregistered validation split")
    check = report.get("training_validation_test_disjoint_check", {})
    if check.get("pass") is not True or check.get("training_vs_eval_collision_count") != 0:
        raise ValueError("evaluation report does not attest a disjoint held-out split")
    rows = report.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation report has no policy runs")
    by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("policy") not in POLICIES:
            raise ValueError("evaluation contains an unknown or malformed policy row")
        family, seed = row.get("scenario"), row.get("world_seed")
        if not isinstance(family, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("evaluation policy row has an invalid family or world seed")
        cell = by_cell.setdefault((family, seed), {})
        policy = row["policy"]
        if policy in cell:
            raise ValueError("evaluation contains duplicate policy rows for a paired world")
        cell[policy] = row
    if any(set(cell) != set(POLICIES) for cell in by_cell.values()):
        raise ValueError("every held-out world must contain all three paired policies")

    declared = split.get("validation_world_seeds")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("validation report has no preregistered per-family world schedule")
    expected_cells = {(family, int(seed)) for family, seeds in declared.items() for seed in seeds}
    if set(by_cell) != expected_cells:
        raise ValueError("evaluated worlds do not exactly match the preregistered validation schedule")

    families = sorted({family for family, _ in by_cell})
    per_family: dict[str, Any] = {}
    paired_vs_heuristic: dict[str, Any] = {}
    policy_reward_by_family: dict[str, dict[str, float]] = {}
    for family in families:
        family_cells = [(seed, cell) for (name, seed), cell in by_cell.items() if name == family]
        family_result: dict[str, Any] = {"world_count": len(family_cells), "policies": {}}
        policy_reward_by_family[family] = {}
        for policy in POLICIES:
            policy_rows = [cell[policy] for _, cell in family_cells]
            metrics = {"total_reward": _mean([float(row["total_reward"]) for row in policy_rows]),
                       "invalid_actions": _mean([float(row["invalid_actions"]) for row in policy_rows]),
                       **{name: _mean([float(row["reward_components"][name]["count"]) for row in policy_rows])
                          for name in METRICS[2:]}}
            action_totals: dict[str, float] = {}
            for row in policy_rows:
                for action, count in row.get("action_counts", {}).items():
                    action_totals[action] = action_totals.get(action, 0.0) + float(count)
            total_actions = sum(action_totals.values())
            metrics["action_frequencies"] = {key: value / total_actions for key, value in sorted(action_totals.items())} if total_actions else {}
            family_result["policies"][policy] = metrics
            policy_reward_by_family[family][policy] = float(metrics["total_reward"])
        baseline_deltas = {}
        for policy in POLICIES[:2]:
            deltas = [float(cell[policy]["total_reward"]) - float(cell["heuristic_baseline"]["total_reward"])
                      for _, cell in family_cells]
            baseline_deltas[policy] = {"raw": deltas, "mean": _mean(deltas)}
        paired_vs_heuristic[family] = baseline_deltas
        per_family[family] = family_result

    generalization = {}
    reference = "starter" if "starter" in families else families[0]
    for policy in POLICIES:
        reference_reward = policy_reward_by_family[reference][policy]
        generalization[policy] = {"reference_family": reference,
            "mean_reward_by_family": {family: policy_reward_by_family[family][policy] for family in families},
            "reward_gap_from_reference_by_family": {family: policy_reward_by_family[family][policy] - reference_reward
                                                     for family in families}}
    return {"schema_version": 1, "protocol": "pasture-studies-paired-heldout-policy-benchmark-v1",
            "evaluation_split": "validation", "scope": "synthetic simulator policy performance and generalization; not biological validity",
            "training_seed": report.get("training_seed"), "evaluation_manifest_sha256": split.get("manifest_sha256"),
            "heldout_world_count": len(by_cell), "scenario_families": per_family,
            "paired_total_reward_delta_vs_heuristic": paired_vs_heuristic,
            "generalization": generalization,
            "limitations": ["validation results are descriptive and do not unlock locked-test access",
                            "ticks and animals are repeated measures within world episodes",
                            "synthetic simulator outcomes do not establish biological or husbandry validity"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize_evaluation(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "heldout_world_count": summary["heldout_world_count"],
                      "scenario_families": sorted(summary["scenario_families"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
