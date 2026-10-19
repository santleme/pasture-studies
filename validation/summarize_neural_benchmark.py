#!/usr/bin/env python3
"""Summarize paired descriptive neural-policy evaluation reports.

This is a reporting helper, not an experiment runner or an acceptance test. It
accepts validation reports only and never accesses locked-test worlds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

FAMILIES = ("starter", "resource_scarce_relocated", "wide", "unseen_corridor")
POLICIES = ("trained", "untrained_same_seed", "heuristic_baseline")
NOVEL_FAMILIES = FAMILIES[1:]


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Create reproducible policy/family aggregates and matched baseline deltas."""
    split = report.get("seed_split", {})
    split_name = split.get("evaluation_split")
    if split_name == "locked_test":
        raise ValueError("locked_test reports cannot be summarized by the descriptive benchmark")
    if split_name not in (None, "validation"):
        raise ValueError(f"unsupported evaluation split {split_name!r}")
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("evaluation report must contain non-empty runs")

    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    by_family_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        family, policy = row.get("scenario"), row.get("policy")
        if family not in FAMILIES or policy not in POLICIES:
            raise ValueError("run contains an unknown scenario family or policy")
        world_seed = row.get("world_seed")
        if isinstance(world_seed, bool) or not isinstance(world_seed, int):
            raise ValueError("run world_seed must be an integer")
        key = (family, world_seed, policy)
        if key in indexed:
            raise ValueError(f"duplicate run for {key}")
        total_reward = row.get("total_reward")
        if isinstance(total_reward, bool) or not isinstance(total_reward, (int, float)) or not math.isfinite(total_reward):
            raise ValueError("run total_reward must be finite")
        if row.get("ticks", 0) <= 0 or sum(row.get("action_counts", {}).values()) <= 0:
            raise ValueError("run must contain positive tick and decision counts")
        indexed[key] = row
        by_family_policy[(family, policy)].append(row)

    worlds = {(family, seed) for family, seed, _ in indexed}
    for family, seed in worlds:
        missing = [policy for policy in POLICIES if (family, seed, policy) not in indexed]
        if missing:
            raise ValueError(f"paired world {family}:{seed} is missing policies: {missing}")

    families: dict[str, Any] = {}
    for family in FAMILIES:
        policy_summary = {}
        for policy in POLICIES:
            rows = by_family_policy.get((family, policy), [])
            if not rows:
                raise ValueError(f"evaluation has no {policy} runs for {family}")
            rewards = [float(row["total_reward"]) for row in rows]
            decisions = [sum(row["action_counts"].values()) for row in rows]
            policy_summary[policy] = {
                "episodes": len(rows),
                "mean_total_reward": mean(rewards),
                "sd_total_reward_population": pstdev(rewards),
                "mean_reward_per_decision": sum(rewards) / sum(decisions),
                "mean_policy_entropy_nats": _mean(rows, "mean_policy_entropy_nats"),
                "invalid_action_rate": sum(int(row["invalid_actions"]) for row in rows) / sum(decisions),
                "mean_action_frequencies": {
                    action: mean(float(row["action_frequencies"][action]) for row in rows)
                    for action in sorted(rows[0]["action_frequencies"])
                },
            }
        paired_deltas = {}
        for comparator in POLICIES[1:]:
            differences = [float(indexed[(family, seed, "trained")] ["total_reward"])
                           - float(indexed[(family, seed, comparator)]["total_reward"])
                           for f, seed in worlds if f == family]
            paired_deltas[f"trained_minus_{comparator}"] = {
                "paired_worlds": len(differences), "mean_total_reward_delta": mean(differences),
                "sd_total_reward_delta_population": pstdev(differences),
            }
        families[family] = {"policies": policy_summary, "paired_reward_deltas": paired_deltas}

    trained = families
    starter_reward = trained["starter"]["policies"]["trained"]["mean_total_reward"]
    novel_rewards = [trained[family]["policies"]["trained"]["mean_total_reward"]
                     for family in NOVEL_FAMILIES]
    return {
        "schema_version": 1,
        "report_kind": "descriptive_neural_policy_benchmark_summary",
        "evaluation_split": split_name or "unregistered_descriptive",
        "scenario_families": list(FAMILIES),
        "novel_world_families": list(NOVEL_FAMILIES),
        "seed_split_sha256": split.get("sha256"),
        "source_report_generation_manifest_sha256": report.get("generation_manifest_sha256"),
        "families": families,
        "generalization": {
            "trained_mean_reward_starter": starter_reward,
            "trained_mean_reward_novel_family_average": mean(novel_rewards),
            "trained_novel_minus_starter_reward": mean(novel_rewards) - starter_reward,
            "novel_family_mean_rewards": {family: trained[family]["policies"]["trained"]["mean_total_reward"]
                                          for family in NOVEL_FAMILIES},
        },
        "interpretation": "Descriptive synthetic-simulator metrics only; paired deltas and generalization gaps are not inferential acceptance tests or biological evidence.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON output from evaluate_torch_mvp1.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = summarize(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "families": len(summary["families"]),
                      "evaluation_split": summary["evaluation_split"]}, indent=2))


if __name__ == "__main__":
    main()
