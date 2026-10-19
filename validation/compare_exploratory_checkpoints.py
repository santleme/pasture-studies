#!/usr/bin/env python3
"""Compare exploratory A-D checkpoint groups on one shared synthetic world set.

This is descriptive evaluation only. It never trains a policy, reads formal
split manifests, or uses the training reward as an evaluation score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.sax.saxutils as xml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farm_sim.model import Action
from farm_sim.torch_checkpoint import load_generation, resolve_generation
from farm_sim.torch_training import ActorCritic, derive_seed, require_torch
from validation.evaluate_torch_mvp1 import (
    FAMILIES,
    evaluate,
    evaluation_world_schedule,
    sha_file,
)

CONDITIONS = ("A", "B", "C", "D")
CONDITION_DESCRIPTIONS = {
    "A": "original A2C + original training reward",
    "B": "original reward + entropy/advantage training improvements",
    "C": "state-aware training reward + original A2C",
    "D": "state-aware training reward + entropy/advantage improvements",
}
POLICIES = ("trained", "untrained_same_seed", "heuristic_baseline")


def load_checkpoint_manifest(path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], Path]:
    """Read and validate an explicit exploratory-only A-D, three-seed manifest."""
    manifest_path = Path(path).resolve()
    try:
        doc = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint manifest is missing or invalid JSON") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ValueError("checkpoint manifest must use schema_version 1")
    if (doc.get("mode") != "exploratory_comparison" or doc.get("formal_eligible") is not False
            or doc.get("locked_test_access") is not False):
        raise ValueError("A-D comparison requires explicit exploratory-only and locked-data-denial flags")
    raw = doc.get("conditions")
    if not isinstance(raw, dict) or set(raw) != set(CONDITIONS):
        raise ValueError("conditions must contain exactly A, B, C, and D")

    conditions: dict[str, list[dict[str, Any]]] = {}
    expected_seeds: set[int] | None = None
    for condition in CONDITIONS:
        entries = raw[condition]
        if not isinstance(entries, list) or len(entries) != 3:
            raise ValueError(f"condition {condition} must contain exactly three checkpoint entries")
        normalized = []
        seeds = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"generation", "algorithm", "seed"}:
                raise ValueError(f"condition {condition} entries require generation, algorithm, and seed")
            if entry["algorithm"] != "a2c" or type(entry["seed"]) is not int or entry["seed"] < 0:
                raise ValueError(f"condition {condition} entries must identify an A2C checkpoint and integer seed")
            generation = Path(entry["generation"])
            if not generation.is_absolute():
                generation = manifest_path.parent / generation
            target = resolve_generation(generation)
            checkpoint_manifest = json.loads((target / "manifest.json").read_text())
            config = checkpoint_manifest.get("config", {})
            if checkpoint_manifest.get("algorithm") != "a2c" or config.get("seed") != entry["seed"]:
                raise ValueError(f"condition {condition} checkpoint algorithm/seed binding mismatch: {target}")
            if (config.get("episodes") != 512 or config.get("exploratory_only") is not True
                    or config.get("formal_eligible") is not False
                    or config.get("locked_outcomes_accessed") is not False):
                raise ValueError(f"condition {condition} checkpoint is not a 512-episode exploratory-only A2C run: {target}")
            # Resolve and safely load the committed generation so provenance is
            # checked from the checkpoint payload, not inferred from its name.
            model = ActorCritic(value_head=True, seed=entry["seed"], dtype=require_torch().float32)
            loaded = load_generation(target, model=model, expected_algorithm="a2c")
            state = loaded["state"]
            if (state.get("episode_cursor") != 512 or state.get("exploratory_only") is not True
                    or state.get("formal_eligible") is not False
                    or state.get("locked_outcomes_accessed", False) is not False):
                raise ValueError(f"condition {condition} checkpoint state is not complete/nonformal: {target}")
            if entry["seed"] in seeds:
                raise ValueError(f"condition {condition} has duplicate training seeds")
            seeds.add(entry["seed"])
            normalized.append({"generation": target, "algorithm": "a2c", "seed": entry["seed"]})
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("all four conditions must use the same three training seeds")
        conditions[condition] = normalized
    return conditions, manifest_path


def _mean(rows: list[float]) -> float | None:
    return float(sum(rows) / len(rows)) if rows else None


def _aggregate_policy_rows(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for row in report["runs"]:
            rows[row["policy"]].append(row)
    result: dict[str, Any] = {}
    for policy in POLICIES:
        selected = rows[policy]
        if not selected:
            raise ValueError(f"evaluation reports are missing policy group {policy}")
        keys = selected[0]["action_frequencies"]
        result[policy] = {
            "episodes": len(selected),
            "mean_original_reward_per_episode": _mean([float(row["total_reward"]) for row in selected]),
            "mean_policy_entropy_nats": _mean([float(row["mean_policy_entropy_nats"]) for row in selected]),
            "mean_entropy_of_action_distribution_nats": _mean([float(row["entropy_of_mean_action_distribution_nats"]) for row in selected]),
            "mean_action_frequencies": {key: _mean([float(row["action_frequencies"][key]) for row in selected]) for key in keys},
            "mean_final_drives": {key: _mean([float(row["final_drive_means"][key]) for row in selected]) for key in selected[0]["final_drive_means"]},
            "mean_remaining_resources": {key: _mean([float(row["resource_state"][key]) for row in selected]) for key in selected[0]["resource_state"]},
            "reward_component_counts": {key: sum(int(row["reward_components"][key]["count"]) for row in selected) for key in selected[0]["reward_components"]},
            "invalid_actions": sum(int(row["invalid_actions"]) for row in selected),
        }
    return result


def _svg_plot(ranking: list[dict[str, Any]]) -> str:
    """Render a small dependency-free SVG; ranking is descriptive, not a gate."""
    width, height = 900, 420
    left, right, top, bottom = 230, 35, 45, 55
    values = [row["mean_original_reward_per_episode"] for row in ranking]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    x0, x1 = left, width - right
    scale = (x1 - x0) / (hi - lo)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="28" font-family="sans-serif" font-size="18">Exploratory A–D — mean original reward on common synthetic worlds</text>']
    for idx, row in enumerate(ranking):
        y = top + idx * 72
        x = x0 + (row["mean_original_reward_per_episode"] - lo) * scale
        label = f'{row["condition"]}: {CONDITION_DESCRIPTIONS[row["condition"]]}'
        lines.append(f'<text x="20" y="{y + 20}" font-family="sans-serif" font-size="13">{xml.escape(label)}</text>')
        lines.append(f'<line x1="{x0}" y1="{y + 15}" x2="{x1}" y2="{y + 15}" stroke="#ddd"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y + 15}" r="8" fill="#176b87"/>')
        lines.append(f'<text x="{x + 12:.2f}" y="{y + 20}" font-family="sans-serif" font-size="13">{row["mean_original_reward_per_episode"]:.4f} (n={row["trained_episode_count"]})</text>')
    lines.append(f'<text x="{x0}" y="{height - 20}" font-family="sans-serif" font-size="12">mean original simulator reward / episode (descriptive only; no acceptance threshold)</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def compare( manifest_path: str | Path, *, eval_seed: int, output: str | Path,
             episodes_per_family: int = 3, device: str = "cpu") -> dict[str, Any]:
    if type(eval_seed) is not int or eval_seed < 0:
        raise ValueError("evaluation seed must be a nonnegative integer")
    if type(episodes_per_family) is not int or episodes_per_family <= 0:
        raise ValueError("episodes_per_family must be positive")
    conditions, source_manifest = load_checkpoint_manifest(manifest_path)
    training_seeds = {entry["seed"] for entries in conditions.values() for entry in entries}
    if eval_seed in training_seeds:
        raise ValueError("evaluation root seed must be separate from every training seed")
    schedule = evaluation_world_schedule(eval_seed=eval_seed, episodes_per_family=episodes_per_family)
    eval_worlds = {seed for _, seed in schedule}
    if len(eval_worlds) != len(schedule):
        raise ValueError("evaluation schedule contains duplicate common worlds")
    for entries in conditions.values():
        for entry in entries:
            checkpoint_manifest = json.loads((entry["generation"] / "manifest.json").read_text())
            train_worlds = {derive_seed(entry["seed"], "train-world", i)
                            for i in range(int(checkpoint_manifest["config"]["episodes"]))}
            if train_worlds.intersection(eval_worlds):
                raise ValueError("common evaluation worlds collide with a checkpoint training split")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_checkpoint = []
    ranking = []
    condition_reports: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        reports = []
        for entry in conditions[condition]:
            seed = entry["seed"]
            report_path = output_dir / condition / f"seed-{seed}-evaluation.json"
            evaluate(entry["generation"], algorithm="a2c", seed=seed, eval_seed=eval_seed,
                     output=report_path, episodes_per_family=episodes_per_family, device=device)
            report = json.loads(report_path.read_text())
            # ``evaluate_torch_mvp1`` writes this field as ``seed_split``;
            # checking the legacy/nonexistent ``split_seed`` key made every
            # otherwise valid common-world evaluation fail after its first
            # checkpoint had completed.
            if report.get("seed_split", {}).get("evaluation_root_seed") != eval_seed:
                raise ValueError("underlying evaluation report did not preserve the shared evaluation root seed")
            reports.append(report)
            per_checkpoint.append({"condition": condition, "training_seed": seed,
                                   "generation": str(entry["generation"]),
                                   "generation_manifest_sha256": report["generation_manifest_sha256"],
                                   "report": str(report_path), "report_sha256": sha_file(report_path)})
        condition_reports[condition] = reports
        aggregate = _aggregate_policy_rows(reports)
        trained = aggregate["trained"]
        ranking.append({"condition": condition, "description": CONDITION_DESCRIPTIONS[condition],
                        "mean_original_reward_per_episode": trained["mean_original_reward_per_episode"],
                        "trained_episode_count": trained["episodes"]})

    ranking.sort(key=lambda row: row["mean_original_reward_per_episode"], reverse=True)
    rank_by_condition = {row["condition"]: idx + 1 for idx, row in enumerate(ranking)}
    conditions_doc = {}
    for condition in CONDITIONS:
        conditions_doc[condition] = {
            "description": CONDITION_DESCRIPTIONS[condition],
            "descriptive_rank_by_mean_original_reward": rank_by_condition[condition],
            "policies": _aggregate_policy_rows(condition_reports[condition]),
            "memory_perception_interventions": [
                {"training_seed": entries[idx]["seed"], "probe": report["cognition_influence"]}
                for idx, report in enumerate(condition_reports[condition])],
            "common_worlds": [{"scenario": family, "seed": seed} for family, seed in schedule],
        }
    summary_path = output_dir / "exploratory-comparison.json"
    plot_path = output_dir / "exploratory-comparison.svg"
    doc = {
        "schema_version": 1,
        "mode": "exploratory_comparison",
        "title": "A-D common-world checkpoint comparison",
        "formal_eligible": False,
        "benchmark_eligible": False,
        "locked_test_access": False,
        "training_executed": False,
        "evaluation_reward_source": "original simulator reward from per-animal step rewards; training reward/config is not used for evaluation scoring",
        "evaluation_root_seed": eval_seed,
        "evaluation_world_seed_derivation": "sha256(root,heldout,family,index)",
        "episodes_per_family": episodes_per_family,
        "scenario_families": list(FAMILIES),
        "ticks_per_episode": 72,
        "training_seed_sets": {condition: sorted(entry["seed"] for entry in conditions[condition]) for condition in CONDITIONS},
        "seed_disjointness": {"eval_root_distinct_from_training_roots": True,
                              "world_seeds_unique": True,
                              "common_worlds_disjoint_from_training_worlds": True},
        "ranking_method": "descending descriptive mean original simulator reward per trained episode; no acceptance threshold or inference",
        "ranking": ranking,
        "conditions": conditions_doc,
        "checkpoint_reports": per_checkpoint,
        "checkpoint_manifest_sha256": sha_file(source_manifest),
        "runner_source_sha256": sha_file(Path(__file__)),
        "limitations": ["synthetic software-task outcomes only; not biological or husbandry evidence",
                        "exploratory comparison is not formal validation, hypothesis testing, selection, or promotion",
                        "memory/perception probes are paired first-decision input interventions, not full-trajectory causal effects"],
    }
    summary_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    plot_path.write_text(_svg_plot(ranking), encoding="utf-8")
    return {"output": str(summary_path), "plot": str(plot_path), "conditions": len(conditions),
            "checkpoints": len(per_checkpoint), "episodes_per_checkpoint_policy": len(schedule),
            "episode_policy_rows_total": len(per_checkpoint) * len(schedule) * len(POLICIES),
            "report_sha256": sha_file(summary_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, type=Path,
                        help="exploratory-only JSON manifest with three A2C checkpoints for each A-D")
    parser.add_argument("--eval-seed", required=True, type=int,
                        help="independent shared evaluation root seed; must not be a training seed")
    parser.add_argument("--episodes-per-family", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.checkpoints, eval_seed=args.eval_seed, output=args.output,
                             episodes_per_family=args.episodes_per_family, device=args.device), indent=2))


if __name__ == "__main__":
    main()
