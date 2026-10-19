#!/usr/bin/env python3
"""Evaluate candidate policies against A-F/untrained/heuristic references on new shared worlds.

Exploratory-only. Uses exported inference checkpoints and the unmodified original
simulator reward. Never loads formal/locked seed manifests or outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farm_sim.model import Action, AnimalMemory, FEATURE_NAMES, MemoryEvent
from farm_sim.policy import CategoricalPolicy, HeuristicBaseline, load_checkpoint
from farm_sim.simulation import Simulation, make_world
from validation.evaluate_torch_mvp1 import (FAMILIES, _initial_water_fixture,
    make_scenario, evaluation_world_schedule, evaluate_episode)
from farm_sim.torch_training import derive_seed

POLICIES = ("trained", "untrained_same_seed", "heuristic_baseline")
REFERENCE_VARIANTS = ("A", "B", "C", "D", "E", "F")
TICKS = 72
NEW_EVALUATION_ROOT = 9484999


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_entry(path: Path, variant: str, seed: int, *, candidate: bool = False) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"checkpoint and sidecar are required: {path}")
    doc = json.loads(sidecar_path.read_text())
    metadata = doc.get("metadata", {})
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if int(doc.get("seed", -1)) != seed or int(doc.get("episodes", -1)) != 512:
        raise ValueError(f"checkpoint seed/episode binding invalid: {path}")
    if candidate:
        if config.get("exploratory_only") is not True or config.get("formal_eligible") is not False:
            raise ValueError(f"candidate checkpoint lacks exploratory-only denial binding: {path}")
        if config.get("locked_outcomes_accessed") is not False:
            raise ValueError(f"candidate checkpoint does not deny locked outcomes: {path}")
    policy = load_checkpoint(path)
    return {"path": path.resolve(), "sidecar": sidecar_path.resolve(), "variant": variant,
            "seed": seed, "sha256": sha_file(path), "sidecar_sha256": sha_file(sidecar_path),
            "config": config, "policy": policy}


def load_candidate_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate checkpoint manifest is unreadable") from exc
    required = (manifest.get("mode") == "autoresearch_exploratory"
                and manifest.get("formal_eligible") is False
                and manifest.get("locked_test_access") is False)
    if not required:
        raise ValueError("candidate manifest must be explicitly nonformal and deny locked data")
    raw = manifest.get("candidate_checkpoints")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("at least three candidate checkpoints are required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if not isinstance(row, dict) or set(row) != {"variant", "seed", "policy_npz", "sha256"}:
            raise ValueError("candidate checkpoint entries have an invalid schema")
        candidate_path = Path(row["policy_npz"])
        if not candidate_path.is_absolute():
            candidate_path = path.parent / candidate_path
        entry = _checkpoint_entry(candidate_path, str(row["variant"]), int(row["seed"]), candidate=True)
        if entry["sha256"] != row["sha256"]:
            raise ValueError(f"candidate checkpoint hash mismatch: {candidate_path}")
        groups[entry["variant"]].append(entry)
    if not groups or any(len(entries) != 3 or len({entry["seed"] for entry in entries}) != 3
                         for entries in groups.values()):
        raise ValueError("each candidate variant must contain exactly three distinct training seeds")
    return dict(groups)


def load_reference_models(root: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for variant in REFERENCE_VARIANTS:
        entries = []
        for folder in sorted((root / variant).glob("seed-*")):
            try:
                seed = int(folder.name.removeprefix("seed-"))
            except ValueError as exc:
                raise ValueError(f"invalid reference seed folder: {folder}") from exc
            entry = _checkpoint_entry(folder / f"torch-a2c-seed-{seed}.npz", variant, seed)
            entries.append(entry)
        if len(entries) != 3:
            raise ValueError(f"historical reference {variant} must contain exactly three verified seeds")
        groups[variant] = entries
    return groups


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        raise ValueError("cannot aggregate an empty evaluation cohort")
    reward_components = Counter()
    actions = Counter()
    nav_failures = 0
    invalid = 0
    final_drives: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reward_components.update({key: int(value["count"]) for key, value in row["reward_components"].items()})
        actions.update({key: int(value) for key, value in row["action_counts"].items()})
        nav = row.get("navigation_subreason_counts", {})
        nav_failures += sum(int(nav.get(key, 0)) for key in (
            "topologically_unreachable", "all_target_cells_occupied", "dynamically_blocked"))
        invalid += int(row["invalid_actions"])
        if set(row["final_drives_by_animal"]) != {f"cow-{i}" for i in range(1, 6)}:
            raise ValueError("evaluation episode did not contain exactly the five expected cattle")
        for key in ("hunger", "thirst", "fatigue", "social_need"):
            final_drives[key].append(float(row["final_drive_means"][key]))
    return {
        "episodes": count,
        "animal_count_per_episode": 5,
        "mean_original_reward_per_episode": sum(float(r["total_reward"]) for r in rows) / count,
        "reward_component_counts": dict(reward_components),
        "mean_successful_forage_per_episode": reward_components["forage_success"] / count,
        "mean_successful_water_per_episode": reward_components["water_success"] / count,
        "mean_action_frequencies": {key: actions[key] / (count * TICKS * 5) for key in Action.__members__.values()},
        "mean_final_drives": {key: sum(values) / len(values) for key, values in final_drives.items()},
        "invalid_actions": invalid,
        "navigation_failure_count": nav_failures,
        "navigation_failures_per_episode": nav_failures / count,
    }


class _FirstInputIntervention:
    """Apply one prespecified feature mask to the first animal's first decision."""
    def __init__(self, policy, feature_index: int):
        self.policy = policy
        self.perception_mode = getattr(policy, "perception_mode", "legacy-v1")
        self.feature_index = feature_index
        self.called = False

    def decide(self, features):
        values = list(features)
        if not self.called:
            values[self.feature_index] = 0.0
            self.called = True
        return self.policy.decide(values)


def _rollout_summary(policy, world) -> dict[str, Any]:
    sim = Simulation(world, policy)
    actions = Counter({action.value: 0 for action in Action})
    reward = 0.0
    for _ in range(TICKS):
        result = sim.step()
        reward += sum(animal.last_reward for animal in result.world.animals)
        actions.update(animal.action.value for animal in result.world.animals)
    final = sim.world
    return {"total_original_reward": float(reward), "action_counts": dict(actions),
            "final_drives": {animal.animal_id: {"hunger": animal.physiology.hunger,
                "thirst": animal.physiology.thirst} for animal in final.animals},
            "final_positions": {animal.animal_id: [animal.position.x, animal.position.y]
                                for animal in final.animals},
            "tick": final.tick}


def _paired_cognition_rollouts(policy) -> dict[str, Any]:
    """Paired 72-tick counterfactuals; one memory or perception input changes."""
    initial = _initial_water_fixture(83831)
    cow = initial.animals[0]
    memory_success = replace(cow, memory=AnimalMemory((MemoryEvent(0, "water_access", True, cow.position),)))
    memory_blocked = replace(cow, memory=AnimalMemory((MemoryEvent(0, "water_access", False, cow.position),)))
    success_world = replace(initial, animals=(memory_success, *initial.animals[1:]))
    blocked_world = replace(initial, animals=(memory_blocked, *initial.animals[1:]))
    memory_base = _rollout_summary(policy, success_world)
    memory_changed = _rollout_summary(policy, blocked_world)

    water_index = FEATURE_NAMES.index("water_available")
    perception_control = _rollout_summary(policy, initial)
    perception_intervened = _rollout_summary(_FirstInputIntervention(policy, water_index), initial)
    return {
        "scope": "paired deterministic 72-tick counterfactual; synthetic fixture; one input intervention on cow-1 decision 1",
        "fixture_seed": 83831,
        "memory": {"changed_only": "latest retrieved water_access success -> blocked at same tick/location",
                   "base": memory_base, "changed": memory_changed,
                   "action_counts_changed": memory_base["action_counts"] != memory_changed["action_counts"],
                   "reward_delta": memory_changed["total_original_reward"] - memory_base["total_original_reward"]},
        "perception": {"changed_only": "water_available feature masked to zero for cow-1 decision 1; world unchanged",
                       "base": perception_control, "changed": perception_intervened,
                       "action_counts_changed": perception_control["action_counts"] != perception_intervened["action_counts"],
                       "reward_delta": perception_intervened["total_original_reward"] - perception_control["total_original_reward"]},
        "claim_boundary": "one-model paired counterfactual evidence only; not statistical causal acceptance or biological cognition evidence",
    }


def compare(*, candidate_manifest: Path, reference_root: Path, output: Path,
            eval_seed: int = NEW_EVALUATION_ROOT, episodes_per_family: int = 3) -> dict[str, Any]:
    if eval_seed == 9483999:
        raise ValueError("the previously inspected 9483999 exploratory worlds cannot be reused for selection")
    if type(eval_seed) is not int or eval_seed < 0 or episodes_per_family != 3:
        raise ValueError("use a fixed nonnegative root seed and exactly three episodes per scenario family")
    candidate_groups = load_candidate_manifest(candidate_manifest)
    references = load_reference_models(reference_root)
    schedule = evaluation_world_schedule(eval_seed=eval_seed, episodes_per_family=episodes_per_family)
    eval_worlds = {seed for _, seed in schedule}
    if len(eval_worlds) != len(schedule):
        raise ValueError("evaluation world schedule repeats a world seed")
    all_groups = {f"candidate-{variant}": entries for variant, entries in candidate_groups.items()}
    all_groups.update({f"reference-{k}": v for k, v in references.items()})
    training_seeds = {entry["seed"] for group in all_groups.values() for entry in group}
    if eval_seed in training_seeds:
        raise ValueError("evaluation root seed collides with a training seed")
    for group in all_groups.values():
        for entry in group:
            config = entry["config"]
            seed = entry["seed"]
            train_worlds = {derive_seed(seed, "train-world", idx) for idx in range(int(config.get("episodes", 512)))}
            if train_worlds.intersection(eval_worlds):
                raise ValueError("evaluation worlds collide with training world schedule")

    output.mkdir(parents=True, exist_ok=True)
    condition_docs: dict[str, Any] = {}
    run_artifacts = []
    for condition, entries in all_groups.items():
        policy_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in POLICIES}
        cognition_rows = []
        for entry in entries:
            trained = entry["policy"]
            seed = entry["seed"]
            if condition.startswith("candidate-"):
                cognition_rows.append({"training_seed": seed,
                                       "paired_counterfactuals": _paired_cognition_rollouts(trained)})
            untrained = CategoricalPolicy.initialize(seed)
            untrained.perception_mode = trained.perception_mode
            heuristic = HeuristicBaseline()
            heuristic.perception_mode = trained.perception_mode
            for family, world_seed in schedule:
                for name, policy in (("trained", trained), ("untrained_same_seed", untrained),
                                     ("heuristic_baseline", heuristic)):
                    row = evaluate_episode(policy, family, world_seed)
                    if name == "trained":
                        policy_rows[name].append(row)
                    else:
                        # The same seed-specific untrained/heuristic baseline is
                        # referenced for each condition but evaluated on each exact
                        # common world, so aggregation retains all 18 reference seeds.
                        policy_rows[name].append(row)
                    row["training_seed"] = seed
                    row["condition"] = condition
                    row["policy"] = name
                    run_artifacts.append(row)
        condition_docs[condition] = {
            "common_worlds": [{"scenario": family, "seed": seed} for family, seed in schedule],
            "checkpoint_seeds": [entry["seed"] for entry in entries],
            "checkpoint_hashes": [{"variant": entry["variant"], "seed": entry["seed"],
                                    "policy_sha256": entry["sha256"],
                                    "sidecar_sha256": entry["sidecar_sha256"]} for entry in entries],
            "policies": {name: _aggregate(rows) for name, rows in policy_rows.items()},
            "paired_cognition_rollouts": cognition_rows,
        }
    doc = {
        "schema_version": 1, "mode": "autoresearch_comparison",
        "title": "Autoresearch candidate vs historical references on a new shared world schedule",
        "formal_eligible": False, "benchmark_eligible": False,
        "locked_test_access": False, "locked_outcomes_accessed": False,
        "reward_mutations": False, "reserved_data_read": False,
        "evaluation_root_seed": eval_seed, "evaluation_reward_source": "original simulator reward",
        "episodes_per_family": episodes_per_family, "ticks_per_episode": TICKS,
        "scenario_families": list(FAMILIES),
        "training_seed_sets": {name: [entry["seed"] for entry in entries]
                                for name, entries in all_groups.items()},
        "seed_disjointness": {"eval_root_distinct_from_training_roots": True,
                              "world_seeds_unique": True,
                              "common_worlds_disjoint_from_training_worlds": True},
        "reward_mutations": False, "reserved_data_read": False,
        "limitations": ["exploratory synthetic software-task outcomes only",
                        "first-decision/rollout probe results are not biological cognition evidence"],
        "conditions": condition_docs,
        "runs": run_artifacts,
    }
    output_path = output / "autoresearch-comparison.json"
    output_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return {"output": str(output_path), "sha256": sha_file(output_path),
            "evaluation_root_seed": eval_seed, "conditions": len(condition_docs),
            "episodes_per_checkpoint": len(schedule),
            "candidate_checkpoints": sum(map(len, candidate_groups.values())),
            "reference_checkpoints": sum(map(len, references.values()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="extracted verified A-F checkpoint archive")
    parser.add_argument("--eval-seed", type=int, default=NEW_EVALUATION_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare(candidate_manifest=args.candidate_manifest,
                             reference_root=args.reference_root, output=args.output,
                             eval_seed=args.eval_seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
