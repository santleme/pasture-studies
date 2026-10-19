#!/usr/bin/env python3
"""Build and verify reproducible training-only Pasture Studies dataset bundles.

The bundle includes procedural world snapshots, BFS navigation-oracle route
records, and one-pass DAgger-style queried labels from the existing heuristic
teacher. It makes no biological-validity or optimal-expert claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from farm_sim.model import Action, FEATURE_NAMES, Position
from farm_sim.policy import CategoricalPolicy, HeuristicBaseline
from farm_sim.simulation import Simulation, make_world, serialize_world
from farm_sim.torch_training import derive_seed
from validation.diagnose_navigation_mvp1 import oracle_path
from validation.preregister_torch_mvp1 import FAMILIES, OUTPUT_PATH, verify_manifest


DATA_FILES = ("procedural-worlds.jsonl", "oracle-trajectories.jsonl", "dagger-trajectories.jsonl")
SOURCE_FILES = ("farm_sim/model.py", "farm_sim/policy.py", "farm_sim/simulation.py",
                "farm_sim/navigation.py", "farm_sim/perception.py", "farm_sim/memory.py",
                "farm_sim/physiology.py", "validation/diagnose_navigation_mvp1.py",
                "validation/pasture_studies_dataset.py")
DATASET_SCHEMA_VERSION = 1
TICKS_PER_EPISODE = 72


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _line_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)


def _heldout_seeds(manifest: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for split_name in ("validation", "precision_pilot", "locked_test", "precision_reserve"):
        for seeds in manifest["seed_splits"][split_name].values():
            result.update(int(seed) for seed in seeds)
    return result


def _training_world_ids(manifest: dict[str, Any], root_seeds: list[int], episodes_per_root: int) -> list[tuple[int, int, int]]:
    split = manifest["seed_splits"]["training_world_seeds_by_root"]
    rows = []
    for root_seed in root_seeds:
        seed_list = split[str(root_seed)]
        if episodes_per_root > len(seed_list):
            raise ValueError(f"requested episodes exceed frozen training seeds for root {root_seed}")
        rows.extend((root_seed, index, int(seed_list[index])) for index in range(episodes_per_root))
    return rows


def _world_families(world_seed: int):
    # These are controlled shape variants of the existing deterministic API.
    yield "starter", make_world(world_seed)
    yield "wide", make_world(derive_seed(world_seed, "dataset-world-wide"), width=20, height=14)


def generate_bundle(output_dir: str | Path, *, root_seeds: list[int] | None = None,
                    episodes_per_root: int = 1, data_version: str = "v1",
                    dataset_id: str = "pasture-studies/pasture-training-data",
                    title: str = "Pasture Studies synthetic training data",
                    license_name: str = "CC0-1.0",
                    preregistration_path: str | Path = OUTPUT_PATH) -> dict[str, Any]:
    """Generate three local JSONL data families against frozen training IDs."""
    if isinstance(episodes_per_root, bool) or episodes_per_root <= 0:
        raise ValueError("episodes_per_root must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset_id):
        raise ValueError("dataset id must be an owner/slug pair")
    for label, value in (("data version", data_version), ("title", title), ("license", license_name)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
    manifest = json.loads(Path(preregistration_path).read_text())
    verify_manifest(manifest, verify_current_sources=False)
    allowed_roots = [int(seed) for seed in manifest["seed_splits"]["training_root_seeds"]]
    selected_roots = sorted(set(root_seeds if root_seeds is not None else allowed_roots))
    if not selected_roots or not set(selected_roots).issubset(allowed_roots):
        raise ValueError("dataset generation accepts only frozen preregistered training root seeds")
    world_rows = _training_world_ids(manifest, selected_roots, episodes_per_root)
    heldout = _heldout_seeds(manifest)
    world_ids = {world_seed for _, _, world_seed in world_rows}
    if world_ids & heldout:
        raise ValueError("training world seeds overlap a held-out preregistration split")

    procedural: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    dagger: list[dict[str, Any]] = []
    teacher = HeuristicBaseline()
    for root_seed, episode_index, world_seed in world_rows:
        for scenario, world in _world_families(world_seed):
            common = {"schema_version": DATASET_SCHEMA_VERSION, "split": "training",
                      "scenario_family": scenario, "training_root_seed": root_seed,
                      "episode_index": episode_index, "world_seed": int(world.seed)}
            procedural.append({**common, "artifact_type": "procedural_world",
                               "world": serialize_world(world)})
            occupied_all = {animal.position for animal in world.animals}
            for animal in world.animals:
                for target_type, patches in (("forage", world.forage), ("water", world.water)):
                    for patch in patches:
                        path = oracle_path(world, animal.position, patch.position,
                                           occupied=occupied_all - {animal.position})
                        oracle.append({**common, "artifact_type": "navigation_oracle_route",
                            "oracle_type": "deterministic_bfs_shortest_path_v1",
                            "animal_id": animal.animal_id, "target_type": target_type,
                            "start": [animal.position.x, animal.position.y],
                            "target": [patch.position.x, patch.position.y],
                            "target_available": bool(patch.available),
                            "path": [[p.x, p.y] for p in path] if path is not None else None,
                            "reachable": path is not None})

            # One aggregation pass: a deterministic initialized policy visits
            # states; the existing heuristic supplies labels at those states.
            learner_seed = derive_seed(world_seed, "dagger-rollout-learner")
            learner = CategoricalPolicy.initialize(learner_seed)
            simulation = Simulation(world, learner)
            for _tick in range(TICKS_PER_EPISODE):
                for animal_id, features in simulation.decision_features():
                    feature_row = list(features)
                    teacher_decision = teacher.decide(feature_row)
                    learner_decision = learner.decide(feature_row)
                    dagger.append({**common, "artifact_type": "dagger_trajectory",
                        "aggregation_iteration": 1, "teacher_policy": teacher.policy_id,
                        "rollout_policy": learner.policy_id, "tick": simulation.world.tick,
                        "animal_id": animal_id, "feature_schema": list(FEATURE_NAMES),
                        "features": feature_row, "teacher_action": teacher_decision.action.value,
                        "learner_action": learner_decision.action.value,
                        "action_labels": [action.value for action in Action]})
                simulation.step()

    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or output.is_symlink() or any(output.iterdir()):
            raise ValueError("output directory must be absent or an empty non-symlink directory")
    else:
        output.mkdir(parents=True)
    rows_by_file = dict(zip(DATA_FILES, (procedural, oracle, dagger)))
    for name, rows in rows_by_file.items():
        (output / name).write_bytes(_line_bytes(rows))
    kaggle_metadata = {"id": dataset_id, "title": title.strip(),
        "licenses": [{"name": license_name.strip()}],
        "description": "Synthetic simulator worlds, navigation routes, and one-pass heuristic-teacher DAgger-style trajectories. Not biological evidence.",
        "keywords": ["synthetic", "simulation", "pasture-studies"],
        "resources": [{"path": name, "description": name.removesuffix(".jsonl").replace("-", " ")}
                      for name in DATA_FILES]}
    (output / "dataset-metadata.json").write_text(json.dumps(kaggle_metadata, indent=2, sort_keys=True) + "\n")
    source_hashes = {name: sha_file(Path(__file__).resolve().parents[1] / name) for name in SOURCE_FILES}
    body = {"schema_version": DATASET_SCHEMA_VERSION, "dataset_id": dataset_id,
        "data_version": data_version, "split": "training-only", "training_root_seeds": selected_roots,
        "world_ids": sorted(world_ids), "heldout_world_ids_sha256": canonical_sha(sorted(heldout)),
        "preregistration_manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": source_hashes,
        "family_counts": {name: len(rows) for name, rows in rows_by_file.items()},
        "limitations": ["BFS paths are navigation oracles only, not reward-optimal behavior experts",
                        "DAgger-style trajectories are one-pass heuristic-teacher labels, not iterative DAgger",
                        "all worlds and rewards are synthetic and do not validate livestock or husbandry"],
        "files": {name: {"sha256": sha_file(output / name), "size_bytes": (output / name).stat().st_size}
                  for name in (*DATA_FILES, "dataset-metadata.json")}}
    body["manifest_sha256"] = canonical_sha(body)
    (output / "dataset-manifest.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    verify_bundle(output, verify_sources=True)
    return body


def verify_bundle(directory: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset bundle must be a real directory")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha(body) != manifest.get("manifest_sha256"):
        raise ValueError("dataset manifest hash mismatch")
    expected_files = {*DATA_FILES, "dataset-metadata.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("dataset manifest has an invalid file inventory")
    if {path.name for path in root.iterdir()} != expected_files | {"dataset-manifest.json"}:
        raise ValueError("dataset bundle contains files outside its manifest")
    for name, record in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record.get("size_bytes") or sha_file(path) != record.get("sha256"):
            raise ValueError(f"dataset artifact integrity mismatch: {name}")
    if verify_sources:
        base = Path(__file__).resolve().parents[1]
        actual = {name: sha_file(base / name) for name in manifest["source_sha256"]}
        if actual != manifest["source_sha256"]:
            raise ValueError("dataset generator source provenance has changed")
    prereg_path = root / "preregistration.json"
    # Preregistration is bound by hash rather than bundled, avoiding copying
    # locked seed values into a public training-only dataset.
    if "preregistration_manifest_sha256" not in manifest:
        raise ValueError("dataset bundle lacks preregistration provenance")
    heldout_hash = manifest.get("heldout_world_ids_sha256")
    if not isinstance(heldout_hash, str) or len(heldout_hash) != 64:
        raise ValueError("dataset bundle has an invalid held-out audit hash")
    for name in DATA_FILES:
        with (root / name).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "training" or int(row.get("world_seed", -1)) not in set(manifest["world_ids"]):
                    raise ValueError("dataset contains a non-training or unregistered world row")
    metadata = json.loads((root / "dataset-metadata.json").read_text())
    if metadata.get("id") != manifest.get("dataset_id") or len(metadata.get("licenses", [])) != 1:
        raise ValueError("dataset metadata is missing its id or explicit single license")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--episodes-per-root", type=int, default=1)
    build.add_argument("--data-version", default="v1")
    build.add_argument("--dataset-id", default="pasture-studies/pasture-training-data")
    build.add_argument("--title", default="Pasture Studies synthetic training data")
    build.add_argument("--license", dest="license_name", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (generate_bundle(args.output, episodes_per_root=args.episodes_per_root,
        data_version=args.data_version, dataset_id=args.dataset_id, title=args.title,
        license_name=args.license_name) if args.command == "build" else verify_bundle(args.bundle))
    print(json.dumps({"dataset_id": result["dataset_id"], "data_version": result["data_version"],
                      "manifest_sha256": result["manifest_sha256"], "family_counts": result["family_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
