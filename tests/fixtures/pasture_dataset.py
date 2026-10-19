from __future__ import annotations

import json
from pathlib import Path

from validation.pasture_studies_dataset import (
    DATA_FILES,
    SOURCE_FILES,
    _heldout_seeds,
    _training_world_ids,
    canonical_sha,
    sha_file,
)
from validation.preregister_torch_mvp1 import OUTPUT_PATH


def write_frozen_dataset_bundle(directory: Path) -> Path:
    """Write a tiny checksummed fixture without simulating worlds or fitting DAgger."""
    directory.mkdir(parents=True)
    preregistration = json.loads(OUTPUT_PATH.read_text())
    root_seed = int(preregistration["seed_splits"]["training_root_seeds"][0])
    _, _, world_seed = _training_world_ids(preregistration, [root_seed], 1)[0]
    rows = {
        DATA_FILES[0]: {"split": "training", "artifact_type": "procedural_world", "world_seed": world_seed},
        DATA_FILES[1]: {"split": "training", "artifact_type": "navigation_oracle_route", "world_seed": world_seed},
        DATA_FILES[2]: {"split": "training", "artifact_type": "dagger_trajectory", "world_seed": world_seed},
    }
    for name, row in rows.items():
        (directory / name).write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    dataset_id = "pasture-studies/test-fixture"
    (directory / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset_id, "licenses": [{"name": "CC0-1.0"}],
    }, sort_keys=True) + "\n")
    all_files = (*DATA_FILES, "dataset-metadata.json")
    body = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "data_version": "fixture-v1",
        "split": "training-only",
        "training_root_seeds": [root_seed],
        "training_world_ids": [world_seed],
        "world_ids": [world_seed],
        "heldout_world_ids_sha256": canonical_sha(sorted(_heldout_seeds(preregistration))),
        "preregistration_manifest_sha256": preregistration["manifest_sha256"],
        "source_sha256": {name: sha_file(Path(__file__).resolve().parents[2] / name)
                          for name in SOURCE_FILES},
        "family_counts": {name: 1 for name in DATA_FILES},
        "dagger_aggregation_rounds": 2,
        "dagger_supervised_epochs_per_round": 4,
        "limitations": ["Frozen fixture only; it is not a generated training dataset."],
        "files": {name: {"sha256": sha_file(directory / name),
                         "size_bytes": (directory / name).stat().st_size}
                  for name in all_files},
    }
    body["manifest_sha256"] = canonical_sha(body)
    (directory / "dataset-manifest.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return directory
