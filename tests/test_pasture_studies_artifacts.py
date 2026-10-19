from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from validation.benchmark_policy_report import summarize_evaluation
from validation.package_torch_corpus import package_corpus, verify_package


def evaluation_fixture():
    families = ("starter", "resource_scarce_relocated", "wide", "unseen_corridor")
    runs = []
    for family_index, family in enumerate(families):
        seed = 8001 + family_index
        for policy, reward in (("trained", 10.0 + family_index),
                               ("untrained_same_seed", 4.0),
                               ("heuristic_baseline", 7.0)):
            runs.append({"policy": policy, "scenario": family, "world_seed": seed,
                         "total_reward": reward, "invalid_actions": 0,
                         "action_counts": {"drink": 2, "rest": 3},
                         "reward_components": {name: {"count": count} for name, count in
                             (("forage_success", 2), ("forage_depleted", 1),
                              ("water_success", 1), ("water_blocked", 0))}})
    return {"schema_version": 1, "paired_world_seeds": True,
            "seed_split": {"evaluation_split": "validation", "manifest_sha256": "a" * 64,
                           "validation_world_seeds": {family: [8001 + index]
                                                      for index, family in enumerate(families)},
                           "families": list(families)},
            "training_validation_test_disjoint_check": {
                "pass": True, "training_vs_eval_collision_count": 0,
                "all_registered_splits_pairwise_disjoint": True},
            "training_seed": 71001, "runs": runs}


class PastureStudiesArtifactsTests(unittest.TestCase):
    def test_policy_summary_preserves_pairing_baselines_and_family_generalization(self):
        result = summarize_evaluation(evaluation_fixture())
        self.assertEqual(result["heldout_world_count"], 4)
        self.assertEqual(result["scenario_families"]["wide"]["world_count"], 1)
        self.assertEqual(result["paired_total_reward_delta_vs_heuristic"]["starter"]["trained"]["mean"], 3.0)
        self.assertEqual(result["generalization"]["trained"]["reward_gap_from_reference_by_family"]["wide"], 2.0)
        self.assertAlmostEqual(sum(result["scenario_families"]["starter"]["policies"]["trained"]["action_frequencies"].values()), 1.0)

    def test_policy_summary_rejects_unregistered_or_unpaired_evaluation(self):
        report = evaluation_fixture()
        report["seed_split"]["evaluation_split"] = "locked_test"
        with self.assertRaisesRegex(ValueError, "validation split"):
            summarize_evaluation(report)
        report = evaluation_fixture()
        report["runs"].pop()
        with self.assertRaisesRegex(ValueError, "all three paired policies"):
            summarize_evaluation(report)

    def test_policy_summary_requires_exact_frozen_validation_seed_map(self):
        report = evaluation_fixture()
        for row in report["runs"][:3]:
            row["world_seed"] += 1000
        with self.assertRaisesRegex(ValueError, "frozen validation seed map"):
            summarize_evaluation(report)

        report = evaluation_fixture()
        report["training_validation_test_disjoint_check"]["training_vs_eval_collision_count"] = 1
        with self.assertRaisesRegex(ValueError, "disjoint held-out split"):
            summarize_evaluation(report)

    def test_dataset_package_records_and_verifies_local_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corpus.npz"
            source.write_bytes(b"test archive")
            source.with_suffix(".json").write_text("{}\n")
            arrays = {"world_ids": np.arange(20).reshape(10, 2)}
            config = {"smoke_mode": False,
                      "split": "training-only fixed workload; never held-out; stale off-policy compute replay",
                      "partition_count": 10, "episodes_per_partition": 2,
                      "corpus_sha256": "f" * 64}
            output = root / "release"
            with patch("validation.package_torch_corpus.load_corpus", return_value=(arrays, config)):
                manifest = package_corpus(source, output, dataset_id="owner/pasture-data",
                    title="Pasture training corpus", license_name="CC0-1.0", data_version="v1")
            self.assertEqual(manifest["split"], "training-only")
            self.assertEqual(verify_package(output)["manifest_sha256"], manifest["manifest_sha256"])
            metadata = json.loads((output / "dataset-metadata.json").read_text())
            self.assertEqual(metadata["licenses"], [{"name": "CC0-1.0"}])
            (output / "pasture-studies-v1.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                verify_package(output)


if __name__ == "__main__":
    unittest.main()
