from __future__ import annotations

from validation.opt_in_training_tests import require_opt_in_training_tests

require_opt_in_training_tests()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validation.pasture_studies_dataset import generate_bundle, verify_bundle


class PastureStudiesDatasetTests(unittest.TestCase):
    def test_generates_reproducible_training_only_world_oracle_and_dagger_families(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            one = generate_bundle(first, root_seeds=[71001], episodes_per_root=1, license_name="CC0-1.0")
            two = generate_bundle(second, root_seeds=[71001], episodes_per_root=1, license_name="CC0-1.0")
            self.assertEqual(one["manifest_sha256"], two["manifest_sha256"])
            self.assertEqual(one["family_counts"], {
                "procedural-worlds.jsonl": 2,
                "oracle-trajectories.jsonl": 40,
                "dagger-trajectories.jsonl": 1440,
            })
            self.assertEqual(one["dagger_aggregation_rounds"], 2)
            dagger_rows = [json.loads(line) for line in
                           (first / "dagger-trajectories.jsonl").read_text().splitlines()]
            iterations = {row["aggregation_iteration"] for row in dagger_rows}
            self.assertEqual(iterations, {1, 2})
            for iteration in iterations:
                self.assertEqual({row["tick"] for row in dagger_rows
                                  if row["aggregation_iteration"] == iteration}, set(range(72)))
            self.assertEqual(verify_bundle(first)["split"], "training-only")
            self.assertEqual(verify_bundle(first)["world_ids"], one["world_ids"])

    def test_rejects_nontraining_root_and_tampered_family_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training root seeds"):
                generate_bundle(Path(directory) / "bad", root_seeds=[72001], license_name="CC0-1.0")
            bundle = Path(directory) / "bundle"
            generate_bundle(bundle, root_seeds=[71001], episodes_per_root=1, license_name="CC0-1.0")
            (bundle / "dagger-trajectories.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
