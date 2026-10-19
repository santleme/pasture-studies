from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from tests.fixtures.pasture_dataset import write_frozen_dataset_bundle
from validation.pasture_studies_dataset import verify_bundle


class PastureStudiesDatasetFixtureTests(unittest.TestCase):
    def test_frozen_bundle_verifies_training_split_and_family_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_frozen_dataset_bundle(Path(directory) / "fixture")
            result = verify_bundle(bundle)
            self.assertEqual(result["split"], "training-only")
            self.assertEqual(result["family_counts"], {
                "procedural-worlds.jsonl": 1,
                "oracle-trajectories.jsonl": 1,
                "dagger-trajectories.jsonl": 1,
            })
            self.assertEqual(result["dagger_aggregation_rounds"], 2)

    def test_verifier_rejects_tampered_frozen_family_file(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_frozen_dataset_bundle(Path(directory) / "fixture")
            (bundle / "dagger-trajectories.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
