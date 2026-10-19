from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from farm_sim.torch_training import (
    InvalidPolicyOutput,
    derive_seed,
    shard_episode_indices,
    validate_equal_all_one_masks,
    validate_probabilities,
)


class TorchTrainingContractTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_order_sensitive(self):
        self.assertEqual(derive_seed(10, "world", 3), derive_seed(10, "world", 3))
        self.assertNotEqual(derive_seed(10, "world", 3), derive_seed(10, "world", 4))
        self.assertNotEqual(derive_seed(10, "a", "b"), derive_seed(10, "b", "a"))

    def test_shards_masks_and_policy_probabilities_obey_contracts(self):
        shards = [shard_episode_indices(0, 16, 2, rank) for rank in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), list(range(16)))
        self.assertEqual(validate_equal_all_one_masks([np.ones((72, 5))]), 360)
        probabilities = validate_probabilities([.1, .2, .3, .1, .2, .1])
        self.assertEqual(probabilities.shape, (6,))
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            shard_episode_indices(0, 5, 2, 0)
        with self.assertRaises(InvalidPolicyOutput):
            validate_probabilities([np.nan] * 6)

    def test_training_cli_help_and_denial_are_nontraining_paths(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "validation.train_torch_mvp1", "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--smoke-update", help_result.stdout)
        with subprocess.Popen(
            [sys.executable, "-m", "validation.train_torch_mvp1", "--mode", "run",
             "--episodes", "32", "--output", "/tmp/must-not-train"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as process:
            _, stderr = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Kaggle-only", stderr)


if __name__ == "__main__":
    unittest.main()
