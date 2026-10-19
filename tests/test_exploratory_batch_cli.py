from __future__ import annotations

import unittest
from unittest.mock import patch

from validation import train_torch_mvp1 as trainer


class ExploratoryBatchCliTests(unittest.TestCase):
    def test_batch_suite_is_exactly_four_paired_variants_and_three_fixed_seeds(self):
        specs = trainer.exploratory_batch_specs()
        self.assertEqual(len(specs), 12)
        self.assertEqual({spec["variant"] for spec in specs}, {"A", "B", "C", "D"})
        self.assertEqual({spec["seed"] for spec in specs}, set(trainer.EXPLORATORY_BATCH_SEEDS))
        self.assertEqual({spec["episodes"] for spec in specs}, {512})
        expected = {
            "A": ("original", False, 0.0),
            "B": ("original", True, 0.01),
            "C": ("state_aware", False, 0.0),
            "D": ("state_aware", True, 0.01),
        }
        for spec in specs:
            self.assertEqual((spec["reward_variant"], spec["normalize_advantage"],
                              spec["entropy_coefficient"]), expected[spec["variant"]])
        for variant in expected:
            self.assertEqual(sum(spec["variant"] == variant for spec in specs), 3)

    def test_batch_flag_keeps_its_fixed_episode_gate(self):
        with patch("sys.argv", ["train_torch_mvp1", "--exploratory-batch", "--algorithm", "a2c",
                                "--device", "cuda", "--episodes", "513"]):
            with self.assertRaises(SystemExit):
                trainer.main()

    def test_single_run_and_batch_flags_cannot_be_combined(self):
        with patch("sys.argv", ["train_torch_mvp1", "--exploratory-a2c", "--exploratory-batch"]):
            with self.assertRaises(SystemExit):
                trainer.main()


if __name__ == "__main__":
    unittest.main()
