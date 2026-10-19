from __future__ import annotations

import unittest
import json
from pathlib import Path
from unittest.mock import patch

from scripts import kaggle_deploy, kaggle_release
from validation import train_torch_mvp1 as trainer


class ExploratoryBatchCliTests(unittest.TestCase):
    def test_batch_suite_is_exactly_four_paired_variants_and_three_fixed_seeds(self):
        specs = trainer.exploratory_batch_specs()
        self.assertEqual(len(specs), 12)
        self.assertEqual({spec["variant"] for spec in specs}, {"A", "B", "C", "D"})
        self.assertEqual({spec["seed"] for spec in specs}, set(trainer.EXPLORATORY_BATCH_SEEDS))
        self.assertEqual({spec["episodes"] for spec in specs}, {512})
        expected = {
            "A": ("original", 0.1, False, 0.0, 1.0),
            "B": ("original", 0.1, True, 0.01, 1.0),
            "C": ("state_aware", 0.1, False, 0.0, 1.0),
            "D": ("state_aware", 0.1, True, 0.01, 1.0),
        }
        for spec in specs:
            self.assertEqual((spec["reward_variant"], spec["state_reward_scale"], spec["normalize_advantage"],
                              spec["entropy_coefficient"], spec["discount_factor"]), expected[spec["variant"]])
        for variant in expected:
            self.assertEqual(sum(spec["variant"] == variant for spec in specs), 3)

    def test_batch_flag_keeps_its_fixed_episode_gate(self):
        with patch("sys.argv", ["train_torch_mvp1", "--exploratory-batch", "--algorithm", "a2c",
                                "--device", "cuda", "--episodes", "513"]):
            with self.assertRaises(SystemExit):
                trainer.main()

    def test_release_config_binds_trainer_variants_seeds_and_total_job_cap(self):
        for config in (kaggle_deploy.EXPLORATORY_COMPARISON_CONFIG_VALUE,
                       kaggle_release.EXPLORATORY_COMPARISON_CONFIG_VALUE):
            self.assertEqual(config["variant_config"], trainer.EXPLORATORY_BATCH_VARIANTS)
            self.assertEqual(config["seeds"], list(trainer.EXPLORATORY_BATCH_SEEDS))
            self.assertEqual(config["episodes_per_variant_seed"], trainer.EXPLORATORY_BATCH_EPISODES)
            self.assertEqual(config["evaluation_episodes"], trainer.EXPLORATORY_BATCH_EVAL_EPISODES)
            self.assertEqual(config["training_jobs"], 12)
            self.assertEqual(config["max_training_jobs"], 24)
            self.assertEqual(config["comparison_eval_seed"], 9483999)
            self.assertEqual(config["comparison_episodes_per_family"], 3)

    def test_locked_data_denial_marker_is_only_added_to_variant_checkpoint_state(self):
        ordinary = {"update": 1}
        self.assertIs(trainer._mark_exploratory_checkpoint_state(
            ordinary, exploratory_variant=None), ordinary)
        self.assertNotIn("locked_outcomes_accessed", ordinary)
        variant = {"update": 1}
        self.assertIs(trainer._mark_exploratory_checkpoint_state(
            variant, exploratory_variant="A"), variant)
        self.assertIs(variant["locked_outcomes_accessed"], False)

    def test_notebook_calibrates_then_runs_one_suite_and_compares_common_worlds(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/pasture_studies_kaggle.ipynb"
        notebook = json.loads(notebook_path.read_text())
        code_cells = ["".join(cell["source"]) for cell in notebook["cells"]
                      if cell.get("cell_type") == "code"]
        checkout = next(code for code in code_cells if "sparse-checkout" in code)
        self.assertIn("/validation/evaluate_torch_mvp1.py", checkout)
        self.assertIn("/validation/compare_exploratory_checkpoints.py", checkout)
        comparison = next(code for code in code_cells if 'if MODE == "EXPLORATORY_COMPARISON":' in code)
        self.assertIn("if use_dual:", comparison)
        self.assertIn("else:", comparison)
        self.assertIn("--profile-only", comparison)
        self.assertIn("--exploratory-batch", comparison)
        self.assertEqual(comparison.count("--exploratory-batch"), 2)  # exclusive single/dual branches
        self.assertIn("--checkpoints", comparison)
        self.assertIn("--eval-seed", comparison)
        self.assertIn("9483999", comparison)
        self.assertNotIn("--seed-manifest", comparison)
        self.assertNotIn("--locked-test", comparison)
        self.assertIn("evaluation_summary_sha256", comparison)
        self.assertIn("evaluation_plot_sha256", comparison)

    def test_single_run_and_batch_flags_cannot_be_combined(self):
        with patch("sys.argv", ["train_torch_mvp1", "--exploratory-a2c", "--exploratory-batch"]):
            with self.assertRaises(SystemExit):
                trainer.main()


if __name__ == "__main__":
    unittest.main()
