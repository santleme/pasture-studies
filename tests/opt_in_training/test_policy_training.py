from __future__ import annotations

from validation.opt_in_training_tests import require_opt_in_training_tests

require_opt_in_training_tests()

import tempfile
import unittest

import numpy as np

from farm_sim.execution_policy import is_kaggle_runtime
from farm_sim.policy import load_checkpoint
from farm_sim.training import (
    ALGORITHM_DESCRIPTION,
    TRAINING_SEEDS,
    train_policy,
    train_suite,
)


class PolicyTrainingTests(unittest.TestCase):
    def test_reinforce_updates_finite_parameters(self):
        trained, initial, _ = train_policy(11, episodes=40)
        self.assertGreater(np.linalg.norm(trained.parameter_vector() - initial.parameter_vector()), 0.0)
        self.assertTrue(np.isfinite(trained.parameter_vector()).all())

    @unittest.skipUnless(is_kaggle_runtime(), "full five-seed policy training is Kaggle-only")
    def test_five_seed_empirical_learning_and_counterfactual_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            report = train_suite(directory)
            self.assertEqual(report["algorithm"], ALGORITHM_DESCRIPTION)
            aggregate = report["aggregate"]
            self.assertEqual(report["training_seeds"], list(TRAINING_SEEDS))
            self.assertEqual(report["episodes_per_run"], 2_000)
            self.assertGreaterEqual(aggregate["trained_accuracy"], 0.85, report)
            self.assertGreaterEqual(aggregate["improvement_pp"], 25.0, report)
            self.assertTrue(aggregate["all_counterfactuals_pass"], report)
            self.assertEqual(len(report["runs"]), 5)
            for run in report["runs"]:
                self.assertEqual(run["evaluation"]["cases"], 20)
                self.assertGreater(run["parameter_delta_l2"], 0)
                self.assertTrue(load_checkpoint(run["checkpoint"]).episodes == 2_000)
                self.assertTrue(run["counterfactuals"]["water_memory"]["passes"])
                self.assertTrue(run["counterfactuals"]["forage_memory"]["passes"])
            self.assertTrue(__import__("pathlib").Path(directory, "training-report.json").is_file())
            self.assertTrue(__import__("pathlib").Path(directory, "default-policy.npz").is_file())
