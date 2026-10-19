"""Policy contracts and empirical fixed-task learning thresholds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from farm_sim.model import Action, FEATURE_NAMES
from farm_sim.policy import CategoricalPolicy, CheckpointError, load_checkpoint, save_checkpoint
from farm_sim.training import (
    ALGORITHM_DESCRIPTION, TRAINING_SEEDS, counterfactual_report, task_cases,
    train_policy, train_suite,
)


class PolicyTests(unittest.TestCase):
    def test_distribution_decision_and_seeded_sampling(self):
        policy = CategoricalPolicy.initialize(5)
        features = np.zeros(len(FEATURE_NAMES))
        probabilities = policy.probabilities(features)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all(probabilities >= 0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertEqual(policy.decide(features).action, tuple(Action)[int(np.argmax(probabilities))])
        self.assertEqual(policy.sample(features, np.random.default_rng(99)),
                         policy.sample(features, np.random.default_rng(99)))
        with self.assertRaisesRegex(ValueError, "expected 20"):
            policy.decide([0.0])

    def test_checkpoint_roundtrip_and_strict_schemas(self):
        policy = CategoricalPolicy.initialize(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(policy, path, metadata={"purpose": "test"})
            loaded = load_checkpoint(path)
            np.testing.assert_array_equal(policy.parameter_vector(), loaded.parameter_vector())
            sample = np.linspace(0.0, 1.0, len(FEATURE_NAMES))
            self.assertEqual(policy.decide(sample), loaded.decide(sample))
            metadata_path = path.with_suffix(".json")
            document = json.loads(metadata_path.read_text())
            document["feature_names"][0] = "wrong"
            metadata_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(CheckpointError, "feature schema"):
                load_checkpoint(path)
            document["feature_names"] = list(FEATURE_NAMES)
            document["actions"][0] = "wrong"
            metadata_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(CheckpointError, "action schema"):
                load_checkpoint(path)
        with self.assertRaisesRegex(CheckpointError, "missing"):
            load_checkpoint("/definitely/missing/policy.npz")

    def test_memory_task_pairs_change_only_the_relevant_valid_memory_bit(self):
        cases = task_cases()
        self.assertEqual([case[1] for case in cases], [tuple(Action).index(a) for a in
                         (Action.DRINK, Action.WALK, Action.GRAZE, Action.WALK)])
        for i, j, bit in ((0, 1, "mem_water_success"), (2, 3, "mem_forage_success")):
            changed = np.flatnonzero(cases[i][0] != cases[j][0])
            self.assertEqual(changed.tolist(), [FEATURE_NAMES.index(bit)])
            self.assertEqual(cases[i][0][changed[0]], 1.0)
            self.assertEqual(cases[j][0][changed[0]], 0.0)

    def test_reinforce_updates_finite_parameters(self):
        trained, initial, _ = train_policy(11, episodes=40)
        self.assertGreater(np.linalg.norm(trained.parameter_vector() - initial.parameter_vector()), 0.0)
        self.assertTrue(np.isfinite(trained.parameter_vector()).all())

    def test_five_seed_empirical_learning_and_counterfactual_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            # This is a test fixture for the fixed training protocol; production
            # CLI execution is separately required to run in Kaggle.
            with patch("farm_sim.training.require_kaggle_for_full_workload"):
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
                self.assertTrue(Path(run["checkpoint"]).is_file())
                self.assertTrue(Path(run["initial_checkpoint"]).is_file())
                self.assertTrue(run["counterfactuals"]["water_memory"]["passes"])
                self.assertTrue(run["counterfactuals"]["forage_memory"]["passes"])
                loaded = load_checkpoint(run["checkpoint"])
                self.assertEqual(loaded.episodes, 2_000)
            self.assertTrue(Path(directory, "training-report.json").is_file())
            self.assertTrue(Path(directory, "default-policy.npz").is_file())


if __name__ == "__main__":
    unittest.main()
