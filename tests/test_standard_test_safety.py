from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_standard_test_safety import inspect_standard_tests
from validation.opt_in_training_tests import require_opt_in_training_tests


class StandardTestSafetyTests(unittest.TestCase):
    def test_gate_detects_new_training_calls_before_runner_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_unsafe.py").write_text(
                "def test_new_training():\n"
                "    model.learn(total_timesteps=8)\n"
                "    policy.fit(rows, labels)\n"
                "    optimizer.step()\n"
            )
            findings = inspect_standard_tests(root)
        self.assertEqual(len(findings), 3)
        self.assertTrue(any("learn()" in finding for finding in findings))
        self.assertTrue(any("fit()" in finding for finding in findings))
        self.assertTrue(any("optimizer.step()" in finding for finding in findings))

    def test_gate_checks_module_scope_before_test_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_import_side_effect.py").write_text(
                "from validation.runner import train_algorithm\n"
                "model = PPO('MlpPolicy', env)\n"
                "model.learn(total_timesteps=1)\n"
            )
            findings = inspect_standard_tests(root)
        self.assertEqual(len(findings), 1)
        self.assertIn("learn()", findings[0])

    def test_gate_excludes_only_explicit_opt_in_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opt_in = root / "opt_in_training"
            opt_in.mkdir()
            (opt_in / "test_training.py").write_text("def test_opt_in(): model.learn()\n")
            (root / "test_contract.py").write_text("def test_step_contract(): env.step(0)\n")
            self.assertEqual(inspect_standard_tests(root), [])

    def test_opt_in_training_suite_denies_unflagged_local_execution(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PASTURE_RUN_OPT_IN_TRAINING_TESTS", None)
            with self.assertRaisesRegex(RuntimeError, "training tests are opt-in"):
                require_opt_in_training_tests()


if __name__ == "__main__":
    unittest.main()
