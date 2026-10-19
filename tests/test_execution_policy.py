from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from farm_sim import execution_policy


ROOT = Path(__file__).resolve().parents[1]


class ExecutionPolicyTests(unittest.TestCase):
    def test_runtime_marker_alone_does_not_authorize_full_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(execution_policy, "KAGGLE_WORKING_DIR", Path(directory)), \
                    patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
                self.assertTrue(execution_policy.is_kaggle_runtime())
            with patch.object(execution_policy, "KAGGLE_WORKING_DIR", Path(directory) / "missing"), \
                    patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
                self.assertFalse(execution_policy.is_kaggle_runtime())
                with self.assertRaisesRegex(RuntimeError, "Kaggle-only"):
                    execution_policy.require_kaggle_for_full_workload("test workload")

    def test_explicit_smoke_is_allowed_without_kaggle_runtime(self):
        with patch.object(execution_policy, "KAGGLE_WORKING_DIR", Path("/missing-kaggle-runtime")), \
                patch.dict(os.environ, {}, clear=True):
            execution_policy.require_kaggle_for_full_workload("smoke", smoke=True)

    def test_full_workload_clis_deny_local_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            cases = [
                [sys.executable, "-m", "scripts.train_sb3_ppo", "--output", str(output)],
                [sys.executable, "-m", "validation.pasture_studies_dataset", "build",
                 "--output", str(output), "--license", "CC0-1.0"],
                [sys.executable, "-m", "validation.benchmark_torch_fixed_corpus", "generate-corpus",
                 "--seed", "11", "--output", str(output)],
                [sys.executable, "-m", "validation.train_torch_mvp1", "--readiness-ddp",
                 "--algorithm", "a2c", "--device", "cuda", "--output", str(output)],
            ]
            for command in cases:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                             env={key: value for key, value in os.environ.items()
                                                  if key != "KAGGLE_KERNEL_RUN_TYPE"})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Kaggle-only", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
