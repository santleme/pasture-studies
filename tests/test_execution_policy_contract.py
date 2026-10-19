from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from farm_sim import execution_policy


ROOT = Path(execution_policy.__file__).resolve().parents[1]


class ExecutionPolicyContractTests(unittest.TestCase):
    def test_numpy_policy_training_is_denied_locally_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            env = {key: value for key, value in os.environ.items()
                   if key != "KAGGLE_KERNEL_RUN_TYPE"}
            result = subprocess.run(
                [sys.executable, "-m", "farm_sim.training", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Kaggle-only", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
