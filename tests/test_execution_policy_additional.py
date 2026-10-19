from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from farm_sim import execution_policy


ROOT = Path(execution_policy.__file__).resolve().parents[1]


class AdditionalExecutionEntrypointTests(unittest.TestCase):
    def assert_local_full_job_denied(self, command: list[str], output: Path) -> None:
        env = {key: value for key, value in os.environ.items()
               if key != "KAGGLE_KERNEL_RUN_TYPE"}
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Kaggle-only", result.stderr)
        self.assertFalse(output.exists())

    def test_registered_extensive_evaluation_is_kaggle_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            self.assert_local_full_job_denied(
                [sys.executable, "-m", "validation.evaluate_torch_mvp1",
                 "--generation", str(Path(directory) / "missing-generation"),
                 "--algorithm", "a2c", "--seed", "11",
                 "--seed-manifest", str(ROOT / "validation/kaggle-mvp1/preregistration.json"),
                 "--output", str(output)],
                output,
            )



if __name__ == "__main__":
    unittest.main()
