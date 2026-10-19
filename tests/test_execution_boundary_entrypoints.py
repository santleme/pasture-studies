from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExecutionBoundaryEntrypointTests(unittest.TestCase):
    def test_extensive_evaluation_and_research_runners_fail_closed_locally(self):
        cases = [
            ["-m", "validation.evaluate_torch_mvp1", "--generation", "missing",
             "--algorithm", "a2c", "--seed", "11", "--output"],
            ["validation/evaluate_policies.py"],
            ["-m", "validation.run_dagger_generalization", "--output"],
            ["-m", "validation.autoresearch_pasture_rl", "--output"],
            ["-m", "validation.sb3_pilot"],
            ["-m", "validation.sb3_single_agent_followup"],
            ["-m", "validation.train_torch_mvp1", "--mode", "eval", "--episodes", "1",
             "--resume", "missing", "--output"],
        ]
        optional_modules = ("torch", "gymnasium", "stable_baselines3")
        if not all(__import__("importlib.util", fromlist=["find_spec"]).find_spec(name)
                   for name in optional_modules):
            cases = cases[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.json"
            env = {key: value for key, value in os.environ.items()
                   if key != "KAGGLE_KERNEL_RUN_TYPE"}
            for raw in cases:
                command = [sys.executable, *raw]
                if raw[-1] == "--output":
                    command.append(str(output))
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                            text=True, env=env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Kaggle-only", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
