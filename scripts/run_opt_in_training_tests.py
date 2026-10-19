"""Explicitly run the training-test category only inside an approved Kaggle runtime."""
from __future__ import annotations

import argparse
import subprocess
import sys

from validation.opt_in_training_tests import require_opt_in_training_tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", choices=("unittest", "pytest"))
    args, extra = parser.parse_known_args()
    require_opt_in_training_tests()
    command = ([sys.executable, "-m", "unittest", "discover", "-s", "tests/opt_in_training", "-v"]
               if args.runner == "unittest" else
               [sys.executable, "-m", "pytest", "-q", "tests/opt_in_training"])
    return subprocess.run([*command, *extra], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
