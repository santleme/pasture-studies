"""Run only the default, statically screened test suites."""
from __future__ import annotations

import argparse
import subprocess
import sys

from scripts.check_standard_test_safety import main as safety_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", choices=("unittest", "pytest"))
    args, extra = parser.parse_known_args()
    if safety_gate() != 0:
        return 1
    command = ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
               if args.runner == "unittest" else [sys.executable, "-m", "pytest", "-q", "tests"])
    return subprocess.run([*command, *extra], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
