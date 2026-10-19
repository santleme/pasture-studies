"""Fail-closed gate shared by explicitly requested training-test suites."""
from __future__ import annotations

import os

from farm_sim.execution_policy import is_kaggle_runtime


def require_opt_in_training_tests() -> None:
    if os.environ.get("PASTURE_RUN_OPT_IN_TRAINING_TESTS") != "1":
        raise RuntimeError(
            "training tests are opt-in; set PASTURE_RUN_OPT_IN_TRAINING_TESTS=1 "
            "and run inside Kaggle"
        )
    if not is_kaggle_runtime():
        raise RuntimeError("training tests are Kaggle-only")
