"""Fail-closed execution-location policy for resource-intensive Pasture runs."""
from __future__ import annotations

import os
from pathlib import Path


KAGGLE_WORKING_DIR = Path("/kaggle/working")


def is_kaggle_runtime() -> bool:
    """Require both Kaggle's mounted working directory and its runtime marker.

    An environment variable by itself is caller-controlled and must not authorize
    a full workload on a local machine.
    """
    return KAGGLE_WORKING_DIR.is_dir() and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def require_kaggle_for_full_workload(workload: str, *, smoke: bool = False) -> None:
    """Permit explicitly bounded local smoke; require Kaggle for full workloads."""
    if smoke:
        return
    if not is_kaggle_runtime():
        raise RuntimeError(
            f"{workload} full workload is Kaggle-only; use an explicit bounded --smoke "
            "run for local development"
        )
