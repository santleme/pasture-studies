#!/usr/bin/env python3
"""Plan (but do not execute) the outcome-masked precision pilot.

This draft verifies the preregistered schedule but provides no production
execution or artifact-writing function. Synthetic tests exercise only a pure
sample-SD helper over fabricated values. The historical v1 finalizer is not
safe for real outputs because it does not verify an execution sidecar.

``--plan-only`` is an outcome-free preflight. This version intentionally has
no production pilot execution API: a separately reviewed protocol, signed
authorization verifier, and governed trusted root must be implemented before
pilot outcomes may be collected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.preregister_torch_mvp1 import (  # noqa: E402
    FAMILIES,
    canonical_sha,
    verify_manifest,
)

PILOT_SOURCES = (
    "validation/run_torch_precision_pilot.py",
    "validation/finalize_precision_pilot_v2.py",
    "validation/evaluate_torch_mvp1.py",
    "farm_sim/torch_training.py",
    "farm_sim/simulation.py",
    "farm_sim/model.py",
    "farm_sim/policy.py",
    "farm_sim/navigation.py",
    "farm_sim/perception.py",
    "farm_sim/memory.py",
    "farm_sim/physiology.py",
)
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pending_manifest(path: str | Path, expected_file_sha256: str) -> dict[str, Any]:
    """Load a pending manifest only when both file and internal hashes verify."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pending preregistration manifest not found: {manifest_path}")
    if len(expected_file_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_file_sha256):
        raise ValueError("--manifest-sha256 must be a lowercase 64-character SHA-256")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("pending preregistration could not be read") from exc
    actual_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError("pending preregistration file SHA-256 does not match --manifest-sha256")
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("pending preregistration is not valid JSON") from exc
    if manifest.get("status") != "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0":
        raise ValueError("precision pilot requires the pending pre-outcome preregistration")
    verify_manifest(manifest, verify_current_sources=True)
    return manifest


def build_plan(manifest: dict[str, Any], *, manifest_file_sha256: str) -> dict[str, Any]:
    """Return an outcome-free execution plan derived only from frozen inputs."""
    if manifest.get("status") != "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0":
        raise ValueError("precision pilot requires a pending pre-outcome manifest")
    pins = manifest.get("source_sha256")
    if not isinstance(pins, dict) or not set(PILOT_SOURCES).issubset(pins):
        raise ValueError("precision pilot is not bound by this preregistration; create a separately versioned, reviewed source-bound candidate")
    actual = _source_hashes()
    drift = {name for name in PILOT_SOURCES if pins.get(name) != actual.get(name)}
    if drift:
        raise ValueError(f"precision-pilot source hashes differ from the preregistration candidate: {sorted(drift)}")
    training_seeds = manifest.get("seed_splits", {}).get("training_root_seeds")
    pilot = manifest.get("seed_splits", {}).get("precision_pilot")
    if (not isinstance(training_seeds, list) or not training_seeds or
            any(isinstance(x, bool) or not isinstance(x, int) for x in training_seeds)):
        raise ValueError("manifest must provide registered integer training root seeds")
    if not isinstance(pilot, dict) or set(pilot) != set(FAMILIES):
        raise ValueError("manifest precision_pilot must contain exactly the registered scenario families")
    for family in FAMILIES:
        seeds = pilot[family]
        if (not isinstance(seeds, list) or not seeds or
                any(isinstance(x, bool) or not isinstance(x, int) for x in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError(f"precision-pilot world seeds are invalid for {family}")
    expected_seed_hash = canonical_sha({family: pilot[family] for family in FAMILIES})
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "pilot_seed_ids_sha256": expected_seed_hash,
        "family_order": list(FAMILIES),
        "training_root_seeds": list(training_seeds),
        "precision_pilot_world_seeds": {family: list(pilot[family]) for family in FAMILIES},
        "untrained_actor_seed_derivation": "ActorCritic(value_head=False, seed=registered training root seed)",
        "evaluation_schedule": "family-major, then registered pilot world seed, then registered training root seed",
        "episodes_per_family": len(training_seeds) * len(pilot[FAMILIES[0]]),
        "sd_convention": "sample standard deviation (ddof=1) across registered training-root × pilot-world cells",
        "masked_statistic": "SD of untrained synthetic total returns only; no means, signs, individual cells, trained arms, or differences",
    }


def _source_hashes() -> dict[str, str]:
    missing = [name for name in PILOT_SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"precision pilot source files are missing: {missing}")
    return {name: file_sha256(ROOT / name) for name in PILOT_SOURCES}


def _synthetic_sd_summary_for_test(totals_by_family: dict[str, list[float]]) -> dict[str, Any]:
    """Pure calculation over supplied synthetic numbers; cannot run worlds or write pilot artifacts."""
    if set(totals_by_family) != set(FAMILIES):
        raise ValueError("synthetic test input must contain exactly the registered family labels")
    sd_by_family = {}
    for family, values in totals_by_family.items():
        if len(values) < 2:
            raise ValueError(f"synthetic test input has fewer than two values for {family}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
               for value in values):
            raise ValueError(f"synthetic test input contains a non-finite value for {family}")
        sd = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"synthetic test SD must be finite and nonzero for {family}")
        sd_by_family[family] = sd
    return {"synthetic_test_only": True, "sample_sd_by_family": sd_by_family}


def run_precision_pilot(*args, **kwargs):
    """Fail-closed production boundary; no trust root/approval is configured."""
    raise ValueError("precision pilot execution is unavailable: independent protocol approval and a signed, trusted authorization workflow are not configured")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="exact pending preregistration JSON")
    parser.add_argument("--manifest-sha256", required=True, help="SHA-256 of exact manifest file bytes")
    parser.add_argument("--output", type=Path, required=True, help="masked five-field pilot artifact output")
    parser.add_argument("--run-record", type=Path, help="optional non-outcome provenance sidecar path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="verify and print frozen schedule only; no episodes run")
    mode.add_argument("--execute", action="store_true", help="explicitly authorize execution of precision-pilot episodes")
    args = parser.parse_args(argv)
    try:
        manifest = load_pending_manifest(args.manifest, args.manifest_sha256)
        plan = build_plan(manifest, manifest_file_sha256=args.manifest_sha256)
        if args.plan_only:
            print(json.dumps({"status": "PLAN_ONLY_NO_OUTCOMES", **plan}, sort_keys=True, indent=2))
            return 0
        raise ValueError("pilot execution is not authorized: the versioned protocol candidate is pending independent review and signed authorization")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
