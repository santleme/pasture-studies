#!/usr/bin/env python3
"""Create and verify outcome-blind Pasture Studies experimental release manifests.

This module does not authorize training. It binds exact source, notebook,
protocol, thresholds, runtime, and frozen seed/configuration inputs. Since this
repository currently has no independently established signing trust root, all
release approvals remain pending and EXPERIMENT/FULL_VALIDATION are denied.
Do not add a trust key here without a separately reviewed governance change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path("validation/kaggle-mvp1/preregistration.json")
RUNTIME_PROTOCOL_CANDIDATE = Path("validation/protocols/pasture-experiment-runtime-v2-candidate.json")
NOTEBOOK = Path("notebooks/pasture_studies_kaggle.ipynb")
ENGINE_SOURCE_PATHS = (
    "farm_sim/torch_training.py", "farm_sim/exploratory_learning.py", "farm_sim/torch_checkpoint.py", "farm_sim/simulation.py",
    "farm_sim/model.py", "farm_sim/navigation.py", "farm_sim/perception.py",
    "farm_sim/memory.py", "farm_sim/physiology.py", "farm_sim/policy.py",
    "validation/train_torch_mvp1.py", "validation/evaluate_torch_mvp1.py",
    "validation/benchmark_torch_fixed_corpus.py", "validation/preregister_torch_mvp1.py",
    "validation/profile_torch_training.py", "validation/profile_training_pipeline.py",
    "requirements-training.txt",
)
SCHEMA_VERSION = 1
REQUIRED_RUNTIME_FIELDS = frozenset({
    "python", "numpy", "pytorch", "cuda_runtime", "cuda_available", "gpu_count", "gpu_names",
})
APPROVAL_SCHEMA_VERSION = 1
APPROVAL_STATUS_PENDING = "PENDING_INDEPENDENT_SIGNATURE"
ALLOWED_MODES = frozenset({"PREFLIGHT", "SMOKE"})
EXPLORATORY_CONFIG_VALUE = {
    "task": "exploratory_a2c_training", "algorithm": "a2c", "episodes": 256,
    "optimizer_updates": 16, "seed": 9482817, "evaluation_episodes": 5,
    "device_count": 1, "locked_test_access": False, "formal_eligible": False,
    "benchmark_eligible": False, "checkpoint_promotion": False,
}
EXPLORATORY_COMPARISON_CONFIG_VALUE = {
    "task": "exploratory_a2c_variant_comparison", "algorithm": "a2c",
    "variants": ["A", "B", "C", "D"], "episodes_per_variant_seed": 512,
    "seeds": [9483001, 9483002, 9483003], "training_jobs": 12,
    "max_training_jobs": 24, "max_devices": 2, "locked_test_access": False,
    "variant_config": {
        "A": {"reward_variant": "original", "state_reward_scale": 0.1,
              "normalize_advantage": False, "entropy_coefficient": 0.0},
        "B": {"reward_variant": "original", "state_reward_scale": 0.1,
              "normalize_advantage": True, "entropy_coefficient": 0.01},
        "C": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
              "normalize_advantage": False, "entropy_coefficient": 0.0},
        "D": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
              "normalize_advantage": True, "entropy_coefficient": 0.01},
    },
    "formal_eligible": False, "benchmark_eligible": False, "checkpoint_promotion": False,
}
READINESS_DIAGNOSTIC_TASKS = frozenset({"checkpoint_recovery", "t4x2_path_smoke"})
READINESS_CONFIG_KEYS = frozenset({
    "task", "max_updates", "max_episodes", "max_devices", "locked_test_access",
    "outcome_analysis", "benchmark_eligible", "checkpoint_promotion",
})


class ReleaseError(ValueError):
    """Release evidence is incomplete, inconsistent, or not independently approved."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    # Match validation.preregister_torch_mvp1.canonical_sha exactly (default
    # JSON ASCII escaping is part of the existing immutable protocol contract).
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReleaseError("Git provenance could not be established")
    return result.stdout.strip()


def _source_cell_digest(notebook_bytes: bytes) -> str:
    try:
        notebook = json.loads(notebook_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError("Notebook is not valid UTF-8 JSON") from exc
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise ReleaseError("Notebook has no valid cells list")
    source_cells = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ReleaseError("Notebook cell is malformed")
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]
        if not isinstance(source, list) or any(not isinstance(line, str) for line in source):
            raise ReleaseError("Notebook cell source is malformed")
        source_cells.append({"cell_type": cell.get("cell_type"), "source": "".join(source)})
    return canonical_sha256(source_cells)


def _clean_commit(root: Path) -> tuple[str, str]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ReleaseError("Release manifest requires a clean Git worktree")
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ReleaseError("Git HEAD is not a full immutable commit SHA")
    return commit, _git(root, "rev-parse", "--show-toplevel")


def _commit_file(root: Path, commit: str, relative: Path | str) -> bytes:
    """Read a source artifact from the immutable release commit, not the checkout."""
    path = str(relative)
    if path.startswith("/") or ".." in Path(path).parts:
        raise ReleaseError("Unsafe path in release binding")
    tree_entry = subprocess.run(["git", "-C", str(root), "ls-tree", commit, "--", path],
                                capture_output=True, check=False)
    if tree_entry.returncode:
        raise ReleaseError("Git provenance could not be established")
    entry = tree_entry.stdout.decode("utf-8", errors="replace").strip()
    if not entry or entry.split(None, 1)[0] == "120000":
        raise ReleaseError(f"Required release source is missing or unsafe: {path}")
    result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{path}"],
                            capture_output=True, check=False)
    if result.returncode:
        raise ReleaseError(f"Source commit is missing required content: {path}")
    return result.stdout


def _decode_protocol(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must contain a JSON object")
    return value


def _validate_original_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0":
        raise ReleaseError("Only the preserved pending preregistration is eligible for release preparation")
    if protocol.get("manifest_sha256") != canonical_sha256({key: value for key, value in protocol.items() if key != "manifest_sha256"}):
        raise ReleaseError("Preregistration self-hash is invalid")
    if any(protocol.get(key) is None for key in ("acceptance_thresholds", "seed_splits", "training_config")):
        raise ReleaseError("Preregistration is missing frozen thresholds, seeds, or training configuration")


def _validate_runtime_candidate(candidate: dict[str, Any], protocol: dict[str, Any]) -> None:
    if candidate.get("candidate_sha256_canonical_json_excluding_candidate_sha256") != canonical_sha256(
        {key: value for key, value in candidate.items() if key != "candidate_sha256_canonical_json_excluding_candidate_sha256"}
    ):
        raise ReleaseError("Runtime protocol candidate self-hash is invalid")
    if candidate.get("status") != "DRAFT_REVIEW_REQUIRED_NO_OUTCOMES" or not isinstance(candidate.get("approval"), dict) or candidate["approval"].get("status") != "PENDING_INDEPENDENT_REVIEW_AND_EXPLICIT_APPROVAL":
        raise ReleaseError("Runtime protocol candidate is not in its preserved pending-review state")
    historical = candidate.get("historical_protocol")
    if not isinstance(historical, dict):
        raise ReleaseError("Runtime protocol candidate has no historical protocol binding")
    exact_fields = ("acceptance_thresholds", "seed_splits", "training_config")
    for field in exact_fields:
        if historical.get(field) != protocol.get(field):
            raise ReleaseError(f"Runtime candidate changes historical {field}")


def _validate_runtime(runtime: Any) -> dict[str, Any]:
    if not isinstance(runtime, dict) or set(runtime) != REQUIRED_RUNTIME_FIELDS:
        raise ReleaseError(f"Runtime profile must contain exactly: {', '.join(sorted(REQUIRED_RUNTIME_FIELDS))}")
    for field in ("python", "numpy", "pytorch"):
        if not isinstance(runtime[field], str) or not runtime[field].strip():
            raise ReleaseError(f"Runtime profile field {field} must be a non-empty version string")
    if runtime["cuda_runtime"] is not None and not isinstance(runtime["cuda_runtime"], str):
        raise ReleaseError("cuda_runtime must be a version string or null")
    if not isinstance(runtime["cuda_available"], bool):
        raise ReleaseError("cuda_available must be boolean")
    if isinstance(runtime["gpu_count"], bool) or not isinstance(runtime["gpu_count"], int) or runtime["gpu_count"] < 0:
        raise ReleaseError("gpu_count must be a non-negative integer")
    names = runtime["gpu_names"]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ReleaseError("gpu_names must be a list of non-empty strings")
    if len(names) != runtime["gpu_count"]:
        raise ReleaseError("gpu_count must match gpu_names length")
    if runtime["cuda_available"] != (runtime["gpu_count"] > 0):
        raise ReleaseError("CUDA availability and enumerated GPU count are inconsistent")
    # Copy through JSON so callers cannot mutate the manifest after construction.
    return json.loads(json.dumps(runtime, sort_keys=True))


def build_release_manifest(*, root: Path = ROOT, runtime_profile: dict[str, Any],
                           preregistration: Path = PREREGISTRATION,
                           notebook: Path = NOTEBOOK,
                           staged_notebook: Path) -> dict[str, Any]:
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ReleaseError("Root must be the Git repository top-level directory")
    commit, _ = _clean_commit(root)
    return _build_manifest_from_commit(root=root, commit=commit, runtime_profile=runtime_profile,
                                       preregistration=preregistration, notebook=notebook,
                                       staged_notebook=staged_notebook)


def _build_manifest_from_commit(*, root: Path, commit: str, runtime_profile: dict[str, Any],
                                preregistration: Path, notebook: Path,
                                staged_notebook: Path) -> dict[str, Any]:
    protocol_data = _commit_file(root, commit, preregistration)
    candidate_data = _commit_file(root, commit, RUNTIME_PROTOCOL_CANDIDATE)
    protocol = _decode_protocol(protocol_data, "Preregistration")
    candidate = _decode_protocol(candidate_data, "Runtime protocol candidate")
    _validate_original_protocol(protocol)
    _validate_runtime_candidate(candidate, protocol)
    engine_hashes: dict[str, str] = {}
    for relative in ENGINE_SOURCE_PATHS:
        engine_hashes[relative] = sha256_bytes(_commit_file(root, commit, relative))
    notebook_bytes = _commit_file(root, commit, notebook)
    staged_path = staged_notebook.resolve()
    if not staged_path.is_file() or staged_path.is_symlink():
        raise ReleaseError("Exact staged notebook is missing or unsafe")
    staged_bytes = staged_path.read_bytes()
    body = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "pasture-studies-kaggle-experiment-v1",
        "status": "READY_PENDING_INDEPENDENT_APPROVAL",
        "source": {"commit_sha": commit, "worktree_clean": True},
        "training_engine": {"source_sha256": engine_hashes},
        "notebook": {"source_path": str(notebook), "source_sha256": sha256_bytes(notebook_bytes),
                     "source_cells_sha256": _source_cell_digest(notebook_bytes),
                     # Informational only: do not bind a workstation-specific
                     # absolute path into a manifest consumed by Kaggle.
                     "staged_path": "staged_notebook",
                     "staged_sha256": sha256_bytes(staged_bytes)},
        "protocol": {"path": str(preregistration), "protocol_id": protocol["protocol_id"],
                     "manifest_sha256": protocol["manifest_sha256"],
                     "file_sha256": sha256_bytes(protocol_data)},
        "runtime_protocol_candidate": {
            "path": str(RUNTIME_PROTOCOL_CANDIDATE), "protocol_id": candidate["protocol_id"],
            "status": candidate["status"], "approval_status": candidate["approval"]["status"],
            "candidate_sha256": candidate["candidate_sha256_canonical_json_excluding_candidate_sha256"],
            "file_sha256": sha256_bytes(candidate_data),
            "historical_thresholds_sha256": canonical_sha256(candidate["historical_protocol"]["acceptance_thresholds"]),
            "historical_seed_splits_sha256": canonical_sha256(candidate["historical_protocol"]["seed_splits"]),
            "historical_training_config_sha256": canonical_sha256(candidate["historical_protocol"]["training_config"]),
        },
        "acceptance_thresholds_sha256": canonical_sha256(protocol["acceptance_thresholds"]),
        "runtime_profile": _validate_runtime(runtime_profile),
        "seeds_and_configs": {"seed_splits": protocol["seed_splits"],
                              "training_config": protocol["training_config"],
                              "analysis_protocol": protocol["analysis_protocol"]},
        "locked_test_policy": "No locked-test outcomes may be read until a separately approved pre-outcome freeze is verified.",
        "approval": {"status": APPROVAL_STATUS_PENDING,
                     "reason": "No independently established signing trust root is configured; caller-supplied keys are not trusted."},
    }
    body["release_manifest_sha256"] = canonical_sha256(body)
    return body


def verify_release_manifest(manifest: dict[str, Any], *, root: Path = ROOT,
                            preregistration: Path = PREREGISTRATION,
                            notebook: Path = NOTEBOOK,
                            staged_notebook: Path | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ReleaseError("Release manifest must be a JSON object")
    claimed = manifest.get("release_manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "release_manifest_sha256"}
    if not isinstance(claimed, str) or claimed != canonical_sha256(body):
        raise ReleaseError("Release manifest checksum mismatch")
    source = manifest.get("source")
    commit = source.get("commit_sha") if isinstance(source, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ReleaseError("Manifest source commit is not a full immutable SHA")
    # Verification intentionally reads bound files from source_commit even when
    # HEAD has advanced or the caller's worktree contains later changes.
    _git(root.resolve(), "cat-file", "-e", f"{commit}^{{commit}}")
    if staged_notebook is None:
        staged_note = manifest.get("notebook", {})
        if staged_note.get("staged_sha256") == staged_note.get("source_sha256"):
            staged_notebook = root.resolve() / notebook
        else:
            raise ReleaseError("Supply the exact staged notebook bytes with --staged-notebook")
    rebuilt = _build_manifest_from_commit(root=root.resolve(), commit=commit,
        runtime_profile=manifest.get("runtime_profile"), preregistration=preregistration,
        notebook=notebook, staged_notebook=staged_notebook)
    if rebuilt != manifest:
        raise ReleaseError("Release bindings differ from current source, notebook, protocol, seeds, config, or runtime")


def approval_status(approval: Any, release_manifest_sha256: str) -> str:
    """Return pending unless a future reviewed verifier has a pinned trust root.

    A syntactically plausible artifact is not a signature. There is deliberately
    no caller-controlled key, environment override, or unsigned approval path.
    """
    if not isinstance(approval, dict):
        return APPROVAL_STATUS_PENDING
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        return APPROVAL_STATUS_PENDING
    if approval.get("release_manifest_sha256") != release_manifest_sha256:
        return APPROVAL_STATUS_PENDING
    # No trusted signer/key registry exists in this release; fail closed.
    return APPROVAL_STATUS_PENDING


def authorize_mode(mode: str, manifest: dict[str, Any], approval: Any = None, *,
                   config: Any = None) -> str:
    normalized = mode.strip().upper() if isinstance(mode, str) else ""
    if normalized in ALLOWED_MODES:
        if config is not None:
            raise ReleaseError("A bounded diagnostic config is accepted only with READINESS")
        return normalized
    if normalized == "READINESS":
        return authorize_readiness(config)
    if normalized == "EXPLORATORY":
        if config != EXPLORATORY_CONFIG_VALUE or not isinstance(manifest, dict):
            raise ReleaseError("EXPLORATORY requires the exact fixed A2C-only nonformal config")
        return "EXPLORATORY"
    if normalized == "EXPLORATORY_COMPARISON":
        if config != EXPLORATORY_COMPARISON_CONFIG_VALUE or not isinstance(manifest, dict):
            raise ReleaseError("EXPLORATORY_COMPARISON requires the exact bounded A-D nonformal config")
        return normalized
    if normalized in {"EXPERIMENT", "FULL_VALIDATION"}:
        if config is not None:
            raise ReleaseError("Experimental execution does not accept an unreviewed runtime config")
        digest = manifest.get("release_manifest_sha256") if isinstance(manifest, dict) else ""
        if approval_status(approval, digest) == "APPROVED":
            return normalized
        raise ReleaseError("EXPERIMENT and FULL_VALIDATION are denied: independently verifiable approval signature is unavailable")
    raise ReleaseError("Unsupported execution mode")


def authorize_readiness(config: Any) -> str:
    """Permit only explicitly bounded recovery/distributed diagnostics.

    This is not an experiment mode: it cannot access outcomes, produce
    benchmark-eligible artifacts, or promote a checkpoint.
    """
    if not isinstance(config, dict) or set(config) != READINESS_CONFIG_KEYS:
        raise ReleaseError("READINESS requires the exact bounded diagnostic config schema")
    if config.get("task") not in READINESS_DIAGNOSTIC_TASKS:
        raise ReleaseError("READINESS task is not an allowlisted diagnostic workload")
    workload_bounds = {
        "checkpoint_recovery": {"max_updates": 2, "max_episodes": 32, "max_devices": 2},
        "t4x2_path_smoke": {"max_updates": 1, "max_episodes": 16, "max_devices": 2},
    }[config["task"]]
    for key, expected in (*workload_bounds.items(), ("locked_test_access", False), ("outcome_analysis", False),
                          ("benchmark_eligible", False), ("checkpoint_promotion", False)):
        actual = config.get(key)
        if isinstance(expected, bool):
            valid = type(actual) is bool and actual is expected
        else:
            valid = type(actual) is int and actual == expected
        if not valid:
            raise ReleaseError(f"READINESS config violates fixed bound: {key}")
    return "READINESS"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read valid JSON: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate outcome-blind release bindings")
    generate.add_argument("--runtime-profile", type=Path, required=True, help="JSON with exact runtime version/GPU fields")
    generate.add_argument("--staged-notebook", type=Path, required=True,
                          help="Exact notebook file that will be uploaded by the launcher")
    # A manifest cannot be committed into the source tree it binds without
    # creating a self-referential commit hash; keep it as an external artifact.
    generate.add_argument("--output", type=Path, default=Path("/tmp/pasture-studies-release-manifest.json"))
    verify = commands.add_parser("verify", help="Recompute and verify release bindings")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--staged-notebook", type=Path,
                        help="Exact staged notebook; otherwise the recorded staged path is used")
    gate = commands.add_parser("gate", help="Check mode gate; experimental modes remain denied without trusted signature")
    gate.add_argument("mode", choices=("PREFLIGHT", "SMOKE", "READINESS", "EXPLORATORY", "EXPLORATORY_COMPARISON", "EXPERIMENT", "FULL_VALIDATION"))
    gate.add_argument("manifest", type=Path)
    gate.add_argument("--approval", type=Path, help="Separate approval artifact; not trusted without pinned verifier")
    gate.add_argument("--config", type=Path, help="Exact bounded config required for READINESS or EXPLORATORY mode")
    gate.add_argument("--staged-notebook", type=Path,
                      help="Exact staged notebook bytes bound by the manifest; required when verifying outside the staging host")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            manifest = build_release_manifest(runtime_profile=_load_json(args.runtime_profile),
                                              staged_notebook=args.staged_notebook)
            output = args.output.resolve()
            try:
                output.relative_to(ROOT.resolve())
            except ValueError:
                pass
            else:
                raise ReleaseError("Manifest output must be outside the bound source repository")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"manifest": str(output), "sha256": manifest["release_manifest_sha256"],
                              "status": manifest["status"], "approval": manifest["approval"]["status"]}, indent=2))
        elif args.command == "verify":
            manifest = _load_json(args.manifest)
            verify_release_manifest(manifest, staged_notebook=args.staged_notebook)
            print(json.dumps({"valid": True, "sha256": manifest["release_manifest_sha256"],
                              "status": manifest["status"], "approval": manifest["approval"]["status"]}, indent=2))
        else:
            manifest = _load_json(args.manifest)
            verify_release_manifest(manifest, staged_notebook=args.staged_notebook)
            approval = _load_json(args.approval) if args.approval else None
            if args.mode in {"READINESS", "EXPLORATORY"}:
                if not args.config:
                    raise ReleaseError(f"{args.mode} requires --config")
                authorize_mode(args.mode, manifest, approval, config=_load_json(args.config))
            else:
                if args.config:
                    raise ReleaseError("--config is accepted only with READINESS")
                authorize_mode(args.mode, manifest, approval)
            print(json.dumps({"authorized": True, "mode": args.mode}, indent=2))
    except ReleaseError as exc:
        print(json.dumps({"authorized": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
