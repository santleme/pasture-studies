#!/usr/bin/env python3
"""Fail-closed Kaggle CLI helpers for publishing Pasture Studies notebooks.

No command logs credentials or starts OAuth. Publishing is refused unless the
installed CLI advertises an explicit no-run flag.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime, timezone
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct `python scripts/kaggle_deploy.py ...` execution otherwise places
    # only the scripts/ directory on sys.path, breaking source-root package
    # imports used by the READINESS and EXPLORATORY gates.
    sys.path.insert(0, str(ROOT))
NOTEBOOK = ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"
MODES = {"PREFLIGHT", "SMOKE", "READINESS", "EXPLORATORY", "EXPLORATORY_COMPARISON"}
T4X2_SHAPE = "NvidiaTeslaT4"
RELEASE_MANIFEST = "release-manifest.json"
DEPLOYMENT_CONFIG = "pasture-deployment.json"
READINESS_CONFIG = "readiness-config.json"
READINESS_CONFIG_VALUE = {
    "task": "checkpoint_recovery", "max_updates": 2, "max_episodes": 32,
    "max_devices": 2, "locked_test_access": False, "outcome_analysis": False,
    "benchmark_eligible": False, "checkpoint_promotion": False,
}
EXPLORATORY_CONFIG_VALUE = {
    "task": "exploratory_a2c_training", "algorithm": "a2c", "episodes": 256,
    "optimizer_updates": 16, "seed": 9482817, "evaluation_episodes": 5,
    "device_count": 1, "locked_test_access": False, "formal_eligible": False,
    "benchmark_eligible": False, "checkpoint_promotion": False,
}
EXPLORATORY_COMPARISON_CONFIG_VALUE = {
    "task": "exploratory_a2c_variant_comparison", "algorithm": "a2c",
    "variants": ["A", "B", "C", "D"], "episodes_per_variant_seed": 512,
    "seeds": [9483001, 9483002, 9483003], "training_jobs": 12, "evaluation_episodes": 5,
    "comparison_eval_seed": 9483999, "comparison_episodes_per_family": 3,
    "max_training_jobs": 24, "max_devices": 2, "locked_test_access": False,
    "variant_config": {
        "A": {"reward_variant": "original", "state_reward_scale": 0.1,
              "normalize_advantage": False, "entropy_coefficient": 0.0, "discount_factor": 1.0},
        "B": {"reward_variant": "original", "state_reward_scale": 0.1,
              "normalize_advantage": True, "entropy_coefficient": 0.01, "discount_factor": 1.0},
        "C": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
              "normalize_advantage": False, "entropy_coefficient": 0.0, "discount_factor": 1.0},
        "D": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
              "normalize_advantage": True, "entropy_coefficient": 0.01, "discount_factor": 1.0},
    },
    "formal_eligible": False, "benchmark_eligible": False, "checkpoint_promotion": False,
}
def _strict_notebook_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys,
                           parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeploymentError("Kaggle notebook is unreadable or has ambiguous JSON") from exc
    if not isinstance(value, dict):
        raise DeploymentError("Kaggle notebook must contain a JSON object")
    return value


def _write_new_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise DeploymentError(f"Refusing to overwrite existing evidence path: {path.name}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_replace_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Replace a file atomically without following a raced final-component symlink."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def deployment_record_path(folder: Path) -> Path:
    return folder.parent / f"{folder.name}-deployment.json"


def readiness_config_path(folder: Path) -> Path:
    return folder.parent / f"{folder.name}-readiness-config.json"


def exploratory_config_path(folder: Path) -> Path:
    return folder.parent / f"{folder.name}-config.json"


class DeploymentError(RuntimeError):
    """An unsafe or invalid deployment operation was refused."""


def run(args: list[str], *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a CLI command without exposing captured errors or the process environment."""
    result = subprocess.run(args, check=False, text=True, capture_output=capture_output)
    if check and result.returncode:
        raise DeploymentError(f"{Path(args[0]).name} command failed (exit {result.returncode}); details suppressed to avoid leaking credentials")
    return result


def kaggle_version() -> str:
    result = run(["kaggle", "--version"])
    match = re.search(r"(?:Kaggle CLI\s+)?([0-9]+(?:\.[0-9]+)+)", result.stdout)
    if not match:
        raise DeploymentError("Unable to determine Kaggle CLI version")
    return match.group(1)


def auth_preflight() -> dict:
    """Report auth as unverified; do not inspect credentials or start OAuth."""
    return {
        "auth_status": "not_verified",
        "auth_check": "not_performed",
        "auth_hint": "Verify existing login separately with `kaggle kernels list --mine --page-size 1`; this helper does not initiate OAuth.",
    }


def check_cli(*, require_no_run: bool = False) -> tuple[str, str]:
    version = kaggle_version()
    help_text = run(["kaggle", "kernels", "push", "--help"]).stdout
    if require_no_run and not re.search(r"(?<!\w)--no-run(?:\b|[= ])", help_text):
        raise DeploymentError(
            f"Kaggle CLI {version} does not advertise `kernels push --no-run`; refusing push because this CLI may execute the notebook. Upgrade Kaggle CLI and re-check its help."
        )
    return version, help_text


def validate_mode(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in MODES:
        raise DeploymentError("Only PREFLIGHT, SMOKE, READINESS, and exploratory modes are permitted; formal experiments remain disabled")
    return normalized


def metadata_for(owner: str, *, title: str = "Pasture Studies", mode: str = "SMOKE",
                 gpu: bool = False) -> dict:
    validate_mode(mode)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise DeploymentError("Kaggle owner must be a plain username slug")
    title = title.strip()
    if not title:
        raise DeploymentError("Kernel title must not be empty")
    metadata = {
        "id": f"{owner}/pasture-studies",
        "title": title,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": T4X2_SHAPE if gpu else "",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    return metadata


def _source_commit() -> str:
    status = run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"])
    if status.stdout.strip():
        raise DeploymentError("Source checkout must be clean and committed before staging")
    return _head_commit()


def _head_commit() -> str:
    result = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DeploymentError("Could not establish the exact source commit SHA")
    return commit


def _committed_notebook_bytes(commit: str) -> bytes:
    result = run(["git", "-C", str(ROOT), "show", f"{commit}:notebooks/pasture_studies_kaggle.ipynb"])
    return result.stdout.encode("utf-8")


def _stage_notebook_bytes(source_bytes: bytes, deployment: dict) -> bytes:
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    deployment = canonical(deployment)
    try:
        notebook = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentError("Committed source notebook is unreadable or invalid JSON") from exc
    marker = "PASTURE_DEPLOYMENT = None  # KAGGLE_DEPLOYMENT_CONFIG_MARKER"
    matching = [cell for cell in notebook.get("cells", []) if marker in "".join(cell.get("source", []))]
    if len(matching) != 1:
        raise DeploymentError("Committed notebook must contain exactly one safe deployment-config injection marker")
    cell = matching[0]
    source = "".join(cell.get("source", []))
    source = source.replace(marker, "PASTURE_DEPLOYMENT = " + repr(deployment), 1)
    cell["source"] = source.splitlines(keepends=True)
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _manifest_digest(path: Path, *, staged_notebook: Path | None = None) -> str:
    try:
        manifest = json.loads(path.read_text())
        from scripts.kaggle_release import verify_release_manifest
        verify_release_manifest(manifest, root=ROOT, staged_notebook=staged_notebook)
    except Exception as exc:
        raise DeploymentError("Release manifest failed source/protocol verification") from exc
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder: Path, owner: str, *, mode: str = "SMOKE", gpu: bool = False) -> Path:
    mode = validate_mode(mode)
    if folder.is_symlink():
        raise DeploymentError(f"Staging path must not be a symlink: {folder}")
    source_commit = _source_commit()
    if folder.exists():
        if not folder.is_dir() or any(folder.iterdir()):
            raise DeploymentError(f"Staging directory must be empty to prevent uploading stale files: {folder}")
    else:
        folder.mkdir(parents=True)
    _ensure_detached_paths_available(folder)
    if mode == "READINESS":
        raise DeploymentError("READINESS staging requires --runtime-profile; use prepare-release")
    if mode in {"EXPLORATORY", "EXPLORATORY_COMPARISON"}:
        raise DeploymentError("Exploratory staging requires --runtime-profile; use a prepare-exploratory command")
    source_bytes = _committed_notebook_bytes(source_commit)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    deployment = {"schema_version": 1, "mode": mode, "source_commit": source_commit,
                  "source_notebook_sha256": source_sha,
                  "readiness_config": None, "exploratory_config": None, "runtime_profile": None,
                  "release_manifest_sha256": None, "gpu_required": bool(gpu)}
    return _write_staging(folder, owner, deployment)


def _write_staging(folder: Path, owner: str, deployment: dict) -> Path:
    gpu = deployment["gpu_required"]
    staged_bytes = _stage_notebook_bytes(_committed_notebook_bytes(deployment["source_commit"]), deployment)
    (folder / NOTEBOOK.name).write_bytes(staged_bytes)
    staged_hash = hashlib.sha256(staged_bytes).hexdigest()
    _write_new_file(deployment_record_path(folder), (json.dumps({**deployment, "notebook_sha256": staged_hash}, sort_keys=True, indent=2) + "\n").encode())
    target = folder / "kernel-metadata.json"
    target.write_text(json.dumps(metadata_for(owner, mode=deployment["mode"], gpu=gpu), indent=2) + "\n")
    if deployment["mode"] == "READINESS":
        _write_new_file(readiness_config_path(folder), (json.dumps(READINESS_CONFIG_VALUE, sort_keys=True, indent=2) + "\n").encode())
    if deployment["mode"] in {"EXPLORATORY", "EXPLORATORY_COMPARISON"}:
        exploratory_config = (EXPLORATORY_CONFIG_VALUE if deployment["mode"] == "EXPLORATORY"
                              else EXPLORATORY_COMPARISON_CONFIG_VALUE)
        _write_new_file(exploratory_config_path(folder),
                        (json.dumps(exploratory_config, sort_keys=True, indent=2) + "\n").encode())
    return target


def prepare_release(folder: Path, owner: str, *, runtime_profile_path: Path,
                    gpu: bool = True) -> Path:
    """Prepare READINESS with a fixed workload and pinned target runtime profile."""
    try:
        runtime_profile = json.loads(runtime_profile_path.read_text())
        from scripts.kaggle_release import _validate_runtime
        runtime_profile = _validate_runtime(runtime_profile)
    except Exception as exc:
        raise DeploymentError("READINESS runtime profile is invalid") from exc
    if (not gpu or not runtime_profile["cuda_available"] or runtime_profile["gpu_count"] != 2
            or any("T4" not in name for name in runtime_profile["gpu_names"])):
        raise DeploymentError("READINESS requires a CUDA-enabled two-T4 runtime profile")
    if folder.is_symlink():
        raise DeploymentError(f"Staging path must not be a symlink: {folder}")
    source_commit = _source_commit()
    if folder.exists():
        if not folder.is_dir() or any(folder.iterdir()):
            raise DeploymentError(f"Staging directory must be empty to prevent uploading stale files: {folder}")
    else:
        folder.mkdir(parents=True)
    _ensure_detached_paths_available(folder)
    source_bytes = _committed_notebook_bytes(source_commit)
    deployment = {"schema_version": 1, "mode": "READINESS", "source_commit": source_commit,
                  "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "readiness_config": READINESS_CONFIG_VALUE, "exploratory_config": None,
                  "runtime_profile": runtime_profile,
                  "release_manifest_sha256": None, "gpu_required": True}
    return _write_staging(folder, owner, deployment)


def prepare_exploratory(folder: Path, owner: str, *, runtime_profile_path: Path,
                        gpu: bool = True, comparison: bool = False) -> Path:
    """Stage a fixed-budget nonformal A2C run or paired variant comparison."""
    try:
        runtime_profile = json.loads(runtime_profile_path.read_text())
        from scripts.kaggle_release import _validate_runtime
        runtime_profile = _validate_runtime(runtime_profile)
    except Exception as exc:
        raise DeploymentError("Exploratory runtime profile is invalid") from exc
    if (not gpu or not runtime_profile["cuda_available"] or runtime_profile["gpu_count"] != 2
            or any("T4" not in name for name in runtime_profile["gpu_names"])):
        raise DeploymentError("Exploratory runs require a CUDA-enabled T4x2 Kaggle profile")
    if folder.is_symlink():
        raise DeploymentError(f"Staging path must not be a symlink: {folder}")
    source_commit = _source_commit()
    if folder.exists():
        if not folder.is_dir() or any(folder.iterdir()):
            raise DeploymentError(f"Staging directory must be empty to prevent uploading stale files: {folder}")
    else:
        folder.mkdir(parents=True)
    _ensure_detached_paths_available(folder)
    source_bytes = _committed_notebook_bytes(source_commit)
    mode = "EXPLORATORY_COMPARISON" if comparison else "EXPLORATORY"
    exploratory_config = EXPLORATORY_COMPARISON_CONFIG_VALUE if comparison else EXPLORATORY_CONFIG_VALUE
    deployment = {"schema_version": 1, "mode": mode, "source_commit": source_commit,
                  "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "readiness_config": None, "exploratory_config": exploratory_config,
                  "runtime_profile": runtime_profile, "release_manifest_sha256": None,
                  "gpu_required": True}
    return _write_staging(folder, owner, deployment)


def _ensure_detached_paths_available(folder: Path) -> None:
    for detached in (deployment_record_path(folder), readiness_config_path(folder),
                     exploratory_config_path(folder),
                     detached_manifest_path(folder), folder.parent / f"{folder.name}-publication-attestation.json",
                     folder.parent / f"{folder.name}-execution-attestation.json"):
        if detached.is_symlink():
            raise DeploymentError("Detached deployment evidence path must not be a symlink")
        if detached.exists():
            raise DeploymentError("Detached deployment evidence already exists; use a fresh staging path")


def bind_release(folder: Path, release_manifest: Path) -> str:
    """Verify exact staged bytes; retain the manifest as detached host evidence."""
    if not release_manifest.is_file():
        raise DeploymentError("Release manifest file is missing")
    if folder.is_symlink() or not folder.is_dir():
        raise DeploymentError("Staging folder must be a real directory")
    metadata_path = folder / "kernel-metadata.json"
    config_path = deployment_record_path(folder)
    manifest_output = detached_manifest_path(folder)
    if config_path.is_symlink():
        raise DeploymentError("Deployment config path must not be a symlink")
    if manifest_output.exists() or manifest_output.is_symlink():
        raise DeploymentError("Detached release-manifest path already exists")
    try:
        metadata = json.loads(metadata_path.read_text())
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Readiness staging metadata/config is missing or malformed") from exc
    notebook = folder / metadata.get("code_file", "")
    if (config.get("mode") not in {"READINESS", "EXPLORATORY", "EXPLORATORY_COMPARISON"} or metadata.get("enable_gpu") != "true"
            or not notebook.is_file() or notebook.is_symlink()):
        raise DeploymentError("Release manifests may only bind a GPU READINESS/EXPLORATORY staging folder")
    manifest_sha = _manifest_digest(release_manifest, staged_notebook=notebook)
    manifest = json.loads(release_manifest.read_text())
    staged = manifest.get("notebook", {})
    if (manifest.get("source", {}).get("commit_sha") != config.get("source_commit")
            or staged.get("staged_sha256") != hashlib.sha256(notebook.read_bytes()).hexdigest()
            or staged.get("source_sha256") != config.get("source_notebook_sha256")):
        raise DeploymentError("Release manifest does not bind the exact source and staged notebook")
    config["release_manifest_sha256"] = manifest_sha
    _atomic_replace_file(config_path, (json.dumps(config, sort_keys=True, indent=2) + "\n").encode())
    _write_new_file(manifest_output, release_manifest.read_bytes())
    validate_staging_folder(folder)
    return manifest_sha


def detached_manifest_path(folder: Path) -> Path:
    return folder.parent / f"{folder.name}-{RELEASE_MANIFEST}"


def publish(folder: Path) -> None:
    """Publish only when `--no-run` is supported; never silently execute push."""
    validate_staging_folder(folder)
    version, _ = check_cli(require_no_run=True)
    result = run(["kaggle", "kernels", "push", "--path", str(folder), "--no-run"])
    metadata, _ = validate_staging_folder(folder)
    revision = revision_from_push_output(result.stdout)
    return write_publication_attestation(folder, revision, cli_version=version)


def revision_from_push_output(output: str) -> int:
    """Extract the server-assigned revision printed by Kaggle CLI after push."""
    # Official --no-run says "successfully saved without running", whereas an
    # executing push says "successfully pushed". Both include the same
    # server-assigned version number; accepting only the latter made a
    # successful no-run save look like a failed publish after it had happened.
    matches = re.findall(
        r"Kernel version\s+([1-9][0-9]*)\s+successfully\s+(?:pushed|saved without running)",
        output, flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        raise DeploymentError("Kaggle push output did not confirm exactly one numeric kernel version")
    return int(matches[0])


def _notebook_source_digest(path: Path) -> str:
    notebook = _strict_notebook_json(path)
    sources = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        sources.append({"cell_type": cell.get("cell_type"), "source": "".join(source) if isinstance(source, list) else source})
    return hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _notebook_content_digest(path: Path) -> str:
    """Hash all notebook content after normalizing API source list/string encoding."""
    notebook = _strict_notebook_json(path)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise DeploymentError("Kaggle notebook has no valid cell collection")
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise DeploymentError("Kaggle notebook contains a malformed cell")
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            cell["source"] = "".join(source)
        elif not isinstance(source, str):
            raise DeploymentError("Kaggle notebook contains an invalid cell source")
    canonical = json.dumps(notebook, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def embedded_deployment_config(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PASTURE_DEPLOYMENT" for target in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    return value
    raise DeploymentError("Notebook has no literal embedded deployment configuration")


def pull_and_compare(folder: Path, kernel: str, revision: int) -> dict:
    """Pull a numeric Kaggle revision and compare exact bytes and source cells."""
    metadata, staged = validate_staging_folder(folder)
    if not isinstance(revision, int) or revision <= 0:
        raise DeploymentError("Remote revision must be a positive numeric version")
    with tempfile.TemporaryDirectory(prefix="pasture-kaggle-pull-") as tmp:
        run(["kaggle", "kernels", "pull", f"{kernel}/{revision}", "--path", tmp])
        candidates = list(Path(tmp).glob("*.ipynb"))
        if len(candidates) != 1:
            raise DeploymentError("Exact Kaggle pull must return exactly one notebook")
        remote = candidates[0]
        staged_hash = hashlib.sha256(staged.read_bytes()).hexdigest()
        remote_hash = hashlib.sha256(remote.read_bytes()).hexdigest()
        staged_cells = _notebook_source_digest(staged)
        remote_cells = _notebook_source_digest(remote)
        staged_content = _notebook_content_digest(staged)
        remote_content = _notebook_content_digest(remote)
        if staged_content != remote_content or staged_cells != remote_cells:
            raise DeploymentError("Pulled numeric Kaggle revision differs from the staged notebook content")
    config = json.loads(deployment_record_path(folder).read_text())
    return {"kernel": kernel, "revision": revision, "source_commit": config["source_commit"],
            "notebook_sha256": staged_hash, "source_cells_sha256": staged_cells,
            "remote_notebook_sha256": remote_hash, "notebook_content_sha256": staged_content,
            "remote_notebook_content_sha256": remote_content, "raw_bytes_identical": staged_hash == remote_hash,
            "exact_match": True}


def write_publication_attestation(folder: Path, revision: int, *, cli_version: str,
                                  execution: bool = False) -> dict:
    """Pull the exact numeric revision, compare it, then persist the attestation."""
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    evidence = pull_and_compare(folder, metadata["id"], revision)
    config = json.loads(deployment_record_path(folder).read_text())
    _, staged_notebook = validate_staging_folder(folder)
    local_raw = hashlib.sha256(staged_notebook.read_bytes()).hexdigest()
    local_content = _notebook_content_digest(staged_notebook)
    local_cells = _notebook_source_digest(staged_notebook)
    raw_hashes_match = evidence.get("remote_notebook_sha256") == evidence.get("notebook_sha256")
    if (evidence.get("exact_match") is not True
            or evidence.get("kernel") != metadata.get("id")
            or evidence.get("source_commit") != config.get("source_commit")
            or evidence.get("notebook_sha256") != config.get("notebook_sha256")
            or evidence.get("notebook_sha256") != local_raw
            or evidence.get("notebook_content_sha256") != local_content
            or evidence.get("source_cells_sha256") != local_cells
            or evidence.get("notebook_content_sha256") != evidence.get("remote_notebook_content_sha256")
            or evidence.get("raw_bytes_identical") is not raw_hashes_match
            or not isinstance(evidence.get("revision"), int) or evidence["revision"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("remote_notebook_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("notebook_content_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("source_cells_sha256", "")))):
        raise DeploymentError("Cannot attest publication without exact source-bound notebook content")
    record = {
        "schema_version": 1, "kernel": metadata["id"],
        "kernel_version": int(evidence["revision"]),
        "source_commit": config["source_commit"], "mode": config["mode"],
        "release_manifest_sha256": config.get("release_manifest_sha256"),
        "staged_notebook_sha256": evidence["notebook_sha256"],
        "remote_notebook_sha256": evidence["remote_notebook_sha256"],
        "canonical_notebook_content_sha256": evidence["notebook_content_sha256"],
        "raw_notebook_bytes_identical": bool(evidence.get("raw_bytes_identical")),
        "notebook_source_cells_sha256": evidence["source_cells_sha256"],
        "canonical_content_match": True, "kaggle_cli_version": cli_version,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    suffix = "execution-attestation" if execution else "publication-attestation"
    output = folder.parent / f"{folder.name}-{suffix}.json"
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _write_new_file(output, payload)
    result = dict(evidence)
    result["attestation_path"] = str(output)
    return result



def staged_mode(notebook_path: Path) -> str:
    try:
        notebook = json.loads(notebook_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Staged notebook is unreadable or invalid JSON") from exc
    content = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    if "PASTURE_DEPLOYMENT = " not in content or "KAGGLE_DEPLOYMENT_CONFIG_MARKER" in content:
        raise DeploymentError("Notebook must embed the explicit immutable deployment config")
    return "SOURCE_BOUND"


def validate_staging_folder(folder: Path) -> tuple[dict, Path]:
    """Require a private, generated staging folder containing only its notebook and metadata."""
    if folder.is_symlink() or not folder.is_dir():
        raise DeploymentError("Staging path must be a real directory")
    metadata_path = folder / "kernel-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Staging metadata is missing or malformed") from exc
    if not isinstance(metadata, dict):
        raise DeploymentError("Staging metadata must be a JSON object")
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file or Path(code_file).name != code_file:
        raise DeploymentError("Metadata code_file must be a local filename")
    notebook_path = folder / code_file
    if metadata.get("is_private") != "true":
        raise DeploymentError("Refusing to publish a kernel that is not explicitly private")
    if not notebook_path.is_file() or notebook_path.is_symlink() or metadata_path.is_symlink():
        raise DeploymentError("Staging notebook and metadata must be regular local files")
    allowed = {"kernel-metadata.json", code_file}
    if {item.name for item in folder.iterdir()} != allowed:
        raise DeploymentError("Staging folder contains unexpected/unbound files")
    if metadata.get("enable_gpu") == "true":
        if metadata.get("machine_shape") != T4X2_SHAPE:
            raise DeploymentError("GPU metadata must select the supported T4 x2 machine shape")
    elif metadata.get("machine_shape") not in {"", None}:
        raise DeploymentError("CPU metadata must not select an accelerator machine shape")
    config_path = deployment_record_path(folder)
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Deployment configuration is missing or malformed") from exc
    mode = validate_mode(config.get("mode", ""))
    if hashlib.sha256(notebook_path.read_bytes()).hexdigest() != config.get("notebook_sha256"):
        raise DeploymentError("Deployment config does not bind the exact staged notebook bytes")
    required_config_keys = {"schema_version", "mode", "source_commit", "source_notebook_sha256",
                            "readiness_config", "exploratory_config", "runtime_profile", "release_manifest_sha256",
                            "gpu_required", "notebook_sha256"}
    if set(config) != required_config_keys or config.get("schema_version") != 1:
        raise DeploymentError("Deployment record has an unsupported or incomplete schema")
    if config.get("source_commit") != _head_commit():
        raise DeploymentError("Deployment record source commit is not the current immutable checkout")
    try:
        committed_source = _committed_notebook_bytes(config["source_commit"])
    except DeploymentError:
        raise
    if hashlib.sha256(committed_source).hexdigest() != config.get("source_notebook_sha256"):
        raise DeploymentError("Deployment record source notebook SHA differs from immutable source commit")
    expected_embedded = {key: value for key, value in config.items() if key != "notebook_sha256"}
    if config.get("release_manifest_sha256"):
        expected_embedded["release_manifest_sha256"] = None
    if embedded_deployment_config(notebook_path) != expected_embedded:
        raise DeploymentError("Embedded deployment config differs from detached source-bound record")
    expected_staged = _stage_notebook_bytes(committed_source, expected_embedded)
    if notebook_path.read_bytes() != expected_staged:
        raise DeploymentError("Staged notebook differs from the exact committed source plus approved config injection")
    if mode in {"READINESS", "EXPLORATORY", "EXPLORATORY_COMPARISON"}:
        expected_config = (READINESS_CONFIG_VALUE if mode == "READINESS" else
                           EXPLORATORY_CONFIG_VALUE if mode == "EXPLORATORY" else
                           EXPLORATORY_COMPARISON_CONFIG_VALUE)
        config_field = "readiness_config" if mode == "READINESS" else "exploratory_config"
        other_field = "exploratory_config" if mode == "READINESS" else "readiness_config"
        if (metadata.get("enable_gpu") != "true" or config.get(config_field) != expected_config
                or config.get(other_field) is not None or not isinstance(config.get("runtime_profile"), dict)):
            raise DeploymentError(f"{mode} staging requires GPU, release manifest, and exact mode config")
        try:
            from scripts.kaggle_release import _validate_runtime
            profile = _validate_runtime(config["runtime_profile"])
        except Exception as exc:
            raise DeploymentError("READINESS runtime profile failed validation") from exc
        if not profile["cuda_available"] or profile["gpu_count"] != 2 or any("T4" not in n for n in profile["gpu_names"]):
            raise DeploymentError("READINESS deployment is not bound to two T4 CUDA devices")
        manifest_path = detached_manifest_path(folder)
        if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config.get("release_manifest_sha256"):
            raise DeploymentError("Release manifest digest differs from deployment config")
        manifest_digest = _manifest_digest(manifest_path, staged_notebook=notebook_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest_digest != config.get("release_manifest_sha256") or manifest.get("source", {}).get("commit_sha") != config.get("source_commit"):
            raise DeploymentError("Staged release manifest is not bound to deployment source commit")
        if manifest.get("notebook", {}).get("staged_sha256") != config.get("notebook_sha256"):
            raise DeploymentError("Staged release manifest does not bind the exact notebook bytes")
        if config.get(config_field) != expected_config:
            raise DeploymentError(f"{mode} config differs from the fixed allowlisted workload")
        if mode in {"EXPLORATORY", "EXPLORATORY_COMPARISON"}:
            from scripts.kaggle_release import authorize_mode
            authorize_mode(mode, manifest, config=expected_config)
    elif config.get("release_manifest_sha256"):
        raise DeploymentError("Non-readiness deployment must not carry a release manifest claim")
    elif (config.get("readiness_config") is not None or config.get("exploratory_config") is not None
          or config.get("runtime_profile") is not None):
        raise DeploymentError("Non-readiness deployment must not carry bounded-run configuration")
    if type(config.get("gpu_required")) is not bool:
        raise DeploymentError("Deployment GPU requirement must be boolean")
    if bool(config.get("gpu_required")) != (metadata.get("enable_gpu") == "true"):
        raise DeploymentError("Deployment config GPU requirement does not match Kaggle metadata")
    notebook_doc = json.loads(notebook_path.read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook_doc.get("cells", []))
    if "EXPERIMENT" not in notebook_source or "FULL_VALIDATION" not in notebook_source:
        raise DeploymentError("Notebook must explicitly hard-block experimental modes")
    if "PASTURE_DEPLOYMENT = " not in notebook_source or "KAGGLE_DEPLOYMENT_CONFIG_MARKER" in notebook_source:
        raise DeploymentError("Notebook is missing the deployment provenance contract")
    return metadata, notebook_path


def execute(folder: Path, *, mode: str, confirm_execution: bool, confirm_gpu: bool = False) -> None:
    """Run a staged notebook only after explicit execution and optional GPU consent."""
    mode = validate_mode(mode)
    if not confirm_execution:
        raise DeploymentError("Execution requires explicit --confirm-execution")
    metadata, notebook = validate_staging_folder(folder)
    if staged_mode(notebook) != "SOURCE_BOUND":
        raise DeploymentError("Notebook does not implement the source-bound deployment contract")
    config = json.loads(deployment_record_path(folder).read_text())
    if config.get("mode") != mode:
        raise DeploymentError("Requested execution mode does not match the staged deployment config")
    gpu = metadata.get("enable_gpu") == "true"
    if gpu and not confirm_gpu:
        raise DeploymentError("T4 x2 execution requires separate explicit --confirm-gpu")
    if not gpu and confirm_gpu:
        raise DeploymentError("--confirm-gpu was supplied for a CPU staging folder")
    args = ["kaggle", "kernels", "push", "--path", str(folder)]
    if gpu:
        args += ["--accelerator", T4X2_SHAPE]
    result = run(args)
    revision = revision_from_push_output(result.stdout)
    version, _ = check_cli()
    return write_publication_attestation(folder, revision, cli_version=version, execution=True)


def retrieve(kind: str, kernel: str, destination: Path | None = None) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?", kernel):
        raise DeploymentError("Kernel must be an owner/slug identifier with an optional numeric version")
    commands = {
        "status": ["kaggle", "kernels", "status", kernel],
        "logs": ["kaggle", "kernels", "logs", kernel],
        "output": ["kaggle", "kernels", "output", kernel],
    }
    args = commands[kind]
    if destination is not None and kind == "output":
        destination.mkdir(parents=True, exist_ok=True)
        args += ["--path", str(destination)]
    result = run(args)
    return redact(result.stdout)


def redact(value: str) -> str:
    """Redact common bearer/token/credential assignments from CLI output."""
    value = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)
    return re.sub(
        r"(?i)((?:api[_-]?token|access[_-]?token|refresh[_-]?token|kaggle[_-]?key|password|secret|key)\s*[:=]\s*[\"']?)[^\s\"',}]+",
        r"\1[REDACTED]", value,
    )


def verify_manifest(manifest_path: Path, artifact_root: Path) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DeploymentError("Artifact manifest is missing or malformed") from exc
    if not isinstance(entries, list):
        raise DeploymentError("Artifact manifest files must be an array")
    if not entries:
        raise DeploymentError("Artifact manifest contains no files to verify")
    verified = []
    root = artifact_root.resolve()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DeploymentError("Malformed artifact manifest entry")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DeploymentError(f"Unsafe artifact path in manifest: {entry['path']}")
        path = (root / Path(*relative.parts)).resolve()
        if path != root and root not in path.parents:
            raise DeploymentError(f"Artifact escapes output directory: {entry['path']}")
        if not path.is_file():
            raise DeploymentError(f"Manifest artifact is missing: {entry['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) or digest != entry["sha256"]:
            raise DeploymentError(f"SHA-256 mismatch: {entry['path']}")
        if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
            raise DeploymentError(f"Size mismatch: {entry['path']}")
        verified.append(entry["path"])
    return verified


def write_deployment_record(path: Path, *, kernel: str, version: str, source_commit: str,
                            kernel_version: str, mode: str, metadata_path: Path) -> None:
    if path.is_symlink():
        raise DeploymentError("Deployment record output path must not be a symlink")
    validate_mode(mode)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise DeploymentError("source_commit must be a full 40-character Git SHA-1")
    if not re.fullmatch(r"[1-9][0-9]*", kernel_version.strip()):
        raise DeploymentError("kernel_version must be an exact numeric Kaggle revision")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Staging metadata is unreadable or invalid JSON") from exc
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or Path(code_file).name != code_file:
        raise DeploymentError("Metadata must identify a staged notebook filename")
    notebook_path = metadata_path.parent / code_file
    if not notebook_path.is_file() or notebook_path.is_symlink():
        raise DeploymentError("Staged notebook for provenance record is missing")
    mode = validate_mode(mode)
    if staged_mode(notebook_path) != "SOURCE_BOUND":
        raise DeploymentError("Deployment record notebook is not source-bound")
    folder = metadata_path.parent
    config = json.loads(deployment_record_path(folder).read_text())
    if config.get("mode") != mode or config.get("source_commit") != source_commit:
        raise DeploymentError("Deployment config mode/source SHA does not match the record")
    remote = pull_and_compare(folder, kernel, int(kernel_version))
    record = {
        "kernel": kernel, "kaggle_kernel_version": kernel_version.strip(),
        "source_commit": source_commit,
        "kaggle_cli_version": version, "mode": mode,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "staged_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "staged_source_cells_sha256": remote["source_cells_sha256"],
        "remote_notebook_sha256": remote["remote_notebook_sha256"],
        "remote_exact_content_match": remote["exact_match"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_replace_file(path, (json.dumps(record, indent=2) + "\n").encode())


def preflight() -> dict:
    version, help_text = check_cli()
    auth = auth_preflight()
    return {"kaggle_cli_version": version, **auth,
            "no_run_supported": bool(re.search(r"(?<!\w)--no-run(?:\b|[= ])", help_text))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    prep = sub.add_parser("prepare")
    prep.add_argument("--path", type=Path, required=True)
    prep.add_argument("--owner", required=True)
    prep.add_argument("--mode", default="SMOKE")
    prep.add_argument("--gpu", action="store_true", help="Select Kaggle T4 x2 machine shape")
    prep_release = sub.add_parser("prepare-release", help="Stage fixed READINESS notebook for a verified T4x2 runtime")
    prep_release.add_argument("--path", type=Path, required=True)
    prep_release.add_argument("--owner", required=True)
    prep_release.add_argument("--runtime-profile", type=Path, required=True)
    prep_exploratory = sub.add_parser("prepare-exploratory", help="Stage a fixed-budget nonformal A2C Kaggle run")
    prep_exploratory.add_argument("--path", type=Path, required=True)
    prep_exploratory.add_argument("--owner", required=True)
    prep_exploratory.add_argument("--runtime-profile", type=Path, required=True)
    prep_comparison = sub.add_parser("prepare-exploratory-comparison", help="Stage a bounded nonformal A-D comparison")
    prep_comparison.add_argument("--path", type=Path, required=True)
    prep_comparison.add_argument("--owner", required=True)
    prep_comparison.add_argument("--runtime-profile", type=Path, required=True)
    bind = sub.add_parser("bind-release")
    bind.add_argument("--path", type=Path, required=True)
    bind.add_argument("--manifest", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--path", type=Path, required=True)
    run_cmd = sub.add_parser("execute")
    run_cmd.add_argument("--path", type=Path, required=True)
    run_cmd.add_argument("--mode", choices=sorted(MODES), required=True)
    run_cmd.add_argument("--confirm-execution", action="store_true")
    run_cmd.add_argument("--confirm-gpu", action="store_true")
    for command in ("status", "logs", "output"):
        item = sub.add_parser(command)
        item.add_argument("kernel")
        if command == "output":
            item.add_argument("--path", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--path", type=Path, required=True)
    compare.add_argument("--kernel", required=True)
    compare.add_argument("--version", type=int, required=True, help="Exact numeric Kaggle version reported by kernels push")
    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--artifacts", type=Path, required=True)
    record = sub.add_parser("record")
    record.add_argument("path", type=Path)
    record.add_argument("--kernel", required=True)
    record.add_argument("--source-commit", required=True)
    record.add_argument("--kernel-version", required=True,
                        help="Exact Kaggle kernel version/revision captured from Kaggle")
    record.add_argument("--mode", default="SMOKE")
    record.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(), indent=2))
        elif args.command == "prepare":
            print(prepare(args.path, args.owner, mode=args.mode, gpu=args.gpu))
        elif args.command == "prepare-release":
            print(prepare_release(args.path, args.owner, runtime_profile_path=args.runtime_profile))
        elif args.command == "prepare-exploratory":
            print(prepare_exploratory(args.path, args.owner, runtime_profile_path=args.runtime_profile))
        elif args.command == "prepare-exploratory-comparison":
            print(prepare_exploratory(args.path, args.owner, runtime_profile_path=args.runtime_profile,
                                      comparison=True))
        elif args.command == "bind-release":
            print(json.dumps({"release_manifest_sha256": bind_release(args.path, args.manifest)}, indent=2))
        elif args.command == "publish":
            print(json.dumps(publish(args.path), indent=2))
        elif args.command == "execute":
            print(json.dumps(execute(args.path, mode=args.mode, confirm_execution=args.confirm_execution,
                    confirm_gpu=args.confirm_gpu), indent=2))
        elif args.command in {"status", "logs", "output"}:
            print(retrieve(args.command, args.kernel, getattr(args, "path", None)))
        elif args.command == "compare":
            print(json.dumps(pull_and_compare(args.path, args.kernel, args.version), indent=2))
        elif args.command == "verify":
            print(json.dumps({"verified": verify_manifest(args.manifest, args.artifacts)}, indent=2))
        elif args.command == "record":
            version, _ = check_cli()
            write_deployment_record(args.path, kernel=args.kernel, version=version,
                                    source_commit=args.source_commit, kernel_version=args.kernel_version,
                                    mode=args.mode,
                                    metadata_path=args.metadata)
    except (DeploymentError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
