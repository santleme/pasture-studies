#!/usr/bin/env python3
"""Fail-closed Kaggle CLI helpers for publishing Pasture Studies notebooks.

No command logs credentials or starts OAuth. Publishing is refused unless the
installed CLI advertises an explicit no-run flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"
MODES = {"PREFLIGHT", "SMOKE"}
T4X2_SHAPE = "NvidiaTeslaT4"


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
        raise DeploymentError("Only PREFLIGHT and SMOKE are permitted; experimental modes are disabled")
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


def prepare(folder: Path, owner: str, *, mode: str = "SMOKE", gpu: bool = False) -> Path:
    mode = validate_mode(mode)
    if not NOTEBOOK.is_file():
        raise DeploymentError(f"Notebook not found: {NOTEBOOK}")
    if folder.is_symlink():
        raise DeploymentError(f"Staging path must not be a symlink: {folder}")
    if folder.exists():
        if not folder.is_dir() or any(folder.iterdir()):
            raise DeploymentError(f"Staging directory must be empty to prevent uploading stale files: {folder}")
    else:
        folder.mkdir(parents=True)
    try:
        notebook = json.loads(NOTEBOOK.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Source notebook is unreadable or invalid JSON") from exc
    marker = 'MODE = os.environ.get("PASTURE_MODE", "SMOKE")'
    matching_cells = [cell for cell in notebook.get("cells", []) if marker in "".join(cell.get("source", []))]
    if len(matching_cells) != 1:
        raise DeploymentError("Could not safely identify the notebook execution-mode cell")
    cell = matching_cells[0]
    source = "".join(cell["source"])
    source = source.replace(marker, f'os.environ["PASTURE_MODE"] = "{mode}"\n{marker}', 1)
    if gpu:
        cuda_marker = 'MODE = os.environ.get("PASTURE_MODE", "SMOKE").strip().upper()'
        if source.count(cuda_marker) != 1:
            raise DeploymentError("Could not safely identify the notebook CUDA-required guard")
        source = source.replace(cuda_marker,
            'os.environ["PASTURE_REQUIRE_CUDA"] = "1"\n' + cuda_marker, 1)
    cell["source"] = source.splitlines(keepends=True)
    (folder / NOTEBOOK.name).write_text(json.dumps(notebook, indent=1) + "\n")
    target = folder / "kernel-metadata.json"
    target.write_text(json.dumps(metadata_for(owner, mode=mode, gpu=gpu), indent=2) + "\n")
    return target


def publish(folder: Path) -> None:
    """Publish only when `--no-run` is supported; never silently execute push."""
    validate_staging_folder(folder)
    check_cli(require_no_run=True)
    run(["kaggle", "kernels", "push", "--path", str(folder), "--no-run"])


def staged_mode(notebook_path: Path) -> str:
    try:
        notebook = json.loads(notebook_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("Staged notebook is unreadable or invalid JSON") from exc
    content = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    matches = [mode for mode in MODES if f'os.environ["PASTURE_MODE"] = "{mode}"' in content]
    if len(matches) != 1:
        raise DeploymentError("Staged notebook must contain exactly one fixed PREFLIGHT/SMOKE mode")
    return matches[0]


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
    if {item.name for item in folder.iterdir()} != {"kernel-metadata.json", code_file}:
        raise DeploymentError("Staging folder may contain only kernel-metadata.json and its code_file")
    if metadata.get("enable_gpu") == "true":
        if metadata.get("machine_shape") != T4X2_SHAPE:
            raise DeploymentError("GPU metadata must select the supported T4 x2 machine shape")
    elif metadata.get("machine_shape") not in {"", None}:
        raise DeploymentError("CPU metadata must not select an accelerator machine shape")
    staged_mode(notebook_path)
    notebook_doc = json.loads(notebook_path.read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook_doc.get("cells", []))
    if metadata.get("enable_gpu") == "true" and notebook_source.count(
            'os.environ["PASTURE_REQUIRE_CUDA"] = "1"') != 1:
        raise DeploymentError("GPU staging must contain exactly one fail-closed PASTURE_REQUIRE_CUDA=1 assertion")
    return metadata, notebook_path


def execute(folder: Path, *, mode: str, confirm_execution: bool, confirm_gpu: bool = False) -> None:
    """Run a staged notebook only after explicit execution and optional GPU consent."""
    mode = validate_mode(mode)
    if not confirm_execution:
        raise DeploymentError("Execution requires explicit --confirm-execution")
    metadata, notebook = validate_staging_folder(folder)
    if staged_mode(notebook) != mode:
        raise DeploymentError("Requested execution mode does not match the staged notebook mode")
    gpu = metadata.get("enable_gpu") == "true"
    if gpu and not confirm_gpu:
        raise DeploymentError("T4 x2 execution requires separate explicit --confirm-gpu")
    if not gpu and confirm_gpu:
        raise DeploymentError("--confirm-gpu was supplied for a CPU staging folder")
    args = ["kaggle", "kernels", "push", "--path", str(folder)]
    if gpu:
        args += ["--accelerator", T4X2_SHAPE]
    run(args)


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
    validate_mode(mode)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise DeploymentError("source_commit must be a full 40-character Git SHA-1")
    if not kernel_version.strip():
        raise DeploymentError("kernel_version is required and must be captured from Kaggle, not inferred")
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
    if staged_mode(notebook_path) != mode:
        raise DeploymentError("Deployment record mode does not match the staged notebook")
    record = {
        "kernel": kernel, "kaggle_kernel_version": kernel_version.strip(),
        "source_commit": source_commit,
        "kaggle_cli_version": version, "mode": mode,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "staged_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


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
        elif args.command == "publish":
            publish(args.path)
        elif args.command == "execute":
            execute(args.path, mode=args.mode, confirm_execution=args.confirm_execution,
                    confirm_gpu=args.confirm_gpu)
        elif args.command in {"status", "logs", "output"}:
            print(retrieve(args.command, args.kernel, getattr(args, "path", None)))
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
