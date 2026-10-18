#!/usr/bin/env python3
"""Fail-closed Kaggle CLI helpers for publishing Pasture Studies notebooks.

No command logs credentials or starts OAuth. Publishing is refused unless the
installed CLI advertises an explicit no-run flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"
KAGGLE_CONFIG = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
MODES = {"PREFLIGHT", "SMOKE"}


class DeploymentError(RuntimeError):
    """An unsafe or invalid deployment operation was refused."""


def run(args: list[str], *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a CLI command without passing secrets via argv or printing its env."""
    result = subprocess.run(args, check=False, text=True, capture_output=capture_output)
    if check and result.returncode:
        message = result.stderr.strip() or f"command exited {result.returncode}"
        raise DeploymentError(f"{Path(args[0]).name} command failed: {message}")
    return result


def kaggle_version() -> str:
    result = run(["kaggle", "--version"])
    match = re.search(r"(?:Kaggle CLI\s+)?([0-9]+(?:\.[0-9]+)+)", result.stdout)
    if not match:
        raise DeploymentError("Unable to determine Kaggle CLI version")
    return match.group(1)


def auth_preflight() -> dict:
    """Check for existing local auth material without displaying or requesting it."""
    config = KAGGLE_CONFIG
    token_file = config.parent / "access_token"
    env_present = bool(os.environ.get("KAGGLE_API_TOKEN") or
                       (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))
    config_present = config.is_file()
    token_file_present = token_file.is_file()
    if token_file_present:
        if os.name != "nt" and token_file.stat().st_mode & 0o077:
            raise DeploymentError(f"Credentials file permissions are too broad: {token_file} (expected 0600)")
        if not token_file.read_text().strip():
            raise DeploymentError(f"Credentials file is empty: {token_file}")
    if config_present:
        if os.name != "nt" and config.stat().st_mode & 0o077:
            raise DeploymentError(f"Credentials file permissions are too broad: {config} (expected 0600)")
        try:
            data = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DeploymentError(f"Credentials file is unreadable or invalid JSON: {config}") from exc
        if not ((data.get("username") and data.get("key")) or data.get("key")):
            raise DeploymentError("Credentials file has no recognized Kaggle credential fields")
    if not config_present and not token_file_present and not env_present:
        raise DeploymentError("No existing Kaggle credentials found. Authenticate interactively with `kaggle auth login`; this helper never starts OAuth.")
    source = "legacy_file" if config_present else "access_token_file" if token_file_present else "environment"
    return {"credentials_available": True, "credentials_source": source}


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
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4" if gpu else "",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    return metadata


def prepare(folder: Path, owner: str, *, mode: str = "SMOKE", gpu: bool = False) -> Path:
    validate_mode(mode)
    if not NOTEBOOK.is_file():
        raise DeploymentError(f"Notebook not found: {NOTEBOOK}")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK, folder / NOTEBOOK.name)
    target = folder / "kernel-metadata.json"
    target.write_text(json.dumps(metadata_for(owner, mode=mode, gpu=gpu), indent=2) + "\n")
    return target


def publish(folder: Path) -> None:
    """Publish only when `--no-run` is supported; never silently execute push."""
    check_cli(require_no_run=True)
    run(["kaggle", "kernels", "push", "--path", str(folder), "--no-run"])


def retrieve(kind: str, kernel: str, destination: Path | None = None) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel):
        raise DeploymentError("Kernel must be an owner/slug identifier")
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
    return result.stdout


def verify_manifest(manifest_path: Path, artifact_root: Path) -> list[str]:
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DeploymentError("Artifact manifest is missing or malformed") from exc
    if not isinstance(entries, list):
        raise DeploymentError("Artifact manifest files must be an array")
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
                            mode: str, metadata_path: Path) -> None:
    validate_mode(mode)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise DeploymentError("source_commit must be a full 40-character Git SHA-1")
    record = {
        "kernel": kernel, "source_commit": source_commit,
        "kaggle_cli_version": version, "mode": mode,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def preflight() -> dict:
    version, _ = check_cli()
    auth = auth_preflight()
    return {"kaggle_cli_version": version, **auth,
            "no_run_supported": "--no-run" in run(["kaggle", "kernels", "push", "--help"]).stdout}


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
        elif args.command in {"status", "logs", "output"}:
            print(retrieve(args.command, args.kernel, getattr(args, "path", None)))
        elif args.command == "verify":
            print(json.dumps({"verified": verify_manifest(args.manifest, args.artifacts)}, indent=2))
        elif args.command == "record":
            version, _ = check_cli()
            write_deployment_record(args.path, kernel=args.kernel, version=version,
                                    source_commit=args.source_commit, mode=args.mode,
                                    metadata_path=args.metadata)
    except (DeploymentError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
