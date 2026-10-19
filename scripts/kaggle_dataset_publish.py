#!/usr/bin/env python3
"""Dry-run-first Kaggle dataset create/version helper for verified bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.pasture_studies_dataset import verify_bundle


def command_for(operation: str, directory: Path, message: str | None = None) -> list[str]:
    if operation == "create":
        return ["kaggle", "datasets", "create", "-p", str(directory)]
    if operation == "version":
        if not message or not message.strip():
            raise ValueError("version operation requires a non-empty message")
        return ["kaggle", "datasets", "version", "-p", str(directory), "-m", message.strip()]
    raise ValueError("operation must be create or version")


def publish(directory: str | Path, *, operation: str, message: str | None = None,
            execute: bool = False) -> dict:
    root = Path(directory)
    manifest = verify_bundle(root)
    command = command_for(operation, root, message)
    result = {"operation": operation, "dataset_id": manifest["dataset_id"],
              "data_version": manifest["data_version"], "manifest_sha256": manifest["manifest_sha256"],
              "dry_run": not execute, "command": command}
    if execute:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "version"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--message", help="required version note for operation=version")
    parser.add_argument("--publish", action="store_true",
                        help="actually invoke Kaggle CLI; without this flag only prints the verified command")
    args = parser.parse_args(argv)
    result = publish(args.directory, operation=args.operation, message=args.message, execute=args.publish)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
