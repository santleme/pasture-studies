#!/usr/bin/env python3
"""Validate and stage a frozen training-only corpus for optional Kaggle upload.

This is an offline packaging step. It never invokes the Kaggle CLI or publishes
data; uploading the resulting private-by-default directory is a separate act.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from validation.benchmark_torch_fixed_corpus import load_corpus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_dataset_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError("dataset id must be an owner/slug pair")
    return value


def package_corpus(corpus_path: str | Path, output_dir: str | Path, *,
                   dataset_id: str, title: str, license_name: str,
                   data_version: str) -> dict[str, Any]:
    """Validate a full corpus, then stage only it and its provenance metadata."""
    source = Path(corpus_path)
    arrays, config = load_corpus(source)
    if config.get("smoke_mode") or config.get("split") != "training-only fixed workload; never held-out; stale off-policy compute replay":
        raise ValueError("only a full training-only corpus can be packaged")
    if not isinstance(data_version, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", data_version):
        raise ValueError("data version must be a non-empty version slug")
    dataset_id = _safe_dataset_id(dataset_id)
    title = title.strip()
    license_name = license_name.strip()
    if not title or not license_name:
        raise ValueError("dataset title and an explicitly selected license are required")
    if arrays["world_ids"].size != config["partition_count"] * config["episodes_per_partition"]:
        raise ValueError("corpus episode count does not match its configuration")

    target = Path(output_dir)
    if target.exists():
        if not target.is_dir() or target.is_symlink() or any(target.iterdir()):
            raise ValueError("output directory must be absent or an empty, non-symlink directory")
    else:
        target.mkdir(parents=True)
    corpus_name = f"pasture-studies-{data_version}.npz"
    sidecar_name = f"pasture-studies-{data_version}.json"
    shutil.copyfile(source, target / corpus_name)
    shutil.copyfile(source.with_suffix(".json"), target / sidecar_name)
    metadata = {"id": dataset_id, "title": title,
                "licenses": [{"name": license_name}],
                "description": "Synthetic Pasture Studies fixed training-only simulator corpus. Not held-out policy evaluation or biological evidence.",
                "keywords": ["synthetic", "simulation", "pasture-studies"],
                "resources": [{"path": corpus_name, "description": "Frozen training-only trajectory arrays"},
                              {"path": sidecar_name, "description": "Corpus configuration and provenance"}]}
    metadata_path = target / "dataset-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    names = (corpus_name, sidecar_name, metadata_path.name)
    body = {"schema_version": 1, "dataset_id": dataset_id, "data_version": data_version,
            "split": "training-only", "smoke_mode": False,
            "episode_count": int(arrays["world_ids"].size),
            "corpus_sha256": config["corpus_sha256"],
            "files": {name: {"sha256": sha256_file(target / name), "size_bytes": (target / name).stat().st_size}
                      for name in names}}
    body["manifest_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (target / "dataset-manifest.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def verify_package(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset package must be a real directory")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected != manifest.get("manifest_sha256"):
        raise ValueError("dataset package manifest hash mismatch")
    if manifest.get("split") != "training-only" or manifest.get("smoke_mode") is not False:
        raise ValueError("dataset package must be full and training-only")
    listed = manifest.get("files")
    if (not isinstance(listed, dict) or len(listed) != 3 or "dataset-metadata.json" not in listed
            or sum(name.endswith(".npz") for name in listed) != 1
            or sum(name.endswith(".json") and name != "dataset-metadata.json" for name in listed) != 1):
        raise ValueError("dataset package file inventory is invalid")
    for name, entry in listed.items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("dataset package contains a missing or unsafe listed file")
        if path.stat().st_size != entry.get("size_bytes") or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"dataset package file integrity mismatch: {name}")
    metadata = json.loads((root / "dataset-metadata.json").read_text())
    if metadata.get("id") != manifest.get("dataset_id") or len(metadata.get("licenses", [])) != 1:
        raise ValueError("dataset metadata does not match the manifest or has no single explicit license")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--corpus", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--dataset-id", required=True, help="Kaggle owner/slug")
    package.add_argument("--title", required=True)
    package.add_argument("--license", dest="license_name", required=True, help="select a license you have rights to apply")
    package.add_argument("--data-version", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (package_corpus(args.corpus, args.output, dataset_id=args.dataset_id,
                             title=args.title, license_name=args.license_name,
                             data_version=args.data_version)
              if args.command == "package" else verify_package(args.package))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
