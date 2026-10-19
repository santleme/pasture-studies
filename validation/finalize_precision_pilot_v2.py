#!/usr/bin/env python3
"""Fail-closed v2 precision-pilot finalizer.

This path never accepts caller-supplied standard deviations. It verifies a
separately approved protocol/release, signed pilot authorization, independent
run attestation, exact raw-result bytes, and the preregistered schedule before
recomputing sample SD and delta0. Trust roots are intentionally empty until an
independent governance process provisions them. No locked outcomes are read.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from validation.preregister_torch_mvp1 import FAMILIES, canonical_sha
from validation.run_torch_precision_pilot import PILOT_SOURCES

MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_RESULTS_BYTES = 64 * 1024 * 1024
# Populate only through an independently reviewed governance change.
TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256: frozenset[str] = frozenset()
TRUSTED_PILOT_ATTESTATION_ROOT_SHA256: frozenset[str] = frozenset()
TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256: frozenset[str] = frozenset()
# This must be provisioned on independently protected durable storage before
# enabling any trust roots. The location is not caller-selectable (otherwise
# an authorization could be replayed against a fresh empty audit file).
PILOT_AUDIT_STORE_PATH = Path("/var/lib/pasture-studies/pilot-audit.jsonl")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bytes(path: str | Path, *, limit: int) -> bytes:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{path.name} must be a bounded regular file")
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{path.name} exceeds its size limit")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_json(path: str | Path, *, limit: int = MAX_METADATA_BYTES) -> tuple[dict[str, Any], str]:
    payload = _read_bytes(path, limit=limit)
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{Path(path).name} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return document, _sha(payload)


def _verify_signature(document: Mapping[str, Any], *, root_path: str | Path,
                      allowed_fingerprints: frozenset[str], label: str) -> tuple[str, str]:
    root = Path(root_path)
    root_bytes = _read_bytes(root, limit=1024 * 1024)
    encoded = document.get("signature_ed25519_base64")
    if not isinstance(encoded, str):
        raise ValueError(f"{label} denied: signature is missing")
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{label} denied: signature encoding is invalid") from exc
    body = {key: value for key, value in document.items() if key != "signature_ed25519_base64"}
    message = _canonical(body)
    with tempfile.TemporaryDirectory(prefix="pasture-pilot-auth-") as temporary:
        # Verify the exact bytes that were fingerprinted and allowlisted above.
        # Do not reopen root_path here: an attacker could replace it between
        # the digest check and OpenSSL's open().
        root_snapshot = Path(temporary) / "trusted-root.pem"
        message_path = Path(temporary) / "message.json"
        signature_path = Path(temporary) / "signature.bin"
        root_snapshot.write_bytes(root_bytes)
        try:
            canonical_key = subprocess.run(
                ["openssl", "pkey", "-pubin", "-in", str(root_snapshot), "-outform", "DER"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as exc:
            raise ValueError(f"{label} denied: OpenSSL Ed25519 verifier is unavailable") from exc
        if canonical_key.returncode != 0 or not canonical_key.stdout:
            raise ValueError(f"{label} denied: public-key root is invalid")
        # Fingerprint the canonical SubjectPublicKeyInfo DER, not its PEM
        # serialization, so whitespace/re-encoding cannot create a new signer.
        root_sha = _sha(canonical_key.stdout)
        if root_sha not in allowed_fingerprints:
            raise ValueError(f"{label} denied: signer root is not allowlisted")
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(root_snapshot), "-rawin",
                 "-in", str(message_path), "-sigfile", str(signature_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as exc:
            raise ValueError(f"{label} denied: OpenSSL Ed25519 verifier is unavailable") from exc
    if result.returncode != 0:
        raise ValueError(f"{label} denied: signature verification failed")
    return root_sha, _sha(_canonical(document))


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def _runtime(document: Any) -> bool:
    required = {"python", "numpy", "pytorch", "cuda_runtime", "cuda_available", "gpu_count", "gpu_names"}
    return (isinstance(document, dict) and set(document) == required
            and all(isinstance(document[key], str) and document[key]
                    for key in ("python", "numpy", "pytorch", "cuda_runtime"))
            and type(document["cuda_available"]) is bool
            and type(document["gpu_count"]) is int
            and isinstance(document["gpu_names"], list)
            and all(isinstance(name, str) and name for name in document["gpu_names"]))


def _current_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("pilot denied: PyTorch runtime is unavailable") from exc
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = ([torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                 if cuda_available else [])
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pytorch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda or "none"),
            "cuda_available": cuda_available, "gpu_count": len(gpu_names), "gpu_names": gpu_names}


def _audit_append(path: str | Path, entry: dict[str, Any]) -> str:
    """Append/fsync metadata before the first read of raw pilot outcomes.

    The JSONL hash chain is tamper-evident only when the configured storage
    independently protects against truncation/replacement.
    """
    target = Path(path)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("pilot audit must be an owner-only regular file")
        os.lseek(fd, 0, os.SEEK_SET)
        existing = bytearray()
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            existing.extend(chunk)
            if len(existing) > MAX_METADATA_BYTES:
                raise ValueError("pilot audit exceeds size limit")
        if existing and not existing.endswith(b"\n"):
            raise ValueError("pilot audit has incomplete trailing record")
        previous = "0" * 64
        used_ids = set()
        for line in existing.splitlines():
            row = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            body = {key: value for key, value in row.items() if key != "record_sha256"}
            if row.get("previous_record_sha256") != previous or row.get("record_sha256") != _sha(_canonical(body)):
                raise ValueError("pilot audit hash chain is invalid")
            if row.get("authorization_id"):
                used_ids.add(row["authorization_id"])
            previous = row["record_sha256"]
        if entry.get("authorization_id") in used_ids:
            raise ValueError("pilot authorization has already been consumed")
        body = {**entry, "previous_record_sha256": previous}
        record = {**body, "record_sha256": _sha(_canonical(body))}
        payload = _canonical(record) + b"\n"
        os.lseek(fd, 0, os.SEEK_END)
        if os.write(fd, payload) != len(payload):
            raise OSError("short pilot audit write")
        os.fsync(fd)
        return record["record_sha256"]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _validate_cells(doc: dict[str, Any], *, manifest: dict[str, Any],
                    schedule_sha256: str, candidate_sha256: str,
                    release_manifest_sha256: str) -> dict[str, float]:
    allowed = {"schema_version", "status", "protocol_id", "candidate_sha256",
               "release_manifest_sha256", "manifest_sha256", "pilot_seed_ids_sha256",
               "schedule_sha256", "cells"}
    if set(doc) != allowed or doc.get("schema_version") != 2 or doc.get("status") != "RAW_PRECISION_PILOT_CELLS":
        raise ValueError("pilot raw-results schema is invalid; SD-only or extra-field inputs are rejected")
    if (doc.get("candidate_sha256") != candidate_sha256
            or doc.get("release_manifest_sha256") != release_manifest_sha256
            or doc.get("manifest_sha256") != manifest["manifest_sha256"]
            or doc.get("protocol_id") != manifest["protocol_id"]
            or doc.get("pilot_seed_ids_sha256") != manifest["pilot_seed_ids_sha256"]
            or doc.get("schedule_sha256") != schedule_sha256):
        raise ValueError("pilot raw results do not match the authorized protocol/schedule")
    seeds = manifest["seed_splits"]["training_root_seeds"]
    worlds = manifest["seed_splits"]["precision_pilot"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(seeds) != len(set(seeds))
            or not isinstance(worlds, dict) or set(worlds) != set(FAMILIES)):
        raise ValueError("pilot registered seed schedule has invalid types or duplicates")
    for family in FAMILIES:
        if (not isinstance(worlds[family], list) or not worlds[family]
                or any(type(seed) is not int for seed in worlds[family])
                or len(worlds[family]) != len(set(worlds[family]))):
            raise ValueError(f"pilot registered world schedule is invalid for {family}")
    expected = {(family, train_seed, world_seed)
                for family in FAMILIES for train_seed in seeds for world_seed in worlds[family]}
    cells = doc.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("pilot raw-results cells are incomplete or contain extras")
    seen = set()
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {"family", "training_root_seed", "world_seed", "episode_total_return"}:
            raise ValueError("pilot cell schema contains unsupported or missing fields")
        family, train_seed, world_seed = cell["family"], cell["training_root_seed"], cell["world_seed"]
        if not isinstance(family, str) or type(train_seed) is not int or type(world_seed) is not int:
            raise ValueError("pilot cell identity fields must be a family string and integer seeds")
        key = (family, train_seed, world_seed)
        value = cell["episode_total_return"]
        if key not in expected or key in seen:
            raise ValueError("pilot cells include an unregistered or duplicate schedule cell")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("pilot episode return must be finite numeric data")
        seen.add(key)
        by_family[family].append(float(value))
    if seen != expected:
        raise ValueError("pilot cells do not cover the exact registered schedule")
    sds = {}
    for family, values in by_family.items():
        sd = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"pilot sample SD must be finite and positive for {family}")
        sds[family] = sd
    return sds


def finalize_precision_pilot_v2(*, candidate_path: str | Path, release_manifest_path: str | Path,
                                pending_manifest_path: str | Path, authorization_path: str | Path,
                                authorization_root_path: str | Path, run_attestation_path: str | Path,
                                attestation_root_path: str | Path, protocol_review_path: str | Path,
                                protocol_review_root_path: str | Path, raw_results_path: str | Path,
                                output_path: str | Path) -> dict[str, Any]:
    """Verify external authorization/attestation, then consume and mask pilot cells.

    The trusted-key allowlists are empty in this draft. Under current state the
    function rejects before reading ``raw_results_path`` or writing output.
    """
    candidate, candidate_file_sha = _read_json(candidate_path)
    candidate_sha = candidate.get("candidate_sha256")
    if candidate_sha != canonical_sha({key: value for key, value in candidate.items() if key != "candidate_sha256"}):
        raise ValueError("pilot denied: candidate canonical hash is invalid")
    approval = candidate.get("approval")
    if (candidate.get("status") != "APPROVED_PRE_OUTCOME_PROTOCOL"
            or candidate.get("release_eligible") is not True
            or not isinstance(approval, dict) or approval.get("pilot_authorized") is not True
            or approval.get("independent_review") != "PASS"
            or approval.get("explicit_user_authorization_for_outcomes") is not True):
        raise ValueError("pilot denied: candidate is not independently approved for precision-pilot outcomes")

    release, release_file_sha = _read_json(release_manifest_path)
    if set(release) != {"protocol_candidate_sha256", "source_commit_sha", "source_sha256",
                        "runtime_profile", "notebook_publication", "release_manifest_sha256"}:
        raise ValueError("pilot denied: detached release manifest schema is invalid")
    release_body = {key: value for key, value in release.items() if key != "release_manifest_sha256"}
    if (release.get("release_manifest_sha256") != canonical_sha(release_body)
            or release.get("protocol_candidate_sha256") != candidate_sha):
        raise ValueError("pilot denied: detached release manifest does not bind this candidate")
    expected_sources = candidate.get("provenance", {}).get("source_sha256")
    if (not isinstance(expected_sources, dict) or not set(PILOT_SOURCES).issubset(expected_sources)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in expected_sources.values())):
        raise ValueError("pilot denied: protocol candidate does not pin pilot sources")
    if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
           for name in expected_sources):
        raise ValueError("pilot denied: candidate source paths must be safe relative paths")
    source_digests = {name: _sha(_read_bytes(Path(__file__).resolve().parents[1] / name,
                                                     limit=MAX_RESULTS_BYTES)) for name in expected_sources}
    if (not set(PILOT_SOURCES).issubset(expected_sources)
            or expected_sources != source_digests
            or release.get("source_sha256") != expected_sources):
        raise ValueError("pilot denied: current source hashes differ from approved release")
    commit, dirty = _git_state()
    if dirty or release.get("source_commit_sha") != commit:
        raise ValueError("pilot denied: source checkout is dirty or differs from approved release commit")
    runtime = release.get("runtime_profile")
    if (not _runtime(runtime) or candidate.get("runtime_profile_observed") != runtime
            or _current_runtime() != runtime):
        raise ValueError("pilot denied: runtime profile is not exactly bound by the candidate")
    publication = release.get("notebook_publication")
    if (not isinstance(publication, dict) or not isinstance(publication.get("kernel"), str)
            or type(publication.get("version")) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", str(publication.get("content_sha256", "")))
            or publication != candidate.get("notebook_publication")):
        raise ValueError("pilot denied: notebook publication identity is incomplete or mismatched")

    manifest, manifest_file_sha = _read_json(pending_manifest_path)
    if (manifest.get("status") != "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0"
            or manifest.get("manifest_sha256") != canonical_sha({key: value for key, value in manifest.items()
                                                                  if key != "manifest_sha256"})
            or manifest.get("protocol_id") != candidate.get("candidate_id")
            or not isinstance(manifest.get("source_sha256"), dict)
            or any(manifest["source_sha256"].get(name) != source_digests[name] for name in PILOT_SOURCES)):
        raise ValueError("pilot denied: pending v2 manifest/hash/source binding is invalid")
    training_seeds = manifest.get("seed_splits", {}).get("training_root_seeds")
    pilot_worlds = manifest.get("seed_splits", {}).get("precision_pilot")
    if (not isinstance(training_seeds, list) or not training_seeds
            or any(type(seed) is not int for seed in training_seeds)
            or len(training_seeds) != len(set(training_seeds))
            or not isinstance(pilot_worlds, dict) or set(pilot_worlds) != set(FAMILIES)):
        raise ValueError("pilot denied: preregistered seed schedule is invalid")
    for family in FAMILIES:
        if (not isinstance(pilot_worlds[family], list) or not pilot_worlds[family]
                or any(type(seed) is not int for seed in pilot_worlds[family])
                or len(pilot_worlds[family]) != len(set(pilot_worlds[family]))):
            raise ValueError(f"pilot denied: preregistered world schedule is invalid for {family}")
    schedule = {"training_root_seeds": training_seeds,
                "precision_pilot_world_seeds": {family: pilot_worlds[family] for family in FAMILIES},
                "actor_policy": "untrained ActorCritic(value_head=False, seed=training_root_seed)",
                "evaluation_order": "family-major, world-seed, then training-root-seed"}
    schedule_sha = canonical_sha(schedule)
    seed_ids_sha = canonical_sha({family: pilot_worlds[family] for family in FAMILIES})
    if (manifest.get("protocol_id") != candidate["candidate_id"]
            or manifest.get("pilot_seed_ids_sha256") != seed_ids_sha):
        raise ValueError("pilot denied: pending manifest is not the approved v2 protocol/schedule")

    authorization, authorization_file_sha = _read_json(authorization_path)
    auth_fields = {"schema_version", "authorization_id", "action", "candidate_sha256",
                   "candidate_file_sha256", "release_manifest_sha256", "pending_manifest_file_sha256",
                   "pending_manifest_sha256", "source_commit_sha", "schedule_sha256", "runtime_sha256",
                   "notebook_publication_sha256", "expires_at_utc", "signature_ed25519_base64"}
    if set(authorization) != auth_fields:
        raise ValueError("pilot denied: authorization schema contains unsupported/missing fields")
    if type(authorization.get("schema_version")) is not int or not isinstance(authorization.get("authorization_id"), str):
        raise ValueError("pilot denied: authorization identity/schema is invalid")
    review, review_file_sha = _read_json(protocol_review_path)
    review_fields = {"schema_version", "action", "candidate_sha256", "candidate_file_sha256",
                     "release_manifest_sha256", "pending_manifest_file_sha256",
                     "pending_manifest_sha256", "source_commit_sha", "schedule_sha256",
                     "notebook_publication_sha256", "signature_ed25519_base64"}
    if set(review) != review_fields or type(review.get("schema_version")) is not int:
        raise ValueError("pilot denied: signed protocol-review schema is invalid")
    review_root_sha, review_signature_sha = _verify_signature(
        review, root_path=protocol_review_root_path,
        allowed_fingerprints=TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256, label="independent protocol review")
    review_expected = {"schema_version": 1, "action": "approve_precision_pilot_protocol",
                       "candidate_sha256": candidate_sha,
                       "candidate_file_sha256": candidate_file_sha,
                       "release_manifest_sha256": release_file_sha,
                       "pending_manifest_file_sha256": manifest_file_sha,
                       "pending_manifest_sha256": manifest["manifest_sha256"],
                       "source_commit_sha": commit, "schedule_sha256": schedule_sha,
                       "notebook_publication_sha256": canonical_sha(publication)}
    if any(review.get(key) != value for key, value in review_expected.items()):
        raise ValueError("pilot denied: independent review does not bind exact candidate/release/schedule")

    auth_root_sha, auth_sha = _verify_signature(authorization, root_path=authorization_root_path,
        allowed_fingerprints=TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256, label="pilot authorization")
    auth_expected = {"schema_version": 1, "action": "collect_precision_pilot",
                     "candidate_sha256": candidate_sha,
                     "candidate_file_sha256": candidate_file_sha,
                     "release_manifest_sha256": release_file_sha,
                     "pending_manifest_file_sha256": manifest_file_sha,
                     "pending_manifest_sha256": manifest["manifest_sha256"],
                     "source_commit_sha": commit, "schedule_sha256": schedule_sha,
                     "runtime_sha256": canonical_sha(runtime),
                     "notebook_publication_sha256": canonical_sha(publication)}
    if any(authorization.get(key) != value for key, value in auth_expected.items()):
        raise ValueError("pilot denied: signed authorization does not bind the exact candidate/release/schedule")
    expires = authorization.get("expires_at_utc")
    try:
        expiry = dt.datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("pilot denied: authorization expiry is missing/invalid") from exc
    if expiry.tzinfo is None or expiry <= dt.datetime.now(dt.timezone.utc):
        raise ValueError("pilot denied: authorization has expired or lacks a timezone")

    attestation, attestation_file_sha = _read_json(run_attestation_path)
    attestation_fields = {"schema_version", "run_id", "action", "authorization_id",
                          "authorization_sha256", "candidate_sha256", "release_manifest_sha256",
                          "pending_manifest_file_sha256", "pending_manifest_sha256", "source_commit_sha",
                          "schedule_sha256", "runtime_sha256", "notebook_publication_sha256",
                          "raw_results_sha256", "raw_results_bytes", "signature_ed25519_base64"}
    if set(attestation) != attestation_fields:
        raise ValueError("pilot denied: run attestation schema contains unsupported/missing fields")
    if (type(attestation.get("schema_version")) is not int or not isinstance(attestation.get("run_id"), str)
            or attestation.get("schema_version") != 1 or not attestation["run_id"].strip()
            or type(attestation.get("raw_results_bytes")) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", str(attestation.get("raw_results_sha256", "")))):
        raise ValueError("pilot denied: run attestation identity/schema is invalid")
    att_root_sha, att_sha = _verify_signature(attestation, root_path=attestation_root_path,
        allowed_fingerprints=TRUSTED_PILOT_ATTESTATION_ROOT_SHA256, label="pilot run attestation")
    if len({review_root_sha, auth_root_sha, att_root_sha}) != 3:
        raise ValueError("pilot denied: protocol reviewer, authorizer, and run attester must be distinct signers")
    raw_path = Path(raw_results_path)
    raw_stat = os.stat(raw_path, follow_symlinks=False)
    if not stat.S_ISREG(raw_stat.st_mode) or raw_stat.st_size > MAX_RESULTS_BYTES:
        raise ValueError("pilot denied: raw results must be a bounded regular file")
    att_expected = {"schema_version": 1, "action": "attest_precision_pilot_results",
                    "authorization_id": authorization.get("authorization_id"),
                    "authorization_sha256": authorization_file_sha,
                    "candidate_sha256": candidate_sha,
                    "release_manifest_sha256": release_file_sha,
                    "pending_manifest_file_sha256": manifest_file_sha,
                    "pending_manifest_sha256": manifest["manifest_sha256"],
                    "source_commit_sha": commit, "schedule_sha256": schedule_sha,
                    "runtime_sha256": canonical_sha(runtime),
                    "notebook_publication_sha256": canonical_sha(publication),
                    "raw_results_sha256": attestation.get("raw_results_sha256"),
                    "raw_results_bytes": raw_stat.st_size}
    if any(attestation.get(key) != value for key, value in att_expected.items()):
        raise ValueError("pilot denied: independent run attestation is incomplete or mismatched")
    if not isinstance(authorization.get("authorization_id"), str) or not authorization["authorization_id"].strip():
        raise ValueError("pilot denied: authorization identifier is invalid")

    # Durable metadata access record is synced before the first outcome bytes
    # are read. The operational storage system must prevent log replacement.
    audit_sha = _audit_append(PILOT_AUDIT_STORE_PATH, {
        "schema_version": 1, "action": "precision_pilot_raw_results_access",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_file_sha,
        "trusted_authorizer_root_sha256": auth_root_sha,
        "protocol_review_file_sha256": review_file_sha,
        "protocol_review_signature_sha256": review_signature_sha,
        "trusted_protocol_review_root_sha256": review_root_sha,
        "run_attestation_sha256": attestation_file_sha,
        "trusted_attestation_root_sha256": att_root_sha,
        "candidate_sha256": candidate_sha, "release_manifest_sha256": release_file_sha,
        "pending_manifest_sha256": manifest["manifest_sha256"],
        "raw_results_sha256": attestation["raw_results_sha256"]})
    raw_bytes = _read_bytes(raw_path, limit=MAX_RESULTS_BYTES)
    if len(raw_bytes) != attestation["raw_results_bytes"]:
        raise ValueError("pilot denied: raw result size differs from independent attestation")
    if _sha(raw_bytes) != attestation["raw_results_sha256"]:
        raise ValueError("pilot denied: raw result bytes differ from independent attestation")
    raw_doc = _decode_raw_json(raw_bytes)
    if not isinstance(raw_doc, dict):
        raise ValueError("pilot denied: raw result document must be an object")
    sds = _validate_cells(raw_doc, manifest=manifest, schedule_sha256=schedule_sha,
                          candidate_sha256=candidate_sha, release_manifest_sha256=release_file_sha)
    delta0 = {family: 0.05 * sds[family] for family in FAMILIES}
    result = {"schema_version": 2, "status": "MASKED_PRECISION_PILOT_VERIFIED",
              "protocol_id": candidate["candidate_id"], "candidate_sha256": candidate_sha,
              "release_manifest_file_sha256": release_file_sha,
              "pending_manifest_file_sha256": manifest_file_sha,
              "pending_manifest_sha256": manifest["manifest_sha256"],
              "pilot_seed_ids_sha256": seed_ids_sha, "schedule_sha256": schedule_sha,
              "authorization_id": authorization["authorization_id"],
              "authorization_file_sha256": authorization_file_sha,
              "authorization_signature_sha256": auth_sha,
              "run_attestation_file_sha256": attestation_file_sha,
              "run_attestation_signature_sha256": att_sha,
              "raw_results_sha256": _sha(raw_bytes),
              "source_commit_sha": commit,
              "notebook_publication": publication, "runtime": runtime,
              "cell_count_by_family": {family: len(pilot_worlds[family]) * len(training_seeds)
                                        for family in FAMILIES},
              "sd_method": "sample standard deviation (ddof=1) over exact registered training-root × pilot-world cells",
              "sd_untrained_returns_by_family": sds, "delta0_by_family": delta0,
              "audit_access_record_sha256": audit_sha, "locked_outcomes_read": False}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if os.write(fd, output_bytes) != len(output_bytes):
            raise OSError("short masked pilot result write")
        os.fsync(fd)
    finally:
        os.close(fd)
    return result


def _decode_raw_json(payload: bytes) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("pilot raw results are not strict JSON") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "release-manifest", "pending-manifest", "authorization",
                 "authorization-root", "protocol-review", "protocol-review-root",
                 "run-attestation", "attestation-root", "raw-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize_precision_pilot_v2(candidate_path=args.candidate,
            release_manifest_path=args.release_manifest, pending_manifest_path=args.pending_manifest,
            authorization_path=args.authorization, authorization_root_path=args.authorization_root,
            run_attestation_path=args.run_attestation, attestation_root_path=args.attestation_root,
            protocol_review_path=args.protocol_review,
            protocol_review_root_path=args.protocol_review_root,
            raw_results_path=args.raw_results, output_path=args.output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    # Output only masked cohort statistics and provenance; never the raw file.
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "delta0_by_family": result["delta0_by_family"],
                      "output_sha256": _sha(_canonical(result))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
