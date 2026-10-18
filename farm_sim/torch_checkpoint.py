"""Crash-safe, hash-verified checkpoint generations for resumable training."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import stat
from typing import Any

FORMAT_VERSION = 1
STATE_FILENAME = "training.pt"
_GENERATION_NAME = re.compile(r"^generation-[0-9]{8}$")


def _read_regular_file(path: Path) -> bytes:
    """Read one regular file without following a symlink at open time."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"expected a regular checkpoint file: {path}")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        try:
            dfd = os.open(path.parent, os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except (AttributeError, OSError):
            pass
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_generation(directory: str | Path, *, model, optimizer, state: dict[str, Any],
                    config: dict[str, Any], algorithm: str, rank: int = 0) -> Path:
    """Commit one immutable generation then atomically point LATEST at it.

    Call only after a fully synchronized optimizer step. In DDP only rank zero
    writes; all ranks should barrier after this function returns.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for resumable checkpoints") from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    generation = int(state.get("generation", state.get("update", 0)))
    if generation < 0:
        raise ValueError("checkpoint generation must be non-negative")
    name = f"generation-{generation:08d}"
    target = root / name
    if target.exists():
        raise FileExistsError(f"checkpoint generation already exists: {target}")
    staging = root / f".{name}.tmp-{os.getpid()}"
    staging.mkdir()
    try:
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        _assert_finite_tensors(model_state, "model")
        optimizer_state = optimizer.state_dict()
        _assert_finite_tensors(optimizer_state, "optimizer")
        _assert_finite_tensors(state, "training state")
        torch.save({"model": model_state, "optimizer": optimizer_state, "state": state}, staging / STATE_FILENAME)
        payload = {
            "format_version": FORMAT_VERSION,
            "algorithm": algorithm,
            "config": config,
            "state_file": STATE_FILENAME,
            "state_sha256": _sha256(staging / STATE_FILENAME),
            "generation": generation,
        }
        (staging / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        with (staging / "manifest.json").open("rb") as f:
            os.fsync(f.fileno())
        os.replace(staging, target)
        _atomic_bytes(root / "LATEST", (name + "\n").encode())
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _is_complete_generation(path: Path, *, expected_algorithm: str | None = None,
                            expected_config: dict | None = None,
                            allow_world_size_migration: bool = False) -> bool:
    """Check whether a renamed generation is a compatible complete commit candidate."""
    try:
        import torch
        if path.is_symlink() or not path.is_dir():
            return False
        manifest_path = path / "manifest.json"
        state_path = path / STATE_FILENAME
        if (manifest_path.is_symlink() or state_path.is_symlink() or
                not manifest_path.is_file() or not state_path.is_file()):
            return False
        manifest = json.loads(_read_regular_file(manifest_path).decode("utf-8"))
        match = _GENERATION_NAME.fullmatch(path.name)
        generation = int(path.name.removeprefix("generation-")) if match else -1
        if (manifest.get("format_version") != FORMAT_VERSION or
                manifest.get("state_file") != STATE_FILENAME or
                manifest.get("generation") != generation or
                hashlib.sha256(_read_regular_file(state_path)).hexdigest() != manifest.get("state_sha256")):
            return False
        if expected_algorithm is not None and manifest.get("algorithm") != expected_algorithm:
            return False
        config = manifest.get("config")
        if expected_config is not None and config != expected_config:
            if not allow_world_size_migration or not isinstance(config, dict):
                return False
            old_config, new_config = dict(config), dict(expected_config)
            old_size, new_size = old_config.pop("world_size", None), new_config.pop("world_size", None)
            if (old_config != new_config or isinstance(old_size, bool) or isinstance(new_size, bool) or
                    not isinstance(old_size, int) or old_size <= 0 or
                    not isinstance(new_size, int) or new_size <= 0):
                return False
        payload = torch.load(io.BytesIO(_read_regular_file(state_path)), map_location="cpu", weights_only=True)
        state = payload.get("state") if isinstance(payload, dict) else None
        cursor = state.get("episode_cursor") if isinstance(state, dict) else None
        if (not isinstance(state, dict) or state.get("generation") != generation or
                state.get("update") != generation or isinstance(cursor, bool) or
                not isinstance(cursor, int) or cursor < 0):
            return False
        batch_size = config.get("batch_episodes") if isinstance(config, dict) else None
        episode_budget = config.get("episodes") if isinstance(config, dict) else None
        if isinstance(batch_size, int) and batch_size > 0 and isinstance(episode_budget, int) and episode_budget > 0:
            expected_updates = (cursor + batch_size - 1) // batch_size
            if cursor > episode_budget or generation != expected_updates or (cursor % batch_size and cursor != episode_budget):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError):
        return False


def _reconcile_committed_orphans(root: Path, current_name: str, *,
                                 expected_algorithm: str | None = None,
                                 expected_config: dict | None = None,
                                 allow_world_size_migration: bool = False) -> str:
    """Adopt contiguous complete generations renamed before a crash before LATEST.

    The directory rename is the checkpoint commit point; LATEST is its index.
    A complete, hash-checked, weights-only-readable generation immediately
    following the pointer is therefore safe to adopt. Partial/staging files are
    ignored and cannot advance the pointer.
    """
    current = int(current_name.removeprefix("generation-"))
    adopted = current_name
    while True:
        next_name = f"generation-{current + 1:08d}"
        next_path = root / next_name
        if not next_path.exists() or not _is_complete_generation(
                next_path, expected_algorithm=expected_algorithm, expected_config=expected_config,
                allow_world_size_migration=allow_world_size_migration):
            break
        adopted = next_name
        current += 1
    return adopted


def resolve_generation(path: str | Path, *, expected_algorithm: str | None = None,
                       expected_config: dict | None = None,
                       allow_world_size_migration: bool = False) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("checkpoint paths may not be symlinks")
    if candidate.name == "LATEST" or candidate.is_dir() and not (candidate / "manifest.json").exists():
        root = candidate.parent if candidate.name == "LATEST" else candidate
        latest = root / "LATEST"
        if latest.is_symlink():
            raise ValueError("LATEST pointer may not be a symlink")
        if not latest.is_file():
            raise FileNotFoundError(f"no committed checkpoint pointer at {latest}")
        name = _read_regular_file(latest).decode("utf-8").strip()
        if not _GENERATION_NAME.fullmatch(name):
            raise ValueError("invalid LATEST pointer")
        name = _reconcile_committed_orphans(root, name, expected_algorithm=expected_algorithm,
                                            expected_config=expected_config,
                                            allow_world_size_migration=allow_world_size_migration)
        candidate = root / name
    if candidate.is_symlink():
        raise ValueError("checkpoint generation may not be a symlink")
    if not _GENERATION_NAME.fullmatch(candidate.name):
        raise ValueError("checkpoint generation path has an invalid name")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise
    if resolved.parent != candidate.parent.resolve(strict=True):
        raise ValueError("checkpoint generation resolves outside its checkpoint root")
    if not resolved.is_dir():
        raise ValueError("checkpoint generation must be a directory")
    return resolved


def load_generation(path: str | Path, *, model, optimizer=None, map_location="cpu",
                    expected_algorithm: str | None = None, expected_config: dict | None = None,
                    allow_world_size_migration: bool = False) -> dict[str, Any]:
    """Verify committed manifest/hash before restoring model/optimizer/cursor."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for resumable checkpoints") from exc
    candidate = Path(path)
    latest_root = (candidate.parent if candidate.name == "LATEST" else candidate
                   if candidate.is_dir() and not (candidate / "manifest.json").exists() else None)
    target = resolve_generation(path, expected_algorithm=expected_algorithm,
                                expected_config=expected_config,
                                allow_world_size_migration=allow_world_size_migration)
    manifest_path = target / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest missing: {manifest_path}")
    manifest = json.loads(_read_regular_file(manifest_path).decode("utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    if expected_algorithm is not None and manifest.get("algorithm") != expected_algorithm:
        raise ValueError("checkpoint algorithm mismatch")
    config = manifest.get("config")
    if expected_config is not None and config != expected_config:
        compatible_migration = False
        if allow_world_size_migration and isinstance(config, dict):
            old_comparable, new_comparable = dict(config), dict(expected_config)
            old_world_size = old_comparable.pop("world_size", None)
            new_world_size = new_comparable.pop("world_size", None)
            compatible_migration = (old_comparable == new_comparable and
                                    isinstance(old_world_size, int) and old_world_size > 0 and
                                    isinstance(new_world_size, int) and new_world_size > 0)
        if not compatible_migration:
            raise ValueError("checkpoint configuration mismatch")
    state_name = manifest.get("state_file")
    if not isinstance(state_name, str) or state_name != STATE_FILENAME or Path(state_name).name != state_name:
        raise ValueError("checkpoint state filename is invalid")
    state_file = target / state_name
    if state_file.is_symlink() or not state_file.is_file():
        raise ValueError("checkpoint state must be a regular file within its generation")
    try:
        state_bytes = _read_regular_file(state_file)
    except FileNotFoundError as exc:
        raise ValueError("checkpoint state file is missing") from exc
    if hashlib.sha256(state_bytes).hexdigest() != manifest.get("state_sha256"):
        raise ValueError("checkpoint state checksum mismatch")
    # Training checkpoints contain only tensors and primitive containers. Never
    # fall back to unrestricted pickle loading: integrity hashes do not make a
    # malicious checkpoint safe to unpickle.
    try:
        payload = torch.load(io.BytesIO(state_bytes), map_location=map_location, weights_only=True)
    except TypeError as exc:  # fail closed on Torch without safe weights-only load
        raise RuntimeError("safe weights_only checkpoint loading is required") from exc
    training_state = payload.get("state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint training state is invalid")
    update = training_state.get("update")
    cursor = training_state.get("episode_cursor")
    saved_generation = training_state.get("generation")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (update, cursor, saved_generation)):
        raise ValueError("checkpoint update cursor is invalid")
    if saved_generation != manifest.get("generation") or update != saved_generation:
        raise ValueError("checkpoint generation/update cursor mismatch")
    batch_size = config.get("batch_episodes") if isinstance(config, dict) else None
    episode_budget = config.get("episodes") if isinstance(config, dict) else None
    if isinstance(batch_size, int) and batch_size > 0 and isinstance(episode_budget, int) and episode_budget > 0:
        expected_updates = (cursor + batch_size - 1) // batch_size
        if cursor > episode_budget or update != expected_updates or (cursor % batch_size and cursor != episode_budget):
            raise ValueError("checkpoint cursor is not at a committed global-update boundary")
    model_target = model.module if hasattr(model, "module") else model
    model_target.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if latest_root is not None:
        latest_path = latest_root / "LATEST"
        current_name = _read_regular_file(latest_path).decode("utf-8").strip()
        if current_name != target.name:
            # Publish recovery only after the selected checkpoint has passed all
            # caller-specific validation and restored successfully.
            _atomic_bytes(latest_path, (target.name + "\n").encode())
    migration = None
    if expected_config is not None and config != expected_config:
        migration = {"field": "world_size", "from": config["world_size"],
                    "to": expected_config["world_size"], "committed_update": payload["state"].get("update")}
    return {"manifest": manifest, "execution_migration": migration, **payload}


def _assert_finite_tensors(value: Any, label: str) -> None:
    """Reject non-finite floating tensors before persisting optimizer/model state."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint validation") from exc
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise FloatingPointError(f"non-finite {label} tensor; refusing checkpoint commit")
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_tensors(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_tensors(item, label)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite {label} scalar; refusing checkpoint commit")
