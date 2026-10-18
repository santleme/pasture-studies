"""Small local categorical policy and strict versioned checkpoint handling."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from .model import Action, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, PolicyDecision

CHECKPOINT_VERSION = 1
HIDDEN_SIZE = 32


class CheckpointError(ValueError):
    """A policy checkpoint is missing, corrupt, or incompatible with this runtime."""


@dataclass
class CategoricalPolicy:
    """One-hidden-layer tanh MLP; inference is local and side-effect free."""
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    seed: int = 0
    episodes: int = 0
    policy_id: str = "untrained"

    @classmethod
    def initialize(cls, seed: int, hidden_size: int = HIDDEN_SIZE) -> "CategoricalPolicy":
        rng = np.random.default_rng(seed)
        n_in, n_out = len(FEATURE_NAMES), len(Action)
        return cls(rng.normal(0, np.sqrt(1 / n_in), (n_in, hidden_size)),
                   np.zeros(hidden_size),
                   rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, n_out)),
                   np.zeros(n_out), seed=seed, policy_id=f"numpy-mlp-seed-{seed}")

    @property
    def hidden_size(self) -> int:
        return int(self.b1.shape[0])

    def logits(self, features: Sequence[float]) -> np.ndarray:
        x = _validate_features(features)
        if self.w1.shape != (len(FEATURE_NAMES), self.hidden_size) or self.w2.shape != (self.hidden_size, len(Action)):
            raise ValueError("policy weight dimensions do not match the feature/action schema")
        return np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2

    def probabilities(self, features: Sequence[float]) -> np.ndarray:
        z = self.logits(features)
        z = z - np.max(z)
        exp = np.exp(z)
        return exp / exp.sum()

    def decide(self, features: Sequence[float]) -> PolicyDecision:
        probs = self.probabilities(features)
        index = int(np.argmax(probs))
        return PolicyDecision(tuple(Action)[index], tuple(float(p) for p in probs), self.policy_id)

    def sample(self, features: Sequence[float], rng: np.random.Generator) -> Action:
        return tuple(Action)[int(rng.choice(len(Action), p=self.probabilities(features)))]

    def copy(self) -> "CategoricalPolicy":
        return CategoricalPolicy(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(),
                                 self.seed, self.episodes, self.policy_id)

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate((self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()))


def _validate_features(features: Sequence[float]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features in FEATURE_NAMES order; got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("features must all be finite")
    return x


def save_checkpoint(policy: CategoricalPolicy, path: str | Path, *, metadata: dict | None = None) -> Path:
    """Atomically save NPZ parameters and JSON schema/metadata sidecar."""
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError("checkpoint path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix(".json")
    doc = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "actions": [a.value for a in Action],
        "dimensions": {"input": len(FEATURE_NAMES), "hidden": policy.hidden_size, "output": len(Action)},
        "seed": policy.seed, "episodes": policy.episodes, "policy_id": policy.policy_id,
        "metadata": metadata or {},
    }
    _validate_metadata(doc)
    _atomic_npz(target, policy)
    _atomic_json(sidecar, doc)
    return target


def load_checkpoint(path: str | Path) -> CategoricalPolicy:
    target = Path(path)
    sidecar = target.with_suffix(".json")
    if not target.is_file() or not sidecar.is_file():
        raise CheckpointError(f"trained policy checkpoint missing at {target}; run `python3 -m farm_sim.training --output {target.parent}` first")
    try:
        doc = json.loads(sidecar.read_text())
        _validate_metadata(doc)
        with np.load(target, allow_pickle=False) as weights:
            arrays = [np.array(weights[k], dtype=np.float64) for k in ("w1", "b1", "w2", "b2")]
        policy = CategoricalPolicy(*arrays, seed=int(doc["seed"]), episodes=int(doc["episodes"]), policy_id=str(doc["policy_id"]))
        dims = doc["dimensions"]
        expected = ((dims["input"], dims["hidden"]), (dims["hidden"],),
                    (dims["hidden"], dims["output"]), (dims["output"],))
        if any(a.shape != shape for a, shape in zip(arrays, expected)) or not all(np.isfinite(a).all() for a in arrays):
            raise CheckpointError("checkpoint weight shapes or values do not match metadata")
        return policy
    except CheckpointError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid policy checkpoint {target}: {exc}") from exc


def _validate_metadata(doc: dict) -> None:
    if doc.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointError("unsupported checkpoint version")
    if doc.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or doc.get("feature_names") != list(FEATURE_NAMES):
        raise CheckpointError("checkpoint feature schema does not match FEATURE_NAMES")
    if doc.get("actions") != [a.value for a in Action]:
        raise CheckpointError("checkpoint action schema does not match Action enum")
    dims = doc.get("dimensions", {})
    if dims.get("input") != len(FEATURE_NAMES) or dims.get("output") != len(Action) or not isinstance(dims.get("hidden"), int) or dims["hidden"] < 1:
        raise CheckpointError("checkpoint dimensions are incompatible")


def _atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _atomic_npz(path: Path, policy: CategoricalPolicy) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as stream:
            np.savez_compressed(stream, w1=policy.w1, b1=policy.b1, w2=policy.w2, b2=policy.b2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


class HeuristicBaseline:
    """Explicitly opt-in deterministic experiment baseline; never a fallback."""
    policy_id = "explicit-heuristic-baseline"

    def decide(self, features: Sequence[float]) -> PolicyDecision:
        x = _validate_features(features)
        idx = FEATURE_NAMES.index
        if x[idx("thirst")] > 0.65 and x[idx("water_visible")]: action = Action.DRINK
        elif x[idx("hunger")] > 0.65 and x[idx("forage_visible")]: action = Action.GRAZE
        elif x[idx("fatigue")] > 0.7: action = Action.REST
        elif x[idx("social_need")] > 0.7: action = Action.HERD_INTERACT
        else: action = Action.WALK
        probs = tuple(1.0 if a == action else 0.0 for a in Action)
        return PolicyDecision(action, probs, self.policy_id)
