"""Optional experimental reward and A2C objective helpers.

These helpers are deliberately separate from the simulator reward and the
production trainer. Callers must opt in; defaults preserve their input values.
All reward shaping uses the simulator's normalized toy features, not biology.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import Action, FEATURE_NAMES


_ACTION_INDEX = {action: index for index, action in enumerate(Action)}


def state_aware_rewards(observations, actions, rewards, *, scale: float = 0.1) -> np.ndarray:
    """Return copied rewards with small drive/action alignment shaping.

    Resource intents only receive shaping when that resource is currently marked
    available. REST is discouraged at low fatigue; social interaction uses the
    encoded herd-count feature. Inputs are never mutated.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    obs = np.asarray(observations, dtype=np.float32)
    acts = np.asarray(actions)
    base = np.asarray(rewards)
    if obs.shape[-1:] != (len(FEATURE_NAMES),):
        raise ValueError(f"observations must end in {len(FEATURE_NAMES)} features")
    if acts.shape != obs.shape[:-1] or base.shape != acts.shape:
        raise ValueError("actions and rewards must match observation leading dimensions")
    if not np.isfinite(obs).all() or not np.isfinite(base).all():
        raise ValueError("observations and rewards must be finite")
    if np.any(acts < 0) or np.any(acts >= len(Action)):
        raise ValueError("actions must be valid Action indices")

    result = base.astype(np.result_type(base.dtype, np.float32), copy=True)
    for action, need_name, availability_name in (
        (Action.GRAZE, "hunger", "forage_available"),
        (Action.DRINK, "thirst", "water_available"),
        (Action.REST, "fatigue", None),
        (Action.HERD_INTERACT, "social_need", "herd_count_norm"),
    ):
        mask = acts == _ACTION_INDEX[action]
        if availability_name is not None:
            available = obs[..., FEATURE_NAMES.index(availability_name)] > 0.5
            mask &= available
        need = obs[..., FEATURE_NAMES.index(need_name)]
        if action == Action.HERD_INTERACT:
            need = need * obs[..., FEATURE_NAMES.index("herd_count_norm")]
        result[mask] += scale * (need[mask] - 0.5)
    return result


def normalize_advantages(advantages: Any, *, enabled: bool = False, eps: float = 1e-8):
    """Optionally center/scale a batch of advantages, supporting NumPy or Torch."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not enabled:
        return advantages
    if hasattr(advantages, "detach"):
        centered = advantages - advantages.mean()
        return centered / advantages.std(unbiased=False).clamp_min(eps)
    values = np.asarray(advantages)
    if not np.isfinite(values).all():
        raise ValueError("advantages must be finite")
    return (values - values.mean()) / max(float(values.std()), eps)


def entropy_regularized_actor_loss(actor_loss: Any, entropy: Any, *, coefficient: float = 0.0):
    """Return ``actor_loss - coefficient * entropy`` without mutating either input."""
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    return actor_loss - coefficient * entropy
