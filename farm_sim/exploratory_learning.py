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


def calculate_exploratory_loss(
    model,
    trajectories,
    algorithm: str,
    *,
    reward_variant: str = "original",
    state_reward_scale: float = 0.1,
    entropy_coefficient: float = 0.0,
    normalize_advantage: bool = False,
    previous_return_sum: float = 0.0,
    previous_return_count: int = 0,
):
    """Experimental loss path; its default exactly mirrors ``calculate_loss``.

    This stays separate from the production trainer so callers must explicitly
    select this helper to enable shaping, advantage normalization, or entropy.
    """
    from .torch_training import N_ACTIONS, N_INPUTS, InvalidPolicyOutput, require_torch

    t = require_torch()
    if algorithm not in {"reinforce", "a2c"}:
        raise ValueError("algorithm must be reinforce or a2c")
    if reward_variant not in {"original", "state_aware"}:
        raise ValueError("reward_variant must be original or state_aware")
    if not math.isfinite(state_reward_scale) or state_reward_scale < 0:
        raise ValueError("state_reward_scale must be finite and nonnegative")
    if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
        raise ValueError("entropy_coefficient must be finite and nonnegative")
    if not trajectories:
        raise ValueError("training batch must contain at least one complete trajectory")

    observations = np.concatenate([x.observations.reshape(-1, N_INPUTS) for x in trajectories])
    actions = np.concatenate([x.actions.reshape(-1) for x in trajectories])
    return_chunks = []
    for trajectory in trajectories:
        rewards = trajectory.rewards
        if reward_variant == "state_aware":
            rewards = state_aware_rewards(trajectory.observations, trajectory.actions, rewards,
                                          scale=state_reward_scale)
        if reward_variant == "original":
            return_chunks.append(trajectory.returns_to_go().reshape(-1))
        else:
            returns = np.zeros_like(rewards, dtype=np.float32)
            running = np.zeros(rewards.shape[1], dtype=np.float64)
            for tick in range(rewards.shape[0] - 1, -1, -1):
                running += rewards[tick]
                returns[tick] = running
            return_chunks.append(returns.reshape(-1))
    returns = np.concatenate(return_chunks)
    for name, values in (("observations", observations), ("returns", returns)):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"trajectory {name} contain non-finite values")
    if actions.size == 0 or np.any(actions < 0) or np.any(actions >= N_ACTIONS):
        raise ValueError("trajectory actions must be valid action indices")

    x = t.as_tensor(observations, dtype=t.float32, device=next(model.parameters()).device)
    a = t.as_tensor(actions, dtype=t.long, device=x.device)
    g = t.as_tensor(returns, dtype=t.float32, device=x.device)
    logits, values = model(x)
    if logits.ndim != 2 or logits.shape != (len(actions), N_ACTIONS):
        raise ValueError("model logits must have shape (decision_count, action_count)")
    if not t.isfinite(logits).all().item() or (values is not None and not t.isfinite(values).all().item()):
        raise FloatingPointError("non-finite actor logits or critic values")
    if t.any(a < 0).item() or t.any(a >= N_ACTIONS).item():
        raise ValueError("trajectory contains an invalid action index")
    probabilities = t.softmax(logits, dim=-1)
    if not t.isfinite(probabilities).all().item() or t.any(probabilities < 0).item() or not t.allclose(
            probabilities.sum(dim=-1), t.ones(len(actions), device=x.device), rtol=0.0, atol=1e-6):
        raise InvalidPolicyOutput("actor emitted invalid action probabilities")
    log_probs = t.log_softmax(logits, dim=-1).gather(1, a[:, None]).squeeze(1)
    entropy = -(t.softmax(logits, -1) * t.log_softmax(logits, -1)).sum(-1).mean()
    value_loss = t.zeros((), dtype=t.float32, device=x.device)
    if algorithm == "reinforce":
        baseline = previous_return_sum / previous_return_count if previous_return_count else 0.0
        advantages = g - float(baseline)
        actor_loss = -(log_probs * advantages.detach()).mean()
    else:
        if values is None:
            raise ValueError("A2C model must include value head")
        advantages = g - values
        if normalize_advantage:
            advantages = normalize_advantages(advantages, enabled=True)
        actor_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * ((values - g) ** 2).mean()
    total_loss = entropy_regularized_actor_loss(actor_loss + value_loss, entropy,
                                                coefficient=entropy_coefficient)
    if not t.isfinite(total_loss).all().item():
        raise FloatingPointError("non-finite actor/critic loss")
    return total_loss, {
        "actor_loss": float(actor_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()), "mean_return_to_go": float(g.mean().detach().cpu()),
        "return_sum": float(g.sum().detach().cpu()), "return_count": int(g.numel()),
        "mean_value": float(values.mean().detach().cpu()) if values is not None else 0.0,
        "mean_advantage": float(advantages.mean().detach().cpu()),
        "baseline_used": float(previous_return_sum / previous_return_count if previous_return_count else 0.0)
        if algorithm == "reinforce" else None,
    }
