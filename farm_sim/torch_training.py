"""Optional PyTorch trajectory learners for the synthetic Pasture Studies task.

The simulator and its sequential five-animal transition rules remain the source of
truth; this module only supplies decisions and computes policy gradients.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .model import Action, FEATURE_NAMES, PolicyDecision
from .policy import CategoricalPolicy

try:  # Optional dependency: importing farm_sim never requires torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on minimal installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

ACTIONS = tuple(Action)
N_INPUTS = len(FEATURE_NAMES)
N_ACTIONS = len(ACTIONS)
HIDDEN_SIZE = 32
ROLLOUT_TICKS = 72


class InvalidPolicyOutput(FloatingPointError):
    """Policy emitted an invalid action distribution; never silently repair it."""


def validate_probabilities(probabilities: Sequence[float], *, action_count: int = N_ACTIONS) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (action_count,):
        raise InvalidPolicyOutput(f"expected probability vector shape ({action_count},), got {values.shape}")
    if not np.isfinite(values).all():
        raise InvalidPolicyOutput("policy probabilities contain non-finite values")
    if np.any(values < 0.0):
        raise InvalidPolicyOutput("policy probabilities contain negative values")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise InvalidPolicyOutput(f"policy probability sum must be 1 (got {total})")
    return values


def assert_finite_training_state(loss, model, *, gradients: bool = False) -> None:
    """Fail closed before optimizer/checkpoint operations on any nonfinite value."""
    t = require_torch()
    if not t.isfinite(loss).all().item():
        raise FloatingPointError("non-finite training loss")
    for name, parameter in model.named_parameters():
        if not t.isfinite(parameter.detach()).all().item():
            raise FloatingPointError(f"non-finite model parameter: {name}")
        if gradients and parameter.grad is not None and not t.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"non-finite gradient: {name}")


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch training requires the optional dependency; install requirements-training.txt")
    return torch


def shard_episode_indices(start: int, count: int, world_size: int, rank: int) -> list[int]:
    """Deterministically shard a global batch into equal contiguous episode IDs."""
    if world_size <= 0 or not 0 <= rank < world_size or count % world_size:
        raise ValueError("episode batch must divide evenly across DDP ranks")
    local_count = count // world_size
    return list(range(start + rank * local_count, start + (rank + 1) * local_count))


def validate_equal_all_one_masks(masks: Sequence[np.ndarray]) -> int:
    """Validate the plan's complete-episode DDP contract; return samples/rank."""
    if not masks:
        raise ValueError("at least one rank mask is required")
    counts = []
    for mask in masks:
        values = np.asarray(mask)
        if values.size == 0 or not np.all(values == 1):
            raise ValueError("fixed DDP workload requires all-one complete episode masks")
        counts.append(int(values.sum()))
    if len(set(counts)) != 1:
        raise ValueError("DDP ranks must have equal valid sample counts")
    return counts[0]


def derive_seed(root: int, *parts: object) -> int:
    """Stable, order-independent 63-bit seed derivation (does not use hash())."""
    payload = "\x1f".join([str(int(root)), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def set_deterministic(seed: int, device: str = "cpu") -> None:
    t = require_torch()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    t.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(t.backends, "cudnn"):
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
    if hasattr(t.backends, "cuda") and hasattr(t.backends.cuda, "matmul"):
        t.backends.cuda.matmul.allow_tf32 = False
    if hasattr(t.backends, "cudnn"):
        t.backends.cudnn.allow_tf32 = False
    t.set_float32_matmul_precision("highest")


def resolve_device(requested: str = "auto") -> str:
    t = require_torch()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return "cuda" if requested == "auto" and t.cuda.is_available() else ("cpu" if requested == "auto" else requested)


class ActorCritic(nn.Module if nn is not None else object):
    """20→32 tanh→6 actor; optional scalar value head shares the hidden layer."""
    def __init__(self, *, value_head: bool = False, seed: int = 0, dtype=None):
        t = require_torch()
        super().__init__()
        dtype = dtype or t.float32
        # Preserve the NumPy initialization exactly (same NumPy seed and shapes).
        policy = CategoricalPolicy.initialize(seed)
        self.actor1 = nn.Linear(N_INPUTS, HIDDEN_SIZE, dtype=dtype)
        self.actor2 = nn.Linear(HIDDEN_SIZE, N_ACTIONS, dtype=dtype)
        with t.no_grad():
            self.actor1.weight.copy_(t.as_tensor(policy.w1.T, dtype=dtype))
            self.actor1.bias.copy_(t.as_tensor(policy.b1, dtype=dtype))
            self.actor2.weight.copy_(t.as_tensor(policy.w2.T, dtype=dtype))
            self.actor2.bias.copy_(t.as_tensor(policy.b2, dtype=dtype))
        self.value_head_enabled = bool(value_head)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1, dtype=dtype) if value_head else None
        if value_head:
            generator = np.random.default_rng(derive_seed(seed, "value-head"))
            with t.no_grad():
                self.value_head.weight.copy_(t.as_tensor(generator.normal(0, 1 / math.sqrt(HIDDEN_SIZE), (1, HIDDEN_SIZE)), dtype=dtype))
                self.value_head.bias.zero_()

    def forward(self, observations):
        t = require_torch()
        hidden = t.tanh(self.actor1(observations))
        logits = self.actor2(hidden)
        value = self.value_head(hidden).squeeze(-1) if self.value_head is not None else None
        return logits, value

    def probabilities(self, observations):
        return require_torch().softmax(self.forward(observations)[0], dim=-1)

    def decide(self, features: Sequence[float]) -> PolicyDecision:
        t = require_torch()
        with t.no_grad():
            x = t.as_tensor(np.asarray(features, dtype=np.float32), device=next(self.parameters()).device)
            probs = self.probabilities(x).detach().cpu().numpy().reshape(-1)
        probs = validate_probabilities(probs)
        idx = int(np.argmax(probs))
        return PolicyDecision(ACTIONS[idx], tuple(float(x) for x in probs), "torch-argmax")


@dataclass
class Trajectory:
    observations: np.ndarray  # [ticks, animals, features]
    actions: np.ndarray       # [ticks, animals]
    rewards: np.ndarray       # [ticks, animals]
    uniforms: np.ndarray      # [ticks, animals]
    seed: int
    probabilities: np.ndarray | None = None
    event_counts: dict[str, int] | None = None
    invalid_outputs: int = 0

    def returns_to_go(self) -> np.ndarray:
        # Per-animal Monte Carlo return-to-go; no bootstrap or shaping.
        out = np.zeros_like(self.rewards, dtype=np.float32)
        running = np.zeros(self.rewards.shape[1], dtype=np.float64)
        for tick in range(self.rewards.shape[0] - 1, -1, -1):
            running += self.rewards[tick]
            out[tick] = running
        return out


class _SamplingPolicy:
    """Simulation adapter using precomputed common uniforms and inverse CDF."""
    def __init__(self, model, uniforms: np.ndarray):
        self.model = model
        self.uniforms = uniforms
        self.index = 0
        self.probabilities_seen: list[np.ndarray] = []
        self.sampled: list[int] = []
        self.invalid_outputs = 0

    def decide(self, features: Sequence[float]) -> PolicyDecision:
        t = require_torch()
        if self.index >= self.uniforms.size:
            raise RuntimeError("uniform stream exhausted")
        with t.no_grad():
            inference_model = self.model.module if hasattr(self.model, "module") else self.model
            x = t.as_tensor(np.asarray(features, dtype=np.float32), device=next(inference_model.parameters()).device)
            p = inference_model.probabilities(x).detach().cpu().numpy().reshape(-1)
        try:
            p = validate_probabilities(p)
        except InvalidPolicyOutput:
            self.invalid_outputs = getattr(self, "invalid_outputs", 0) + 1
            raise
        u = float(self.uniforms.flat[self.index])
        idx = min(len(p) - 1, int(np.searchsorted(np.cumsum(p), u, side="right")))
        self.index += 1
        self.probabilities_seen.append(p)
        self.sampled.append(idx)
        return PolicyDecision(ACTIONS[idx], tuple(float(v) for v in p), "torch-sampled")


def collect_episodes_batch(model, episodes: Sequence[tuple[int, int]], *, root_seed: int,
                           ticks: int = ROLLOUT_TICKS,
                           perception_mode: str | None = None,
                           resource_profile: str = "starter") -> list[Trajectory]:
    """Collect independent worlds with batched policy inference and scalar commits.

    ``episodes`` contains ``(world_seed, episode_index)`` pairs. Observations are
    gathered from each world's pre-transition snapshot, concatenated only for the
    neural forward pass, then each world's decisions are committed independently
    through ``Simulation.step_with_decisions``. Counter-derived action uniforms
    make sampling independent of batch order and global RNG state.
    """
    from .simulation import Simulation, make_resource_rich_world, make_world
    if resource_profile not in {"starter", "resource-rich-v1"}:
        raise ValueError(f"unknown training resource profile {resource_profile!r}")
    if ticks != ROLLOUT_TICKS:
        raise ValueError("trajectory experiments require complete 72-tick episodes")
    if len({index for _, index in episodes}) != len(episodes):
        raise ValueError("episode indices must be unique within a batch")
    if not episodes:
        return []
    t = require_torch()
    inference_model = model.module if hasattr(model, "module") else model
    perception_mode = perception_mode or getattr(inference_model, "perception_mode", "legacy-v1")
    device = next(inference_model.parameters()).device
    worlds = []
    for world_seed, episode_index in episodes:
        uniforms = np.empty((ticks, 5), dtype=np.float64)
        for tick in range(ticks):
            for animal in range(5):
                rng = np.random.default_rng(derive_seed(root_seed, "action-uniform", world_seed, episode_index, tick, animal))
                uniforms[tick, animal] = rng.random()
        world = make_world(world_seed) if resource_profile == "starter" else make_resource_rich_world(world_seed)
        worlds.append({"seed": world_seed, "index": episode_index, "sim": Simulation(world, _SamplingPolicy(model, uniforms),
                                                                                      perception_mode=perception_mode),
                       "uniforms": uniforms, "observations": np.empty((ticks, 5, N_INPUTS), np.float32),
                       "actions": np.empty((ticks, 5), np.int64), "rewards": np.empty((ticks, 5), np.float32),
                       "probabilities": np.empty((ticks, 5, N_ACTIONS), np.float32)})
    for tick in range(ticks):
        rows = []
        feature_rows_by_world = []
        for w in worlds:
            features = w["sim"].decision_features()
            feature_rows_by_world.append(features)
            w["observations"][tick] = np.asarray([f for _, f in features], dtype=np.float32)
            rows.extend((w, animal_id, features_for_animal) for (animal_id, features_for_animal) in features)
        with t.no_grad():
            x = t.as_tensor(np.asarray([row[2] for row in rows], dtype=np.float32), device=device)
            probabilities = inference_model.probabilities(x).detach().cpu().numpy()
        for row, p in zip(rows, probabilities):
            validate_probabilities(p)
        cursor = 0
        for w, features_for_world in zip(worlds, feature_rows_by_world):
            decisions = {}
            for animal_position, (animal_id, features) in enumerate(features_for_world):
                p = probabilities[cursor]
                cursor += 1
                u = float(w["uniforms"][tick, animal_position])
                action_index = min(N_ACTIONS - 1, int(np.searchsorted(np.cumsum(p), u, side="right")))
                decisions[animal_id] = (PolicyDecision(ACTIONS[action_index], tuple(float(v) for v in p), "torch-sampled-batched"), features)
                w["probabilities"][tick, animal_position] = p
            w["sim"].step_with_decisions(decisions)
            for animal_position, animal in enumerate(sorted(w["sim"].world.animals, key=lambda a: a.animal_id)):
                w["actions"][tick, animal_position] = ACTIONS.index(animal.action)
                w["rewards"][tick, animal_position] = animal.last_reward
    result = []
    for w in worlds:
        event_counts: dict[str, int] = {}
        for event in w["sim"].events:
            event_counts[event.kind] = event_counts.get(event.kind, 0) + 1
        result.append(Trajectory(w["observations"], w["actions"], w["rewards"], w["uniforms"], w["seed"],
                                 w["probabilities"], event_counts, 0))
    return result


def collect_episode(model, world_seed: int, *, root_seed: int, episode_index: int,
                    ticks: int = ROLLOUT_TICKS) -> Trajectory:
    """Compatibility wrapper around the batched environment collector."""
    return collect_episodes_batch(model, [(world_seed, episode_index)], root_seed=root_seed, ticks=ticks)[0]


def calculate_loss(model, trajectories: Sequence[Trajectory], algorithm: str,
                   *, previous_return_sum: float = 0.0, previous_return_count: int = 0):
    """Return actor/critic loss tensors and auditable scalars for one rollout batch."""
    t = require_torch()
    if algorithm not in {"reinforce", "a2c"}:
        raise ValueError("algorithm must be reinforce or a2c")
    if not trajectories:
        raise ValueError("training batch must contain at least one complete trajectory")
    observations = np.concatenate([x.observations.reshape(-1, N_INPUTS) for x in trajectories])
    actions = np.concatenate([x.actions.reshape(-1) for x in trajectories])
    returns = np.concatenate([x.returns_to_go().reshape(-1) for x in trajectories])
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
        actor_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * ((values - g) ** 2).mean()
    total_loss = actor_loss + value_loss
    if not t.isfinite(total_loss).all().item():
        raise FloatingPointError("non-finite actor/critic loss")
    return total_loss, {
        "actor_loss": float(actor_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()), "mean_return_to_go": float(g.mean().detach().cpu()),
        "return_sum": float(g.sum().detach().cpu()), "return_count": int(g.numel()),
        "mean_value": float(values.mean().detach().cpu()) if values is not None else 0.0,
        "mean_advantage": float(advantages.mean().detach().cpu()),
        "baseline_used": float(previous_return_sum / previous_return_count if previous_return_count else 0.0) if algorithm == "reinforce" else None,
    }


def export_actor_npz(model, path, *, seed: int, episodes: int, metadata: dict[str, Any] | None = None):
    """Convert trained actor to existing CategoricalPolicy NPZ + JSON format."""
    from .policy import save_checkpoint
    t = require_torch()
    with t.no_grad():
        policy = CategoricalPolicy(
            model.actor1.weight.detach().cpu().numpy().T.astype(np.float64),
            model.actor1.bias.detach().cpu().numpy().astype(np.float64),
            model.actor2.weight.detach().cpu().numpy().T.astype(np.float64),
            model.actor2.bias.detach().cpu().numpy().astype(np.float64),
            seed=seed, episodes=episodes, policy_id=f"torch-{seed}-{episodes}",
        )
    return save_checkpoint(policy, path, metadata=metadata or {})


def intervention_probabilities(model, observation: Sequence[float], *, device=None) -> dict[str, Any]:
    """First-decision memory/perception counterfactual probabilities, same model."""
    t = require_torch()
    base = np.asarray(observation, dtype=np.float32).copy()
    if base.shape != (N_INPUTS,):
        raise ValueError("expected one 20-feature observation")
    cases = {
        "full": base,
        "memory_off": base.copy(),
        "water_perception_off": base.copy(),
    }
    cases["memory_off"][12:17] = 0.0
    cases["water_perception_off"][5] = 0.0
    cases["water_perception_off"][9] = 0.0
    with t.no_grad():
        out = {}
        for name, row in cases.items():
            x = t.as_tensor(row, dtype=t.float32, device=device or next(model.parameters()).device)
            out[name] = model.probabilities(x).detach().cpu().numpy().reshape(-1).tolist()
    return {"probabilities": out, "actions": {k: ACTIONS[int(np.argmax(v))].value for k, v in out.items()},
            "memory_delta_l1": float(np.abs(np.asarray(out["full"]) - out["memory_off"]).sum()),
            "perception_delta_l1": float(np.abs(np.asarray(out["full"]) - out["water_perception_off"]).sum())}
