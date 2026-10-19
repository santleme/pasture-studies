"""Optional Gymnasium adapter for training against the existing farm simulator.

This module keeps Gymnasium out of the simulator's runtime dependency set. A
single-agent action controls cow-1 while the other cattle walk; joint mode lets
one PPO policy choose an action for all five cattle each tick.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - depends on optional training extra
    raise ImportError("Gymnasium environments require the optional RL dependencies; install requirements-rl.txt") from exc

from .model import Action, FEATURE_NAMES, PolicyDecision
from .simulation import Simulation, make_world

_ACTIONS = tuple(Action)
_ANIMAL_IDS = tuple(f"cow-{index}" for index in range(1, 6))


class _WalkPolicy:
    """Valid policy injected only to satisfy Simulation's required constructor."""

    def decide(self, features):
        probabilities = tuple(1.0 if action is Action.WALK else 0.0 for action in _ACTIONS)
        return PolicyDecision(Action.WALK, probabilities, "gymnasium-fixed-walk")


class PastureEnv(gym.Env):
    """Finite-horizon Gymnasium environment backed by the canonical simulator.

    ``joint=False`` trains one focused animal (cow-1); ``joint=True`` trains a
    shared policy on the five animals with a ``MultiDiscrete([6] * 5)`` action.
    Rewards are unmodified simulator rewards (focused cow or five-cow mean).
    Reaching the artificial rollout horizon is truncation, not task termination.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, joint: bool = False, horizon: int = 72):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.joint = bool(joint)
        self.horizon = int(horizon)
        self.action_space = (spaces.MultiDiscrete(np.full(5, len(_ACTIONS), dtype=np.int64))
                             if self.joint else spaces.Discrete(len(_ACTIONS)))
        width = len(_ANIMAL_IDS) if self.joint else 1
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(width * len(FEATURE_NAMES),), dtype=np.float32)
        self.simulation: Simulation | None = None
        self.elapsed_steps = 0

    def _observation(self) -> np.ndarray:
        assert self.simulation is not None
        features = dict(self.simulation.decision_features())
        ids = _ANIMAL_IDS if self.joint else _ANIMAL_IDS[:1]
        return np.asarray([value for animal_id in ids for value in features[animal_id]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self.simulation = Simulation(make_world(seed=0 if seed is None else int(seed)), _WalkPolicy())
        self.elapsed_steps = 0
        return self._observation(), {"tick": self.simulation.world.tick, "animals": len(_ANIMAL_IDS)}

    def step(self, action):
        if self.simulation is None:
            raise RuntimeError("reset must be called before step")
        values = np.asarray(action)
        expected_shape = (5,) if self.joint else ()
        if values.shape != expected_shape:
            raise ValueError(f"action must have shape {expected_shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("action values must be integers")
        if np.any(values < 0) or np.any(values >= len(_ACTIONS)):
            raise ValueError("action values must be valid Action indices")

        selected = values.astype(np.int64, copy=False).reshape(5) if self.joint else np.full(5, int(values), dtype=np.int64)
        if not self.joint:
            selected[1:] = tuple(_ACTIONS).index(Action.WALK)
        features = dict(self.simulation.decision_features())
        decisions = {}
        for animal_id, action_index in zip(_ANIMAL_IDS, selected):
            chosen = _ACTIONS[int(action_index)]
            probabilities = tuple(1.0 if candidate is chosen else 0.0 for candidate in _ACTIONS)
            decisions[animal_id] = (PolicyDecision(chosen, probabilities, "gymnasium-action"), features[animal_id])
        result = self.simulation.step_with_decisions(decisions)
        self.elapsed_steps += 1
        rewards = {animal.animal_id: float(animal.last_reward) for animal in result.world.animals}
        reward = rewards["cow-1"] if not self.joint else float(np.mean(tuple(rewards.values())))
        truncated = self.elapsed_steps >= self.horizon
        return self._observation(), reward, False, truncated, {
            "tick": result.world.tick, "individual_rewards": rewards,
            "actions": {animal_id: _ACTIONS[int(index)].value for animal_id, index in zip(_ANIMAL_IDS, selected)},
        }
