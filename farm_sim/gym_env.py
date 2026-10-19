"""Optional Gymnasium adapter for the existing five-animal Pasture simulator.

This keeps the simulator's transition and reward definitions authoritative. A
one-animal environment controls the first cow while the other four take REST;
the five-animal environment exposes a joint MultiDiscrete action and mean reward.
Install the optional dependencies from ``requirements-rl.txt`` to use it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Gymnasium adapter requires gymnasium; install requirements-rl.txt") from exc

from .model import Action, PolicyDecision
from .simulation import ANIMAL_COUNT, Simulation, make_world


class _UnusedPolicy:
    """Simulation requires a policy even though this adapter supplies decisions."""

    def decide(self, features):  # pragma: no cover - step_with_decisions bypasses it
        raise AssertionError("Gymnasium adapter must always provide precomputed decisions")


class PastureEnv(gym.Env):
    """Gymnasium environment for one controlled animal or the five-cow joint task.

    Args:
        controlled_animals: ``1`` for a single cow with four REST background cows,
            or ``5`` for a joint action controlling the complete simulated herd.
        horizon: Maximum number of committed simulator ticks per episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, controlled_animals: int = 1, horizon: int = 72):
        super().__init__()
        if controlled_animals not in (1, ANIMAL_COUNT):
            raise ValueError(f"controlled_animals must be 1 or {ANIMAL_COUNT}")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.controlled_animals = controlled_animals
        self.horizon = horizon
        self.action_space = (spaces.Discrete(len(Action)) if controlled_animals == 1
                             else spaces.MultiDiscrete([len(Action)] * ANIMAL_COUNT))
        shape = (20,) if controlled_animals == 1 else (ANIMAL_COUNT, 20)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self._simulation: Simulation | None = None
        self._steps = 0

    def _observations(self) -> np.ndarray:
        assert self._simulation is not None
        rows = self._simulation.decision_features()
        values = np.asarray([features for _, features in rows[:self.controlled_animals]], dtype=np.float32)
        return values[0] if self.controlled_animals == 1 else values

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        seed = 7 if seed is None else int(seed)
        self._simulation = Simulation(make_world(seed), _UnusedPolicy())
        self._steps = 0
        info = {"seed": seed, "controlled_animal_ids": [a.animal_id for a in self._simulation.world.animals[:self.controlled_animals]]}
        return self._observations(), info

    def step(self, action):
        if self._simulation is None:
            raise RuntimeError("call reset() before step()")
        # Validate the complete action before constructing or committing a tick.
        if not self.action_space.contains(action):
            raise ValueError("action is outside the environment action space")
        indices = [int(action)] if self.controlled_animals == 1 else [int(a) for a in action]
        rows = self._simulation.decision_features()
        decisions = {}
        action_values = []
        rest_index = tuple(Action).index(Action.REST)
        for animal_index, (animal_id, features) in enumerate(rows):
            action_index = indices[animal_index] if animal_index < self.controlled_animals else rest_index
            selected = tuple(Action)[action_index]
            probabilities = tuple(1.0 if candidate == selected else 0.0 for candidate in Action)
            decisions[animal_id] = (PolicyDecision(selected, probabilities, "gymnasium-action"), features)
            if animal_index < self.controlled_animals:
                action_values.append(selected.value)

        result = self._simulation.step_with_decisions(decisions)
        rewards = [animal.last_reward for animal in result.world.animals[:self.controlled_animals]]
        reward = float(rewards[0] if self.controlled_animals == 1 else np.mean(rewards))
        self._steps += 1
        terminated = False
        truncated = self._steps >= self.horizon
        info = {
            "tick": result.world.tick,
            "individual_rewards": rewards,
            "actions": action_values,
            "controlled_animal_ids": [a.animal_id for a in result.world.animals[:self.controlled_animals]],
        }
        return self._observations(), reward, terminated, truncated, info

    def close(self):
        self._simulation = None

