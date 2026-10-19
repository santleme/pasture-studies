"""Optional PettingZoo Parallel adapter for the five-animal pasture pilot.

All agents submit actions against the same pre-transition observation snapshot.
The underlying simulator then resolves movement and resources sequentially in
stable animal-ID order. This is a joint-action interface, not simultaneous
physics or order-independent arbitration.
"""
from __future__ import annotations

from operator import index
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from farm_sim.model import Action, PolicyDecision
from farm_sim.simulation import Simulation, make_world

ACTION_LIST = tuple(Action)
AGENT_IDS = tuple(f"cow-{i}" for i in range(1, 6))
OBSERVATION_SIZE = 20


class _UnusedPolicy:
    """Simulation requires a policy; the adapter always supplies batch decisions."""

    def decide(self, features):
        raise AssertionError("Parallel adapter must use step_with_decisions")


class PastureParallelEnv(ParallelEnv):
    """Five-animal joint-action interface with explicit ordered commit semantics."""

    metadata = {"name": "pasture_parallel_v0", "render_modes": []}

    def __init__(self, *, seed: int = 0, max_cycles: int = 72):
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        self.possible_agents = list(AGENT_IDS)
        self.agents: list[str] = []
        self.observation_spaces = {
            agent: spaces.Box(0.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(len(ACTION_LIST)) for agent in self.possible_agents
        }
        self.max_cycles = int(max_cycles)
        self.seed_value = int(seed)
        self.sim: Simulation | None = None
        self.reset(seed=self.seed_value)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _observations(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        rows = self.sim.decision_features()
        return {
            agent_id: np.asarray(features, dtype=np.float32)
            for agent_id, features in rows
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options  # The fixed starter world has no reset options.
        if seed is not None:
            self.seed_value = int(seed)
        self.sim = Simulation(make_world(self.seed_value), _UnusedPolicy())
        self.agents = self.possible_agents.copy()
        observations = self._observations()
        infos = {agent: {"seed": self.seed_value, "tick": 0} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step called after episode completion; call reset first")
        if not isinstance(actions, dict) or set(actions) != set(self.agents):
            raise ValueError("actions must contain exactly one action for every live agent")
        assert self.sim is not None

        # Freeze every observation before constructing decisions. Simulation then
        # commits those decisions by stable animal ID, arbitrating shared resources.
        features_by_id = dict(self.sim.decision_features())
        decisions = {}
        for agent in self.agents:
            raw_action = actions[agent]
            if isinstance(raw_action, (bool, np.bool_)):
                raise ValueError(f"invalid discrete action for {agent}: {raw_action!r}")
            try:
                action_index = index(raw_action)
            except TypeError as exc:
                raise ValueError(f"invalid discrete action for {agent}: {raw_action!r}") from exc
            if not 0 <= action_index < len(ACTION_LIST):
                raise ValueError(f"action index out of range for {agent}: {action_index}")
            probabilities = tuple(float(i == action_index) for i in range(len(ACTION_LIST)))
            decisions[agent] = (
                PolicyDecision(ACTION_LIST[action_index], probabilities, "pettingzoo-joint"),
                features_by_id[agent],
            )

        before_tick = self.sim.world.tick
        result = self.sim.step_with_decisions(decisions)
        world = result.world
        observations = self._observations()
        animals = {animal.animal_id: animal for animal in world.animals}
        rewards = {agent: float(animals[agent].last_reward) for agent in self.agents}
        episode_done = world.tick >= self.max_cycles
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: episode_done for agent in self.agents}
        commit_order = sorted(animals)
        infos = {
            agent: {
                "tick": world.tick,
                "observation_snapshot_tick": before_tick,
                "action": animals[agent].action.value,
                "commit_order": commit_order,
                "transition_semantics": "joint action collection; stable-ID sequential commit",
            }
            for agent in self.agents
        }
        if episode_done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


def main() -> None:
    from pettingzoo.test import parallel_api_test

    env = PastureParallelEnv(max_cycles=8)
    parallel_api_test(env, num_cycles=10)
    print(f"PettingZoo Parallel API smoke passed (PettingZoo {__import__('pettingzoo').__version__}, Gymnasium {gymnasium.__version__}).")


if __name__ == "__main__":
    main()
