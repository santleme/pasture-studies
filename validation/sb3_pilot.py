"""Low-budget SB3/Gymnasium baseline over the existing Pasture simulator.

The single-agent pilot trains cow-1 while four transparent need-controller
animals act as scripted context. The five-agent pilot is a centralized
MultiDiscrete controller: each per-cow action is chosen from the same
pre-transition feature snapshot and committed through Simulation's existing
arbitration. No production simulator code is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from farm_sim.model import Action, PolicyDecision
from farm_sim.simulation import Simulation, make_world
from validation.viability_v2 import NeedController, marginal_contact_reward
from farm_sim.execution_policy import require_kaggle_for_full_workload

EPISODE_TICKS = 72
ACTION_LIST = tuple(Action)


def _decision(index: int, policy_id: str) -> PolicyDecision:
    action = ACTION_LIST[int(index)]
    return PolicyDecision(action, tuple(float(i == int(index)) for i in range(len(ACTION_LIST))), policy_id)


class _OneControlledAgentPolicy:
    def __init__(self, action: int):
        self.action = action
        self.context = "need_controller"
        self.oracle = NeedController()
        self.decisions = 0

    def decide(self, features):
        # Stable ID order guarantees cow-1 is the first query of each tick.
        if self.decisions % 5 == 0:
            result = _decision(self.action, "sb3-cow-1")
        elif self.context == "rest":
            result = _decision(ACTION_LIST.index(Action.REST), "sb3-context-rest")
        else:
            result = self.oracle.decide(features)
        self.decisions += 1
        return result


class _JointActionPolicy:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        self.cursor = 0

    def decide(self, features):
        result = _decision(self.actions[self.cursor], "sb3-joint")
        self.cursor += 1
        return result


class PastureSB3Env(gym.Env):
    """SB3 wrapper: one trained cow or one centralized joint 5-cow policy."""
    metadata = {"render_modes": []}

    def __init__(self, *, agents: int = 1, regrowth_ticks: int = 3, seed: int = 0):
        if agents not in (1, 5):
            raise ValueError("agents must be 1 or 5")
        self.agents = agents
        self.regrowth_ticks = regrowth_ticks
        self.seed_value = int(seed)
        self.action_space = spaces.Discrete(len(ACTION_LIST)) if agents == 1 else spaces.MultiDiscrete([len(ACTION_LIST)] * 5)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(20,) if agents == 1 else (100,), dtype=np.float32)
        self.sim: Simulation | None = None
        self.policy = None
        self.regrow_at: dict[tuple[str, int], int] = {}
        self.behavior = Counter()
        self.agent_behavior: dict[str, Counter] = {}
        self.action_counts = Counter()
        self.agent_action_counts: dict[str, Counter] = {}
        self.reset(seed=seed)

    def _observation(self):
        assert self.sim is not None
        rows = [np.asarray(features, dtype=np.float32) for _, features in self.sim.decision_features()]
        obs = rows[0] if self.agents == 1 else np.concatenate(rows)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)
        self.sim = Simulation(make_world(self.seed_value), _OneControlledAgentPolicy(0))
        self.policy = self.sim.policy
        self.regrow_at.clear()
        self.behavior = Counter()
        self.agent_behavior = {animal.animal_id: Counter() for animal in self.sim.world.animals}
        self.action_counts = Counter()
        self.agent_action_counts = {animal.animal_id: Counter() for animal in self.sim.world.animals}
        return self._observation(), {"seed": self.seed_value}

    def step(self, action):
        assert self.sim is not None
        tick = self.sim.world.tick
        # Regrowth changes only this validation wrapper's scenario, not production code.
        for kind in ("forage", "water"):
            patches = getattr(self.sim.world, kind)
            refreshed = []
            for i, patch in enumerate(patches):
                deadline = self.regrow_at.get((kind, i))
                if not patch.available and deadline is not None and deadline <= tick:
                    patch = replace(patch, available=True)
                refreshed.append(patch)
            self.sim.world = replace(self.sim.world, **{kind: tuple(refreshed)})

        if self.agents == 1:
            self.policy.action = int(action)
        else:
            actions = np.asarray(action, dtype=np.int64).reshape(5)
            self.sim.policy = _JointActionPolicy(actions)
        before = self.sim.world
        result = self.sim.step()
        after = self.sim.world
        # Local causal correction for the demonstrated no-progress HERD_INTERACT
        # bonus. It is isolated to this adapter; legacy physiology is untouched.
        before_by_id = {a.animal_id: a for a in before.animals}
        corrected = []
        for animal in after.animals:
            previous = before_by_id[animal.animal_id].physiology.social_need
            current = animal.physiology.social_need
            reward = marginal_contact_reward(animal.last_reward, previous, current) \
                if animal.action == Action.HERD_INTERACT else animal.last_reward
            corrected.append(replace(animal, last_reward=reward))
        self.sim.world = replace(after, animals=tuple(corrected))
        after = self.sim.world
        if self.agents == 5:
            self.sim.policy.cursor = 0
            self.sim.policy = self.policy
        for kind in ("forage", "water"):
            for i, (old, new) in enumerate(zip(getattr(before, kind), getattr(after, kind))):
                if old.available and not new.available:
                    self.regrow_at[(kind, i)] = tick + self.regrowth_ticks + 1
        relevant = [event for event in result.events if event.kind in {"forage_success", "water_success", "rested", "ruminated", "herd_contact"}]
        self.behavior.update(event.kind for event in relevant)
        for event in relevant:
            self.agent_behavior[event.animal_id][event.kind] += 1
        self.action_counts.update(a.action.value for a in after.animals)
        for animal in after.animals:
            self.agent_action_counts[animal.animal_id][animal.action.value] += 1
        cow = min(after.animals, key=lambda a: a.animal_id)
        reward = float(cow.last_reward) if self.agents == 1 else float(sum(a.last_reward for a in after.animals) / 5)
        terminated = after.tick >= EPISODE_TICKS
        truncated = False
        info = {"resource_successes": self.behavior["forage_success"] + self.behavior["water_success"],
                "forage_successes": self.behavior["forage_success"], "water_successes": self.behavior["water_success"],
                "final_hunger": float(np.mean([a.physiology.hunger for a in after.animals])),
                "final_thirst": float(np.mean([a.physiology.thirst for a in after.animals])),
                "action_counts": dict(self.action_counts),
                "per_agent": {
                    animal.animal_id: {
                        "forage_successes": self.agent_behavior[animal.animal_id]["forage_success"],
                        "water_successes": self.agent_behavior[animal.animal_id]["water_success"],
                        "resource_successes": (self.agent_behavior[animal.animal_id]["forage_success"]
                                               + self.agent_behavior[animal.animal_id]["water_success"]),
                        "hunger": float(animal.physiology.hunger),
                        "thirst": float(animal.physiology.thirst),
                        "action_counts": dict(self.agent_action_counts[animal.animal_id]),
                    }
                    for animal in after.animals
                },
                "controlled_hunger": float(cow.physiology.hunger),
                "controlled_thirst": float(cow.physiology.thirst)}
        return self._observation(), reward, terminated, truncated, info


def train_and_evaluate(*, agents: int = 1, total_timesteps: int = 7200, seed: int = 1):
    env = PastureSB3Env(agents=agents, seed=seed)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, seed=seed, n_steps=256, batch_size=64, n_epochs=4,
                learning_rate=3e-4, ent_coef=0.01, verbose=0, device="cpu")
    model.learn(total_timesteps=total_timesteps)
    results = []
    for eval_seed in (seed + 100, seed + 101, seed + 102):
        obs, _ = env.reset(seed=eval_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(info)
    baselines = {"need_controller": [], "random": []}
    for name in baselines:
        for eval_seed in (seed + 100, seed + 101, seed + 102):
            obs, _ = env.reset(seed=eval_seed)
            done = False
            rng = np.random.default_rng(eval_seed)
            env.action_space.seed(eval_seed)
            while not done:
                if name == "random":
                    action = env.action_space.sample()
                else:
                    action_indices = []
                    for _, features in env.sim.decision_features()[:agents]:
                        action_indices.append(ACTION_LIST.index(NeedController().decide(features).action))
                    action = action_indices[0] if agents == 1 else np.asarray(action_indices, dtype=np.int64)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            baselines[name].append(info)
    summary = lambda rows: {
        "mean_resource_successes": float(np.mean([x["resource_successes"] for x in rows])),
        "mean_controlled_hunger": float(np.mean([x["controlled_hunger"] for x in rows])),
        "mean_controlled_thirst": float(np.mean([x["controlled_thirst"] for x in rows])),
        "mean_world_hunger": float(np.mean([x["final_hunger"] for x in rows])),
        "mean_world_thirst": float(np.mean([x["final_thirst"] for x in rows])),
        "per_agent": {
            animal_id: {
                metric: float(np.mean([row["per_agent"][animal_id][metric] for row in rows]))
                for metric in ("forage_successes", "water_successes", "resource_successes", "hunger", "thirst")
            }
            for animal_id in rows[0]["per_agent"]
        },
        "action_counts": {k.value: sum(x["action_counts"].get(k.value, 0) for x in rows) for k in ACTION_LIST},
    }
    return {"algorithm": "SB3 PPO", "agents": agents, "training_steps": total_timesteps,
            "seed": seed, "gymnasium": gym.__version__, "sb3": __import__("stable_baselines3").__version__,
            "trained": summary(results), "baselines": {name: summary(rows) for name, rows in baselines.items()},
            "evaluation_seeds": [seed + 100, seed + 101, seed + 102]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, choices=(1, 5), default=1)
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="fixed 128-step local development smoke")
    args = parser.parse_args()
    try:
        require_kaggle_for_full_workload("SB3 PPO pilot", smoke=args.smoke)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.smoke:
        if (args.steps, args.seed) != (7200, 1):
            parser.error("--smoke uses fixed limits; omit --steps and --seed")
        args.steps = 128
    print(train_and_evaluate(agents=args.agents, total_timesteps=args.steps, seed=args.seed))


if __name__ == "__main__":
    main()
