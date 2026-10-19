"""Isolated, non-production RL experiments with explicit renewable supplies.

The historical starter world and its training/reward definitions are unchanged.
This module composes the existing SB3 adapter with a validation-only world
wrapper, behavior-facing observation additions, reward shaping, and imitation
teacher. It deliberately reuses ``Simulation`` for every state transition.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from farm_sim.model import Action, Position, ResourcePatch
from validation.sb3_pilot import ACTION_LIST, PastureSB3Env


def add_resource_sites(world, *, forage_sites: int, water_sites: int):
    """Return a copy of a world with resource counts explicitly set/spread."""
    selected = {p.position for p in world.forage + world.water}
    occupied = {a.position for a in world.animals} | set(world.obstacles)
    candidates = [
        Position(x, y)
        for y in range(1, world.height - 1)
        for x in range(1, world.width - 1)
        if Position(x, y) not in selected | occupied
    ]

    def extend(existing, count, reserved=()):
        chosen = list(existing)
        locations = selected | occupied | set(reserved)
        while len(chosen) < count:
            available = [p for p in candidates if p not in locations]
            if not available:
                raise ValueError("requested resource sites exceed available pasture cells")
            # Max-min spacing avoids clustering all additional supplies at one point.
            point = max(
                available,
                key=lambda p: (
                    min(abs(p.x - q.x) + abs(p.y - q.y) for q in locations),
                    -p.y,
                    -p.x,
                ),
            )
            chosen.append(ResourcePatch(point))
            locations.add(point)
        if len(chosen) > count:
            chosen = chosen[:count]
        return tuple(chosen)

    forage = extend(world.forage, forage_sites)
    # Keep forage sites reserved while spreading water sites.
    water = extend(world.water, water_sites, reserved=(p.position for p in forage))
    return replace(world, forage=forage, water=water)


class RenewablePastureEnv(PastureSB3Env):
    """PastureSB3Env with isolated supply, observation, and reward variants."""

    def __init__(
        self,
        *,
        agents: int,
        seed: int,
        forage_sites: int,
        water_sites: int,
        regrowth_ticks: int = 1,
        global_observations: bool = True,
        reward_mode: str = "legacy",
        context: str = "need_controller",
    ):
        if forage_sites < 1 or water_sites < 1:
            raise ValueError("renewable scenario requires at least one forage and water site")
        if context not in {"need_controller", "rest"}:
            raise ValueError("context must be 'need_controller' or 'rest'")
        if reward_mode not in {"legacy", "balanced_events", "state_cost"}:
            raise ValueError("unknown reward mode")
        self.forage_sites = forage_sites
        self.water_sites = water_sites
        self.global_observations = global_observations
        self.reward_mode = reward_mode
        self.context = context
        super().__init__(agents=agents, regrowth_ticks=regrowth_ticks, seed=seed)
        extra = 10 if global_observations else 0
        shape = (20 + extra,) if agents == 1 else (5 * (20 + extra),)
        self.observation_space = type(self.observation_space)(
            low=0.0, high=1.0, shape=shape, dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        assert self.sim is not None
        self.sim.world = add_resource_sites(
            self.sim.world,
            forage_sites=self.forage_sites,
            water_sites=self.water_sites,
        )
        if self.context == "rest":
            # A context action policy for the four non-controlled animals.
            self.policy.context = "rest"
        return self._observation(), info | {
            "forage_sites": self.forage_sites,
            "water_sites": self.water_sites,
            "regrowth_ticks": self.regrowth_ticks,
        }

    def _observation(self):
        assert self.sim is not None
        rows = [np.asarray(features, dtype=np.float32)
                for _, features in self.sim.decision_features()]
        if self.global_observations:
            world = self.sim.world
            scale = max(world.width, world.height, 1)
            available_forage = [p for p in world.forage if p.available]
            available_water = [p for p in world.water if p.available]
            for i, animal in enumerate(world.animals):
                nearest_forage = min(
                    (abs(animal.position.x - p.position.x) + abs(animal.position.y - p.position.y)
                     for p in available_forage), default=scale
                )
                nearest_water = min(
                    (abs(animal.position.x - p.position.x) + abs(animal.position.y - p.position.y)
                     for p in available_water), default=scale
                )
                rows[i] = np.concatenate((rows[i], np.asarray([
                    len(available_forage) / self.forage_sites,
                    len(available_water) / self.water_sites,
                    min(1.0, nearest_forage / scale),
                    min(1.0, nearest_water / scale),
                    *(float(animal.action == action) for action in ACTION_LIST),
                ], dtype=np.float32)))
        selected = rows[:self.agents]
        obs = selected[0] if self.agents == 1 else np.concatenate(selected)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def step(self, action):
        assert self.sim is not None
        before = self.sim.world
        behavior_before = {
            animal_id: (counts["forage_success"], counts["water_success"])
            for animal_id, counts in self.agent_behavior.items()
        }
        obs, reward, terminated, truncated, info = super().step(action)
        if self.reward_mode != "legacy":
            animals_before = {a.animal_id: a for a in before.animals}
            animals_after = {a.animal_id: a for a in self.sim.world.animals}
            shaped = []
            for animal_id in sorted(animals_after)[:self.agents]:
                old = animals_before[animal_id].physiology
                new = animals_after[animal_id].physiology
                if self.reward_mode == "state_cost":
                    # Direct state objective: persistent high needs are costly;
                    # legacy per-action reward bonuses are excluded.
                    # Normalize each drive by the relief per successful action
                    # so DRINK's larger absolute decrement cannot dominate GRAZE.
                    shaped.append(-(new.hunger / 0.22 + new.thirst / 0.30))
                else:
                    previous_food, previous_water = behavior_before[animal_id]
                    food_success = self.agent_behavior[animal_id]["forage_success"] > previous_food
                    water_success = self.agent_behavior[animal_id]["water_success"] > previous_water
                    hunger_progress = max(0.0, min(1.0, (old.hunger + 0.025 - new.hunger) / 0.22))
                    thirst_progress = max(0.0, min(1.0, (old.thirst + 0.03 - new.thirst) / 0.30))
                    # Equal event credit removes the legacy +0.42 DRINK vs
                    # +0.32 GRAZE asymmetry from this validation-only variant.
                    shaped.append(
                        float(food_success) + float(water_success)
                        + 0.1 * (hunger_progress + thirst_progress)
                        - 0.2 * (new.hunger + new.thirst)
                    )
            reward = float(shaped[0] if self.agents == 1 else np.mean(shaped))
        return obs, reward, terminated, truncated, info


class BehaviorTeacher:
    """Transparent teacher; direct global availability is for an oracle ceiling."""

    def __init__(self, env: RenewablePastureEnv):
        self.env = env
        self.targets: dict[str, Action] = {}
        self.previous_drives: dict[str, tuple[float, float]] = {}

    def reset(self):
        self.targets.clear()
        self.previous_drives.clear()

    def actions(self) -> np.ndarray:
        assert self.env.sim is not None
        world = self.env.sim.world
        availability = {Action.GRAZE: any(p.available for p in world.forage),
                        Action.DRINK: any(p.available for p in world.water)}
        result = []
        for animal in world.animals[:self.env.agents]:
            hunger = animal.physiology.hunger
            thirst = animal.physiology.thirst
            previous = self.previous_drives.get(animal.animal_id)
            target = self.targets.get(animal.animal_id)
            if previous is not None:
                if ((target == Action.GRAZE and hunger < previous[0] - 0.1)
                        or (target == Action.DRINK and thirst < previous[1] - 0.1)):
                    target = None
            if target not in {Action.GRAZE, Action.DRINK}:
                target = None
            if target is not None and not availability[target]:
                target = None
            if target is None:
                scores = []
                if availability[Action.GRAZE] and hunger > 0.15:
                    distance = min(abs(animal.position.x - p.position.x) + abs(animal.position.y - p.position.y)
                                   for p in world.forage if p.available)
                    scores.append((hunger - 0.01 * distance, Action.GRAZE))
                if availability[Action.DRINK] and thirst > 0.15:
                    distance = min(abs(animal.position.x - p.position.x) + abs(animal.position.y - p.position.y)
                                   for p in world.water if p.available)
                    scores.append((thirst - 0.01 * distance, Action.DRINK))
                target = max(scores, default=(0.0, Action.REST), key=lambda item: item[0])[1]
            self.targets[animal.animal_id] = target
            self.previous_drives[animal.animal_id] = (hunger, thirst)
            result.append(ACTION_LIST.index(target))
        return np.asarray(result[0] if self.env.agents == 1 else result, dtype=np.int64)


class BehaviorCloner:
    """Small supervised MLP trained on teacher rollouts, with SB3-like predict."""

    def __init__(self, observation_size: int, action_count: int, seed: int, agents: int = 1):
        torch.manual_seed(seed)
        self.observation_size = observation_size
        self.agents = agents
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_count),
        )

    def fit(self, observations: np.ndarray, labels: np.ndarray, *, epochs: int = 30):
        x = torch.as_tensor(observations, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self.network(x), y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, observation, deterministic=True):
        del deterministic
        array = np.asarray(observation, dtype=np.float32)
        if self.agents == 5:
            array = array.reshape(self.agents, self.observation_size)
        x = torch.as_tensor(array, dtype=torch.float32)
        with torch.no_grad():
            logits = self.network(x)
            action = torch.argmax(logits, dim=-1).cpu().numpy()
        return action, None


def collect_teacher_data(*, agents: int, episodes: int, seed: int,
                         forage_sites: int, water_sites: int,
                         regrowth_ticks: int = 1, context: str = "rest",
                         rollout_policy=None):
    env = RenewablePastureEnv(
        agents=agents, seed=seed, forage_sites=forage_sites,
        water_sites=water_sites, regrowth_ticks=regrowth_ticks,
        global_observations=True, context=context,
    )
    teacher = BehaviorTeacher(env)
    obs_rows, labels = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        teacher.reset()
        done = False
        while not done:
            teacher_actions = teacher.actions()
            if agents == 1:
                obs_rows.append(obs.copy())
                labels.append(int(teacher_actions))
            else:
                obs_rows.extend(obs.reshape(agents, -1).copy())
                labels.extend(teacher_actions.tolist())
            actions = teacher_actions if rollout_policy is None else rollout_policy.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
    env.close()
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def evaluate_policy(policy, *, agents: int, seeds: tuple[int, ...],
                    forage_sites: int, water_sites: int,
                    regrowth_ticks: int = 1, mode: str = "learned",
                    context: str = "rest", global_observations: bool = True,
                    reward_mode: str = "legacy"):
    env = RenewablePastureEnv(
        agents=agents, seed=seeds[0], forage_sites=forage_sites,
        water_sites=water_sites, regrowth_ticks=regrowth_ticks,
        global_observations=global_observations,
        reward_mode=reward_mode, context=context,
    )
    teacher = BehaviorTeacher(env)
    rows = []
    per_seed_rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        teacher.reset()
        env.action_space.seed(seed)
        done = False
        while not done:
            if mode == "teacher":
                action = teacher.actions()
            elif mode == "random":
                action = env.action_space.sample()
            else:
                action = policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append(info)
        per_seed_rows.append({"seed": seed, "per_agent": info["per_agent"]})
    env.close()
    per_agent = {
        animal_id: {
            key: float(np.mean([row["per_agent"][animal_id][key] for row in rows]))
            for key in ("forage_successes", "water_successes", "resource_successes", "hunger", "thirst")
        }
        for animal_id in rows[0]["per_agent"]
    }
    measured = list(per_agent)[:agents]
    return {
        "mode": mode,
        "evaluation_seeds": list(seeds),
        "per_seed": per_seed_rows,
        "per_agent": per_agent,
        "controlled_mean": {
            key: float(np.mean([per_agent[animal][key] for animal in measured]))
            for key in next(iter(per_agent.values()))
        },
        "action_frequencies": {
            animal: {
                action: int(sum(row["per_agent"][animal]["action_counts"].get(action, 0)
                                for row in rows))
                for action in (candidate.value for candidate in Action)
            }
            for animal in measured
        },
    }


def teacher_ceiling(*, agents: int, seeds: tuple[int, ...], forage_sites: int,
                    water_sites: int, regrowth_ticks: int = 1):
    return evaluate_policy(
        None, agents=agents, seeds=seeds, forage_sites=forage_sites,
        water_sites=water_sites, regrowth_ticks=regrowth_ticks, mode="teacher",
    )


def train_dagger(*, agents: int, seed: int, forage_sites: int,
                 water_sites: int, initial_episodes: int = 30,
                 rounds: int = 3, episodes_per_round: int = 10,
                 regrowth_ticks: int = 1):
    """Behavior cloning with dataset aggregation on learner-visited states."""
    x, y = collect_teacher_data(
        agents=agents, episodes=initial_episodes, seed=10_000 + seed,
        forage_sites=forage_sites, water_sites=water_sites,
        regrowth_ticks=regrowth_ticks,
    )
    model = BehaviorCloner(30, len(Action), seed, agents=agents).fit(x, y)
    for round_index in range(rounds):
        x_round, y_round = collect_teacher_data(
            agents=agents, episodes=episodes_per_round,
            seed=20_000 + seed * 100 + round_index,
            forage_sites=forage_sites, water_sites=water_sites,
            regrowth_ticks=regrowth_ticks, rollout_policy=model,
        )
        x = np.concatenate((x, x_round))
        y = np.concatenate((y, y_round))
        model.fit(x, y, epochs=30)
    return model, {"training_samples": int(len(y)), "dagger_rounds": rounds}


def train_ppo(*, agents: int, seed: int, total_timesteps: int,
              forage_sites: int, water_sites: int, regrowth_ticks: int = 1,
              global_observations: bool = False, reward_mode: str = "legacy",
              context: str = "rest"):
    from stable_baselines3 import PPO

    env = RenewablePastureEnv(
        agents=agents, seed=seed, forage_sites=forage_sites,
        water_sites=water_sites, regrowth_ticks=regrowth_ticks,
        global_observations=global_observations,
        reward_mode=reward_mode, context=context,
    )
    model = PPO(
        "MlpPolicy", env, seed=seed, n_steps=256, batch_size=64,
        n_epochs=4, learning_rate=3e-4, ent_coef=0.01,
        verbose=0, device="cpu",
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model
