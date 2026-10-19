"""Paired, validation-only DAgger generalization experiments.

Domain changes are limited to resource counts, replenishment, and resource-site
layout. Every transition still goes through the original Simulation adapter.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from farm_sim.model import Position, ResourcePatch, Action
from farm_sim.navigation import navigate
from validation.renewable_rl import (
    ACTION_LIST, BehaviorTeacher, RenewablePastureEnv, collect_teacher_data,
)

NOMINAL = (8, 6, 1, False)
# Teacher-solvable curricula only; low-supply and very slow-regrowth stress
# profiles are measured separately, not used as noisy supervision targets.
TRAIN_PROFILE = ((8, 6, 1, False), (9, 7, 3, False),
                 (10, 8, 3, False), (12, 10, 3, True))
TEST_DOMAINS = {
    "nominal": (8, 6, 1, False),
    "layout_reference": (10, 8, 1, False),
    "layout_shift": (10, 8, 1, True),
    "supply_shift": (9, 7, 1, False),
    "regrowth_shift": (8, 6, 2, False),
    "combined_unseen": (12, 10, 1, True),
}


def domain_for(seed: int, profile: str):
    if profile == "nominal":
        return NOMINAL
    if profile == "randomized_train":
        rng = np.random.default_rng(seed ^ 0x5A17)
        return TRAIN_PROFILE[int(rng.integers(0, len(TRAIN_PROFILE)))]
    if profile in TEST_DOMAINS:
        return TEST_DOMAINS[profile]
    raise ValueError(f"unknown domain profile: {profile}")


class GeneralizationEnv(RenewablePastureEnv):
    """Renewable environment with held-out domain factors and optional history."""

    def __init__(self, *, agents: int, seed: int, profile: str = "nominal",
                 observation: str = "domain", random_layout: bool | None = None,
                 reward_mode: str = "legacy", refresh_observation: bool = True):
        self.profile = profile
        self.observation_kind = observation
        self.layout_override = random_layout
        self.refresh_observation = refresh_observation
        self._previous_base = None
        self._domain_values = NOMINAL
        super().__init__(agents=agents, seed=seed, forage_sites=8, water_sites=6,
                         regrowth_ticks=1, global_observations=True, context="rest",
                         reward_mode=reward_mode)
        per_animal = 30 + (5 if observation in {"domain", "history_domain"} else 0)
        if observation == "history_domain":
            per_animal += 30
        self.per_animal_observation_size = per_animal
        shape = (per_animal,) if agents == 1 else (agents * per_animal,)
        self.observation_space = spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    def _observation(self):
        if self.refresh_observation and self.sim is not None:
            # The simulator wrapper regrows patches at the start of a step.
            # Expose that already-due regrowth before selecting the next action,
            # not one decision late. Transitions/resources remain identical.
            world = self.sim.world
            changes = {}
            for kind in ("forage", "water"):
                patches = []
                for index, patch in enumerate(getattr(world, kind)):
                    deadline = self.regrow_at.get((kind, index))
                    if not patch.available and deadline is not None and deadline <= world.tick:
                        patch = replace(patch, available=True)
                    patches.append(patch)
                changes[kind] = tuple(patches)
            self.sim.world = replace(world, **changes)
        base = RenewablePastureEnv._observation(self)
        rows = base.reshape(self.agents, 30)
        extras = []
        if self.observation_kind in {"domain", "history_domain"}:
            forage, water, regrowth, _ = self._domain_values
            world = self.sim.world
            extras.extend((forage / 12.0, water / 10.0, regrowth / 4.0,
                           world.width / 14.0, world.height / 10.0))
        result = []
        for row in rows:
            parts = [row]
            if self.observation_kind in {"domain", "history_domain"}:
                parts.append(np.asarray(extras, dtype=np.float32))
            if self.observation_kind == "history_domain":
                history = np.zeros(30, dtype=np.float32) if self._previous_base is None else self._previous_base.reshape(self.agents, 30)[len(result)]
                parts.append(history)
            result.append(np.concatenate(parts))
        flat = result[0] if self.agents == 1 else np.concatenate(result)
        return np.clip(flat, 0.0, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        selected_seed = self.seed_value if seed is None else int(seed)
        self._domain_values = domain_for(selected_seed, self.profile)
        self.forage_sites, self.water_sites, self.regrowth_ticks, random_layout = self._domain_values
        if self.layout_override is not None:
            random_layout = self.layout_override
        self._layout_randomized = random_layout
        self._previous_base = None
        _, info = super().reset(seed=selected_seed, options=options)
        if random_layout:
            self._randomize_sites(selected_seed)
        info = info | {"domain_profile": self.profile,
                       "forage_sites": self.forage_sites,
                       "water_sites": self.water_sites,
                       "regrowth_ticks": self.regrowth_ticks,
                       "random_layout": random_layout}
        return self._observation(), info

    def _randomize_sites(self, seed: int):
        world = self.sim.world
        occupied = {a.position for a in world.animals} | set(world.obstacles)
        cells = [Position(x, y) for y in range(1, world.height - 1)
                 for x in range(1, world.width - 1) if Position(x, y) not in occupied]
        rng = np.random.default_rng(seed ^ 0xA10C)
        order = rng.permutation(len(cells))
        selected = [cells[i] for i in order[:self.forage_sites + self.water_sites]]
        forage = tuple(ResourcePatch(point) for point in selected[:self.forage_sites])
        water = tuple(ResourcePatch(point) for point in selected[self.forage_sites:])
        self.sim.world = replace(world, forage=forage, water=water)

    def step(self, action):
        before = RenewablePastureEnv._observation(self).copy()
        _, reward, terminated, truncated, info = super().step(action)
        self._previous_base = before
        return self._observation(), reward, terminated, truncated, info


class GeneralizationCloner:
    """Shared per-animal MLP; optional wider nonlinear bottleneck."""

    def __init__(self, observation_size: int, seed: int, agents: int,
                 architecture: str = "mlp"):
        torch.manual_seed(seed)
        self.observation_size = observation_size
        self.agents = agents
        self.architecture = architecture
        self.joint = architecture == "joint"
        width = 128 if architecture in {"wide", "history"} else 64
        input_width = observation_size * agents if self.joint else observation_size
        output_width = len(ACTION_LIST) * agents if self.joint else len(ACTION_LIST)
        self.network = nn.Sequential(
            nn.Linear(input_width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, output_width),
        )

    def fit(self, observations, labels, *, epochs=25):
        x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=7e-4, weight_decay=1e-5)
        for _ in range(epochs):
            permutation = torch.randperm(len(y))
            for batch in permutation.split(512):
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(x[batch])
                if self.joint:
                    logits = logits.reshape(-1, self.agents, len(ACTION_LIST))
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, len(ACTION_LIST)), y[batch].reshape(-1))
                else:
                    loss = nn.functional.cross_entropy(logits, y[batch])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, observation, deterministic=True):
        del deterministic
        array = np.asarray(observation, dtype=np.float32)
        rows = array.reshape(self.agents, self.observation_size) if self.agents > 1 else array.reshape(1, -1)
        input_tensor = torch.as_tensor(rows.reshape(1, -1) if self.joint else rows)
        with torch.no_grad():
            logits = self.network(input_tensor)
            if self.joint:
                actions = logits.reshape(self.agents, len(ACTION_LIST)).argmax(-1).cpu().numpy()
            else:
                actions = logits.argmax(-1).cpu().numpy()
        return (int(actions[0]) if self.agents == 1 else actions), None


def collect_domain_data(*, agents, seed, episodes, profile, observation,
                        rollout_policy=None, random_layout=None, joint=False):
    env = GeneralizationEnv(agents=agents, seed=seed, profile=profile,
                            observation=observation, random_layout=random_layout)
    teacher = BehaviorTeacher(env)
    xs, ys = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        teacher.reset()
        done = False
        while not done:
            labels = np.asarray(teacher.actions(), dtype=np.int64).reshape(agents)
            rows = obs.reshape(agents, env.per_animal_observation_size)
            if joint:
                xs.append(obs.copy())
                ys.append(labels.copy())
            else:
                xs.extend(rows.copy())
                ys.extend(labels.tolist())
            action = labels if rollout_policy is None else np.asarray(
                rollout_policy.predict(obs, deterministic=True)[0], dtype=np.int64).reshape(agents)
            env_action = int(action[0]) if agents == 1 else action
            obs, _, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
    env.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train_generalized_dagger(*, agents, seed, observation="history_domain",
                             initial_episodes=40, rounds=6, episodes_per_round=20,
                             training_profile="randomized_train", architecture=None):
    x, y = collect_domain_data(agents=agents, seed=10_000 + seed,
                               episodes=initial_episodes, profile=training_profile,
                               observation=observation, joint=architecture == "joint")
    architecture = architecture or ("history" if observation == "history_domain" else "mlp")
    per_animal_size = x.shape[1] // agents if architecture == "joint" else x.shape[1]
    model = GeneralizationCloner(per_animal_size, seed, agents, architecture).fit(x, y)
    sample_count = len(y)
    for round_index in range(rounds):
        xr, yr = collect_domain_data(
            agents=agents, seed=20_000 + seed * 100 + round_index,
            episodes=episodes_per_round, profile=training_profile,
            observation=observation, rollout_policy=model,
            joint=model.joint,
        )
        x, y = np.concatenate((x, xr)), np.concatenate((y, yr))
        model.fit(x, y)
        sample_count += len(yr)
    return model, {"dagger_rounds": rounds, "episodes_per_round": episodes_per_round,
                   "samples": int(sample_count), "observation": observation,
                   "training_profile": training_profile}


def _mask_observation(obs, env, intervention):
    if intervention is None:
        return obs
    rows = np.asarray(obs, dtype=np.float32).reshape(env.agents, env.per_animal_observation_size).copy()
    base = 30
    if intervention == "mask_global_supply":
        rows[:, 20:24] = 0.0
    elif intervention == "mask_previous_observation":
        if env.observation_kind != "history_domain":
            raise ValueError("history ablation requires history_domain observations")
        rows[:, base + 5:] = 0.0
    elif intervention == "mask_domain_context":
        if env.observation_kind not in {"domain", "history_domain"}:
            raise ValueError("domain ablation requires domain context features")
        rows[:, base:base + 5] = 0.0
    else:
        raise ValueError(intervention)
    return rows[0] if env.agents == 1 else rows.reshape(-1)


def _ordered_navigation_outcomes(world, actions):
    """Preview exact resource outcomes under simulator stable-ID arbitration.

    This mirrors only navigation/resource reservation from Simulation's tick
    transition, so it can distinguish a real earlier-cow contention from an
    action that made no progress or had no available target.
    """
    animals = {animal.animal_id: animal for animal in world.animals}
    forage, water = list(world.forage), list(world.water)
    initial_available = {
        "forage": {patch.position for patch in forage if patch.available},
        "water": {patch.position for patch in water if patch.available},
    }
    outcomes = {}
    for index, original in enumerate(sorted(world.animals, key=lambda animal: animal.animal_id)):
        animal = animals[original.animal_id]
        action = ACTION_LIST[int(actions[index])]
        if action not in {Action.GRAZE, Action.DRINK}:
            continue
        kind = "forage" if action == Action.GRAZE else "water"
        patches = forage if kind == "forage" else water
        current_available = {patch.position for patch in patches if patch.available}
        occupied = {other.position for other in animals.values() if other.animal_id != animal.animal_id}
        nav_world = replace(world, animals=tuple(animals.values()), forage=tuple(forage), water=tuple(water))
        nav = navigate(nav_world, animal, action, occupied)
        target = next((patch_index for patch_index, patch in enumerate(patches)
                       if patch.position == nav.position), None)
        if target is not None and patches[target].available:
            outcomes[animal.animal_id] = "success"
            patches[target] = replace(patches[target], available=False)
        elif not current_available:
            outcomes[animal.animal_id] = (
                "sequential_contention" if initial_available[kind] else "no_target_available")
        elif nav.reason == "no_reachable_target":
            outcomes[animal.animal_id] = f"navigation_{nav.subreason}"
        elif nav.moved:
            outcomes[animal.animal_id] = "navigation_progress"
        else:
            outcomes[animal.animal_id] = f"navigation_{nav.subreason}"
        animals[animal.animal_id] = replace(animal, position=nav.position)
    return outcomes


def evaluate_domain(policy, *, agents, seeds, domain, observation="history_domain",
                    ablation=None, trace=False, refresh_observation=True):
    forage, water, regrowth, random_layout = TEST_DOMAINS[domain]
    env = GeneralizationEnv(agents=agents, seed=seeds[0], profile=domain,
                            observation=observation, random_layout=random_layout,
                            refresh_observation=refresh_observation)
    teacher = BehaviorTeacher(env)
    episodes, disagreement = [], []
    action_counts = {action.value: 0 for action in ACTION_LIST}
    failure_buckets = {"target_resource_unavailable": 0, "resource_not_visible": 0,
                       "high_need_action_disagreement": 0, "other_action_disagreement": 0,
                       "student_teacher_decisions": 0}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        teacher.reset()
        done = False
        while not done:
            features = np.asarray(obs).reshape(agents, env.per_animal_observation_size)
            labels = np.asarray(teacher.actions(), dtype=np.int64).reshape(agents)
            model_obs = _mask_observation(obs, env, ablation)
            action = (labels.copy() if policy is None else np.asarray(
                policy.predict(model_obs, deterministic=True)[0], dtype=np.int64).reshape(agents))
            for action_index in action:
                action_counts[ACTION_LIST[int(action_index)].value] += 1
            before_food = {k: v["forage_success"] for k, v in env.agent_behavior.items()}
            before_water = {k: v["water_success"] for k, v in env.agent_behavior.items()}
            world_before = env.sim.world
            available_before = {
                "forage": {patch.position for patch in world_before.forage if patch.available},
                "water": {patch.position for patch in world_before.water if patch.available},
            }
            positions_before = {a.animal_id: a.position for a in world_before.animals}
            ordered_outcomes = _ordered_navigation_outcomes(world_before, action)
            for i in range(agents):
                animal_id = f"cow-{i + 1}"
                failure_buckets["student_teacher_decisions"] += 1
                if action[i] == labels[i]:
                    continue
                mismatch = {"cow": animal_id, "student_action": ACTION_LIST[action[i]].value,
                            "teacher_action": ACTION_LIST[labels[i]].value,
                            "hunger": float(features[i, 0]), "thirst": float(features[i, 1]),
                            "local_forage_visible": bool(features[i, 4]),
                            "local_water_visible": bool(features[i, 5]),
                            "global_forage_fraction": float(features[i, 20]),
                            "global_water_fraction": float(features[i, 21]),
                            "global_forage_distance": float(features[i, 22]),
                            "global_water_distance": float(features[i, 23]),
                            "previous_action": ACTION_LIST[int(np.argmax(features[i, 24:30]))].value}
                disagreement.append(mismatch)
                wants_forage = ACTION_LIST[labels[i]] == Action.GRAZE
                wants_water = ACTION_LIST[labels[i]] == Action.DRINK
                if (wants_forage and features[i, 20] == 0) or (wants_water and features[i, 21] == 0):
                    failure_buckets["target_resource_unavailable"] += 1
                elif (wants_forage and not features[i, 4]) or (wants_water and not features[i, 5]):
                    failure_buckets["resource_not_visible"] += 1
                elif max(features[i, 0], features[i, 1]) >= .6:
                    failure_buckets["high_need_action_disagreement"] += 1
                else:
                    failure_buckets["other_action_disagreement"] += 1
            env_action = int(action[0]) if agents == 1 else action
            obs, _, terminated, truncated, info = env.step(env_action)
            for i in range(agents):
                animal_id = f"cow-{i + 1}"
                action_type = ACTION_LIST[int(action[i])]
                if action_type in {Action.GRAZE, Action.DRINK}:
                    kind = "forage" if action_type == Action.GRAZE else "water"
                    success_key = "forage_success" if kind == "forage" else "water_success"
                    failure_buckets["resource_actions"] = failure_buckets.get("resource_actions", 0) + 1
                    if env.agent_behavior[animal_id][success_key] > (
                            before_food[animal_id] if kind == "forage" else before_water[animal_id]):
                        failure_buckets["resource_action_successes"] = failure_buckets.get("resource_action_successes", 0) + 1
                    else:
                        key = f"resource_action_{ordered_outcomes[animal_id]}"
                        failure_buckets[key] = failure_buckets.get(key, 0) + 1
            done = terminated or truncated
        episodes.append({"seed": seed, "per_agent": info["per_agent"]})
    env.close()
    metrics = {}
    for i in range(agents):
        animal = f"cow-{i + 1}"
        rows = [episode["per_agent"][animal] for episode in episodes]
        metrics[animal] = {key: float(np.mean([row[key] for row in rows]))
                           for key in ("forage_successes", "water_successes", "hunger", "thirst")}
    return {"domain": domain, "agents": agents, "seeds": list(seeds),
            "scenario": {"forage_sites": forage, "water_sites": water,
                         "regrowth_ticks": regrowth, "random_layout": random_layout},
            "per_agent_mean": metrics, "episodes": episodes,
            "action_distribution": {
                name: count / max(1, sum(action_counts.values()))
                for name, count in action_counts.items()},
            "teacher_disagreement_rate": len(disagreement) / max(1, failure_buckets["student_teacher_decisions"]),
            "disagreement_examples": disagreement[:100], "causal_buckets": failure_buckets,
            "intervention": ablation}


def run_observation_ablation(policy, *, agents, seeds, domains, observation="history_domain"):
    rows = []
    for domain in domains:
        for ablation in (None, "mask_global_supply", "mask_previous_observation", "mask_domain_context"):
            rows.append(evaluate_domain(policy, agents=agents, seeds=seeds, domain=domain,
                                        observation=observation, ablation=ablation))
    return rows


def train_ppo_finetune(*, agents, seed, total_timesteps, observation="domain",
                       warm_start=None, reward_mode="balanced_events"):
    """PPO comparator with a topology-checked one-animal imitation warm start."""
    from stable_baselines3 import PPO

    class TrainEnv(GeneralizationEnv):
        def __init__(self):
            super().__init__(agents=agents, seed=seed, profile="randomized_train",
                             observation=observation, random_layout=True,
                             reward_mode=reward_mode)

        def reset(self, *, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            return obs, info

    env = TrainEnv()
    model = PPO("MlpPolicy", env, seed=seed, n_steps=256, batch_size=64,
                n_epochs=4, learning_rate=1e-4 if warm_start is not None else 3e-4,
                ent_coef=0.01,
                verbose=0, device="cpu")
    if warm_start is not None:
        if agents != 1 or observation != "domain" or warm_start.observation_size != 35:
            raise ValueError("PPO warm start requires a single-animal 35-input cloner")
        policy = model.policy
        source = list(warm_start.network.children())
        target = [policy.mlp_extractor.policy_net[0],
                  policy.mlp_extractor.policy_net[2], policy.action_net]
        for src, dst in zip((source[0], source[2], source[4]), target):
            if src.weight.shape != dst.weight.shape or src.bias.shape != dst.bias.shape:
                raise ValueError("PPO and cloner policy layer shapes differ")
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
