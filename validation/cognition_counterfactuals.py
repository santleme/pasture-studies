#!/usr/bin/env python3
"""Paired, synthetic cognition intervention and feature-ablation experiment.

This harness implements the preregistered one-decision memory and perception
contrasts.  It does not alter rewards, action selection, or policy parameters.
All outcomes are software-task constructs; no biological inference is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from farm_sim import simulation as simulation_module
from farm_sim.memory import retrieve
from farm_sim.model import (Action, AnimalMemory, MemoryEvent, Position, ResourcePatch)
from farm_sim.perception import encode_features, perceive
from farm_sim.policy import load_checkpoint
from farm_sim.simulation import Simulation, make_world

HORIZON = 72
DECISION_TICK = 12
TARGET_ID = "cow-1"
BOOTSTRAP_REPS = 4000
FEATURE_INDEX = {"mem_water_success": 12, "mem_water_blocked": 14, "water_available": 9}
ARM_NAMES = {
    "memory": [("success", None), ("blocked", None), ("sham_success", None),
                ("success", "memory"), ("blocked", "memory")],
    "perception": [("available", None), ("masked", None), ("sham_available", None),
                   ("available", "perception"), ("masked", "perception")],
}


def scenario_world(scenario: str, seed: int):
    """Build the deterministic source scenario before the tick-12 fixture.

    The water patch is anchored only when creating the shared intervention
    fixture; these worlds are not unmodified ecological/generalization tests.
    """
    if scenario == "starter_seed":
        world = make_world(seed)
    elif scenario == "wide_grid_20x14":
        world = make_world(seed, width=20, height=14)
    elif scenario == "resource_scarce_relocated":
        world = make_world(seed)
        world = replace(world, forage=(ResourcePatch(Position(1, 8), True),
                                      ResourcePatch(Position(2, 2), False),
                                      ResourcePatch(Position(3, 7), False)),
                        water=(ResourcePatch(Position(12, 8), True),))
    elif scenario == "obstacle_topology":
        world = make_world(seed)
        # Explicit test topology: same deterministic starter map plus a full
        # barrier with a single gate. It remains a synthetic software fixture.
        wall = {Position(world.width // 2, y) for y in range(1, world.height - 1)}
        wall.discard(Position(world.width // 2, world.height // 2))
        world = replace(world, obstacles=frozenset(set(world.obstacles) | wall))
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return world


def prepare_decision_fixture(world, policy):
    """Replay shared pre-intervention history, then freeze the tick-12 fixture.

    The initial warmup is run once per paired cell with the same checkpoint.
    At the frozen decision snapshot, a single age-12 successful water-access
    memory is installed for cow-1 and an available water patch is placed at its
    current cell. Every treatment/sham/ablation branch clones this exact state.
    """
    if world.tick > DECISION_TICK:
        raise ValueError("decision fixture starts after frozen tick 12")
    sim = Simulation(world, policy.copy() if hasattr(policy, "copy") else policy)
    while sim.world.tick < DECISION_TICK:
        sim.step()
    frozen = sim.world
    target = next(a for a in frozen.animals if a.animal_id == TARGET_ID)
    animals = tuple(replace(a, memory=AnimalMemory((MemoryEvent(0, "water_access", True, target.position),)))
                    if a.animal_id == TARGET_ID else a for a in frozen.animals)
    return replace(frozen, animals=animals,
                   water=(ResourcePatch(target.position, True),))


def intervention_world(world, factor: str, condition: str):
    """Return world clone for a single memory-only intervention or sham."""
    if factor != "memory" or condition not in {"success", "blocked", "sham_success"}:
        return world
    if condition == "sham_success":
        condition = "success"
    animals = []
    for animal in world.animals:
        if animal.animal_id == TARGET_ID:
            memory = AnimalMemory((MemoryEvent(0, "water_access",
                                                condition == "success", animal.position),))
            animal = replace(animal, memory=memory)
        animals.append(animal)
    return replace(world, animals=tuple(animals))


class FeatureAdapter:
    """Apply exactly one encoded input intervention, then delegate unchanged."""
    def __init__(self, policy, *, intervention: tuple[int, float] | None,
                 ablation_index: int | None):
        self.policy = policy
        self.intervention = intervention
        self.ablation_index = ablation_index if isinstance(ablation_index, tuple) else ((ablation_index,) if ablation_index is not None else ())
        self.first_features = None
        self.first_probabilities = None
        self.first_action = None
        self.intervention_count = 0
        self.input_digest = None

    def decide(self, features):
        x = list(float(v) for v in features)
        # Target ID is not encoded, so simulation's stable animal order is used;
        # only the first call is target cow-1 under sorted ID ordering.
        is_target = self.first_features is None
        if is_target:
            if self.intervention is not None:
                index, value = self.intervention
                if x[index] != value:
                    x[index] = value
                    self.intervention_count += 1
            for index in self.ablation_index:
                x[index] = 0.0
            # Save the features actually consumed by the model, and hash input
            # from fixed binary64 repr for sham/replay equality evidence.
            self.first_features = tuple(x)
            self.input_digest = hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()
        result = self.policy.decide(x)
        if is_target:
            self.first_probabilities = tuple(float(v) for v in result.probabilities)
            self.first_action = result.action.value
        return result


def run_episode(policy, initial_world, *, factor, condition, ablation=None, horizon=HORIZON):
    if initial_world.tick != DECISION_TICK:
        initial_world = prepare_decision_fixture(initial_world, policy)
    world = intervention_world(initial_world, factor, condition)
    intervention = None
    if factor == "perception" and condition == "masked":
        intervention = (FEATURE_INDEX["water_available"], 0.0)
    elif factor == "perception" and condition in {"available", "sham_available"}:
        intervention = None
    elif factor == "memory" and condition in {"success", "blocked", "sham_success"}:
        # The memory event itself carries the intervention; encoded features
        # are obtained from ordinary retrieve/encode in Simulation.step.
        intervention = None
    else:
        raise ValueError((factor, condition))
    ablation_index = tuple(range(12, 17)) if ablation == "memory" else (
        (FEATURE_INDEX["water_available"],) if ablation == "perception" else ())
    adapter = FeatureAdapter(policy, intervention=intervention,
                             ablation_index=ablation_index)
    sim = Simulation(world, adapter)
    nav_counts = Counter()
    nav_by_animal = Counter()
    invalid = 0
    trajectories = {a.animal_id: [] for a in world.animals}
    physiology = {a.animal_id: [] for a in world.animals}
    reward_curve = []
    action_counts = Counter()
    resource_use = {"forage": 0, "water": 0}
    original_navigate = simulation_module.navigate
    def counted_navigate(*args, **kwargs):
        outcome = original_navigate(*args, **kwargs)
        nav_counts[outcome.subreason] += 1
        nav_by_animal[(args[1].animal_id, outcome.subreason)] += 1
        return outcome
    simulation_module.navigate = counted_navigate
    try:
        for _ in range(horizon):
            result = sim.step()
            tick_total = 0.0
            for animal in result.world.animals:
                tick_total += animal.last_reward
                action_counts[animal.action.value] += 1
                trajectories[animal.animal_id].append((animal.position.x, animal.position.y, animal.action.value))
                physiology[animal.animal_id].append({k: getattr(animal.physiology, k) for k in ("hunger", "thirst", "fatigue", "social_need")})
                probs = animal.last_probabilities
                if len(probs) != len(Action) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) > 1e-6:
                    invalid += 1
            reward_curve.append(tick_total)
            resource_use["forage"] += sum(e.kind == "forage_success" for e in result.events)
            resource_use["water"] += sum(e.kind == "water_success" for e in result.events)
    finally:
        simulation_module.navigate = original_navigate
    # Capture pre-decision source truth and exact changed feature for fidelity.
    observation = perceive(world, next(a for a in world.animals if a.animal_id == TARGET_ID))
    mem_features = retrieve(next(a for a in world.animals if a.animal_id == TARGET_ID).memory, world.tick)
    expected = list(encode_features(observation, mem_features,
                                    next(a for a in world.animals if a.animal_id == TARGET_ID),
                                    world.width, world.height))
    if intervention is not None:
        expected[intervention[0]] = intervention[1]
    for index in ablation_index:
        expected[index] = 0.0
    # The adapter's first call is sorted cow-1 and should match this vector.
    fidelity = (adapter.first_features == tuple(expected))
    baseline_memory = next(a for a in initial_world.animals if a.animal_id == TARGET_ID).memory
    treated_memory = next(a for a in world.animals if a.animal_id == TARGET_ID).memory
    memory_edit_count = sum(a.success != b.success or a.tick != b.tick or a.kind != b.kind or a.position != b.position
                            for a, b in zip(baseline_memory.events, treated_memory.events))
    baseline_features = list(encode_features(
        perceive(initial_world, next(a for a in initial_world.animals if a.animal_id == TARGET_ID)),
        retrieve(next(a for a in initial_world.animals if a.animal_id == TARGET_ID).memory, initial_world.tick),
        next(a for a in initial_world.animals if a.animal_id == TARGET_ID),
        initial_world.width, initial_world.height))
    if ablation == "memory":
        for index in range(12, 17): baseline_features[index] = 0.0
    elif ablation == "perception":
        baseline_features[FEATURE_INDEX["water_available"]] = 0.0
    changed_features = [i for i, (a, b) in enumerate(zip(baseline_features, expected)) if a != b]
    return {
        "factor": factor, "condition": condition, "ablation": ablation,
        "first_action": adapter.first_action, "first_probabilities": list(adapter.first_probabilities or ()),
        "p_drink": (adapter.first_probabilities or (0.0,) * len(Action))[tuple(Action).index(Action.DRINK)],
        "first_features": list(adapter.first_features or ()), "input_digest": adapter.input_digest,
        "intervention_count": len(changed_features),
        "intervention_changed_feature_indices": changed_features,
        "memory_event_mutation_count": memory_edit_count,
        "intervention_fidelity": fidelity,
        "target_action_counts": dict(Counter(a for _, _, a in trajectories[TARGET_ID])),
        "all_animal_action_counts": dict(action_counts),
        "target_trajectory": [list(p) for p in trajectories[TARGET_ID]],
        "all_animal_trajectories": {k: [list(p) for p in v] for k, v in trajectories.items()},
        "target_physiology_trajectory": physiology[TARGET_ID],
        "all_animal_physiology_trajectories": physiology,
        "reward_curve_all_animals": reward_curve,
        "total_reward": float(sum(reward_curve)), "final_target_physiology": physiology[TARGET_ID][-1],
        "navigation_subreason_counts": dict(nav_counts),
        "navigation_subreason_by_animal": {animal: dict(Counter({reason: count for (aid, reason), count in nav_by_animal.items() if aid == animal})) for animal in trajectories},
        "invalid_actions": invalid,
        "resource_use": resource_use,
    }


def _crossed_bootstrap(cells, value_fn, *, seed=771, reps=BOOTSTRAP_REPS):
    """Resample training-checkpoint and world-seed IDs independently."""
    training_ids = sorted({c["training_seed"] for c in cells})
    world_ids = sorted({c["world_seed"] for c in cells})
    matrix = {(c["training_seed"], c["world_seed"]): float(value_fn(c)) for c in cells}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        selected_training = (rng.choice(training_ids, len(training_ids), replace=True)
                            if len(training_ids) > 1 else training_ids)
        selected_worlds = rng.choice(world_ids, len(world_ids), replace=True)
        values = [matrix[(int(t), int(w))] for t in selected_training for w in selected_worlds
                  if (int(t), int(w)) in matrix]
        if values:
            draws.append(float(np.mean(values)))
    point = float(np.mean(list(matrix.values()))) if matrix else None
    low, high = (float(np.quantile(draws, .025)), float(np.quantile(draws, .975))) if draws else (None, None)
    return {"mean": point, "ci95": [low, high],
            "ci97_5": [float(np.quantile(draws, .0125)), float(np.quantile(draws, .9875))] if draws else [None, None],
            "bootstrap_two_sided_p": min(1.0, 2 * min(float(np.mean(np.asarray(draws) <= 0)),
                                                        float(np.mean(np.asarray(draws) >= 0)))) if draws else None,
            "training_seed_clusters": len(training_ids),
            "world_seed_clusters": len(world_ids), "paired_cells": len(matrix),
            "bootstrap": "crossed training-seed x world-seed percentile; paired arms retained"}


def paired_stats(cells, *, gate_authorized=False):
    """Estimate preregistered contrasts within each fixed family, no pooling."""
    result = {}
    families = sorted({c["environment_family"] for c in cells})
    for family in families:
        subset = [c for c in cells if c["environment_family"] == family]
        def effect(cell, factor, ablated=False):
            arms = cell[factor]
            if factor == "memory":
                a = arms["success_ablation" if ablated else "success"]["p_drink"]
                b = arms["blocked_ablation" if ablated else "blocked"]["p_drink"]
            else:
                a = arms["available_ablation" if ablated else "available"]["p_drink"]
                b = arms["masked_ablation" if ablated else "masked"]["p_drink"]
            return a - b
        result[family] = {}
        raw_factor_p = {}
        for factor in ("memory", "perception"):
            primary = _crossed_bootstrap(subset, lambda c: effect(c, factor), seed=991 + len(family))
            ablation = _crossed_bootstrap(subset, lambda c: effect(c, factor, True), seed=992 + len(family))
            attenuation = _crossed_bootstrap(subset, lambda c: effect(c, factor) - effect(c, factor, True), seed=993 + len(family))
            enough = primary["training_seed_clusters"] >= 10 and primary["world_seed_clusters"] >= 30
            # Intersection-union factor p-value is max(primary, interaction).
            raw_factor_p[factor] = max(primary["bootstrap_two_sided_p"] or 1.0,
                                       attenuation["bootstrap_two_sided_p"] or 1.0)
            result[family][factor] = {
                "primary_effect": primary, "ablation_effect": ablation,
                "attenuation_interaction": attenuation,
                "preregistered_influence_margin": 0.05,
                "required_precision_half_width": 0.025,
                "sample_gate": "SAMPLE_COUNT_MET" if enough else "INCONCLUSIVE_UNDERPOWERED",
                "precision_gate": "INCONCLUSIVE_UNDERPOWERED" if max((primary["ci95"][1] - primary["ci95"][0]) / 2,
                                                                          (attenuation["ci95"][1] - attenuation["ci95"][0]) / 2) > 0.025 else "HALF_WIDTH_MET",
                "causal_gate": "INCONCLUSIVE_UNDERPOWERED" if not enough else "PENDING_HOLM_AND_SIGNED_MARGIN",
            }
        ordered = sorted(raw_factor_p, key=raw_factor_p.get)
        running = 0.0
        for rank, factor in enumerate(ordered):
            adjusted = min(1.0, raw_factor_p[factor] * (len(ordered) - rank))
            running = max(running, adjusted)
            result[family][factor]["holm_adjusted_factor_p"] = running
            estimates = result[family][factor]
            p_ci = estimates["primary_effect"]["ci97_5"]
            i_ci = estimates["attenuation_interaction"]["ci97_5"]
            positive_gate = (estimates["primary_effect"]["mean"] is not None and
                             estimates["primary_effect"]["mean"] >= 0.05 and p_ci[0] > 0 and
                             estimates["attenuation_interaction"]["mean"] > 0 and i_ci[0] > 0 and running < 0.05)
            estimates["holm_conservative_simultaneous_ci"] = "Bonferroni 97.5% intervals (conservative alongside Holm-adjusted factor p)"
            if not gate_authorized:
                estimates["causal_gate"] = "EXPLORATORY_ONLY_NO_PASS"
            elif estimates["sample_gate"] != "SAMPLE_COUNT_MET" or estimates["precision_gate"] != "HALF_WIDTH_MET":
                estimates["causal_gate"] = "INCONCLUSIVE_UNDERPOWERED"
            else:
                estimates["causal_gate"] = "PASS" if positive_gate else "FAIL_NO_DETECTED_PREDECLARED_INFLUENCE"
    return result


def _episode_policy(path):
    return load_checkpoint(path)


def _find_training_summary(checkpoint):
    for directory in Path(checkpoint).resolve().parents:
        candidate = directory / "training-summary.json"
        if candidate.is_file():
            return candidate
    raise ValueError(f"no training-summary.json found for checkpoint {checkpoint}")


def _validate_checkpoint(path, expected_ids=None):
    path = Path(path)
    policy = _episode_policy(path)
    sidecar = path.with_suffix(".json")
    doc = json.loads(sidecar.read_text())
    metadata = doc.get("metadata", {})
    condition = metadata.get("training_condition", metadata.get("condition"))
    required_condition = "encoder_generated_wider_support_current_objective"
    if condition != required_condition:
        raise ValueError(f"checkpoint {path} is not from confirmatory encoder-generated wider-support condition (got {condition!r})")
    if expected_ids is not None and policy.seed not in expected_ids:
        raise ValueError(f"checkpoint seed {policy.seed} is outside the frozen confirmatory cohort")
    if metadata.get("training_seed") != policy.seed:
        raise ValueError(f"checkpoint {path} metadata training_seed does not match checkpoint seed")

    summary_path = _find_training_summary(path)
    summary_doc = json.loads(summary_path.read_text())
    try:
        training = summary_doc["training"][required_condition]
        seed_entry = next(item for item in training["training_seeds"] if item["seed"] == policy.seed)
        selected_episode = int(seed_entry["selected_episode"])
        candidates = [int(n) for n in training["candidate_episodes"]]
        validation = seed_entry["validation_mean_return_by_candidate"]
        highest = max(float(validation[str(n)]) for n in candidates)
        expected_selected = min(n for n in candidates if float(validation[str(n)]) == highest)
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ValueError(f"training summary lacks valid checkpoint-selection provenance for seed {policy.seed}") from exc
    if selected_episode != expected_selected or int(metadata.get("selected_episode", -1)) != selected_episode:
        raise ValueError(f"checkpoint {path} is not the summary-selected validation checkpoint (expected episode {expected_selected})")
    for n in candidates:
        if str(n) not in metadata.get("candidate_validation_means", {}):
            raise ValueError(f"checkpoint {path} omits frozen validation candidate {n}")
        if not math.isclose(float(metadata["candidate_validation_means"][str(n)]),
                            float(validation[str(n)]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"checkpoint {path} validation metadata differs from training summary")

    relative_path = path.resolve().relative_to(ROOT).as_posix()
    relative_sidecar = sidecar.resolve().relative_to(ROOT).as_posix()
    expected_hashes = summary_doc.get("checkpoint_sha256", {})
    weights_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if expected_hashes.get(relative_path) != weights_hash or expected_hashes.get(relative_sidecar) != sidecar_hash:
        raise ValueError(f"checkpoint/sidecar hashes are not allowlisted in frozen training summary: {path}")
    provenance = {"training_summary_path": summary_path.resolve().relative_to(ROOT).as_posix(),
                  "training_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  "selected_episode": selected_episode,
                  "candidate_validation_means": {str(n): float(validation[str(n)]) for n in candidates}}
    return policy, weights_hash, sidecar_hash, provenance


def collect_arms(policy, base):
    outputs = {"memory": {}, "perception": {}}
    for factor, arms in ARM_NAMES.items():
        for condition, ablation in arms:
            label = condition + ("_ablation" if ablation else "")
            outputs[factor][label] = run_episode(policy.copy(), base, factor=factor,
                                                 condition=condition, ablation=ablation)
        sham = outputs[factor]["sham_success" if factor == "memory" else "sham_available"]
        control = outputs[factor]["success" if factor == "memory" else "available"]
        sham["bit_identical_to_control"] = all(sham[k] == control[k] for k in control if k != "condition")
    return outputs


def _variance_components(matrix):
    """Two-way random-effects ANOVA components; only dispersion is returned."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("pilot precision matrix needs >=2 checkpoint and >=2 world clusters")
    k, n = x.shape
    grand = float(np.mean(x))
    row = x.mean(axis=1)
    col = x.mean(axis=0)
    residual = x - row[:, None] - col[None, :] + grand
    ms_checkpoint = n * float(np.sum((row - grand) ** 2)) / (k - 1)
    ms_world = k * float(np.sum((col - grand) ** 2)) / (n - 1)
    ms_residual = float(np.sum(residual ** 2)) / ((k - 1) * (n - 1))
    v_checkpoint = max(0.0, (ms_checkpoint - ms_residual) / n)
    v_world = max(0.0, (ms_world - ms_residual) / k)
    v_residual = max(0.0, ms_residual)
    total = v_checkpoint + v_world + v_residual
    return {"variance_checkpoint": v_checkpoint, "variance_world": v_world,
            "variance_checkpoint_by_world_residual": v_residual,
            "checkpoint_icc": v_checkpoint / total if total else 0.0,
            "world_icc": v_world / total if total else 0.0,
            "pilot_checkpoint_clusters": k, "pilot_world_clusters": n}


def _required_world_count(components, *, checkpoint_count=10, target_half_width=0.025,
                          confidence_z=1.96, maximum=60):
    vc = components["variance_checkpoint"]
    vw = components["variance_world"]
    vr = components["variance_checkpoint_by_world_residual"]
    for n in range(30, maximum + 1):
        half = confidence_z * math.sqrt(vc / checkpoint_count + vw / n + vr / (checkpoint_count * n))
        if half <= target_half_width:
            return {"required_world_seeds": n, "planned_half_width_at_required_n": half,
                    "target_half_width": target_half_width, "precision_attainable_with_approved_reserves": True}
    half = confidence_z * math.sqrt(vc / checkpoint_count + vw / maximum + vr / (checkpoint_count * maximum))
    return {"required_world_seeds": maximum, "planned_half_width_at_required_n": half,
            "target_half_width": target_half_width, "precision_attainable_with_approved_reserves": False}


def pilot_precision_decision(checkpoints, output):
    """Run only excluded pilot seeds; save dispersion/sample plan, never effects.

    The output intentionally excludes any pilot endpoint means, signs, contrasts,
    or policy ordering; it contains only variance components/ICCs and the frozen
    required confirmatory sample count.
    """
    manifest_path = ROOT / ".omx/research/mvp1-remediation-baseline/experiment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_ids = set(manifest["seed_protocol"]["training_checkpoint_seed_ids"])
    if len(checkpoints) != 10:
        raise ValueError("precision pilot requires all 10 validation-selected confirmatory checkpoints")
    policies, checkpoint_hashes = {}, {}
    for path in checkpoints:
        policy, weights_hash, sidecar_hash, provenance = _validate_checkpoint(path, expected_ids)
        if policy.seed in policies:
            raise ValueError(f"duplicate training seed {policy.seed}")
        policies[policy.seed] = policy
        checkpoint_hashes[str(Path(path).resolve())] = {"training_seed": policy.seed,
                                        "weights_sha256": weights_hash,
                                        "sidecar_sha256": sidecar_hash, **provenance}
    if set(policies) != expected_ids:
        raise ValueError("precision pilot checkpoint cohort must exactly match preregistered seeds 11001-11010")
    pilot_seeds = manifest["seed_protocol"]["independent_precision_pilot_world_seeds"]
    # Only paired primary and ablation probability contrasts are retained in
    # memory long enough to estimate variance components; means are discarded.
    values = {f"{factor}_{endpoint}": [] for factor in ("memory", "perception")
              for endpoint in ("primary", "attenuation")}
    pilot_fidelity = True
    pilot_shams_identical = True
    pilot_invalid_actions = 0
    for seed in pilot_seeds:
        per_seed = {key: [] for key in values}
        for training_seed in sorted(policies):
            policy = policies[training_seed]
            base = prepare_decision_fixture(scenario_world("starter_seed", seed), policy)
            arms = collect_arms(policy, base)
            pilot_fidelity = pilot_fidelity and all(a["intervention_fidelity"] for factor in arms.values() for a in factor.values())
            pilot_shams_identical = pilot_shams_identical and all(arms[f]["sham_success" if f == "memory" else "sham_available"]["bit_identical_to_control"] for f in arms)
            pilot_invalid_actions += sum(a["invalid_actions"] for factor in arms.values() for a in factor.values())
            if not pilot_fidelity or not pilot_shams_identical or pilot_invalid_actions:
                raise RuntimeError("precision pilot aborted: intervention fidelity, sham replay, or action validity failed")
            for factor in ("memory", "perception"):
                if factor == "memory":
                    full = arms[factor]["success"]["p_drink"] - arms[factor]["blocked"]["p_drink"]
                    ablated = arms[factor]["success_ablation"]["p_drink"] - arms[factor]["blocked_ablation"]["p_drink"]
                else:
                    full = arms[factor]["available"]["p_drink"] - arms[factor]["masked"]["p_drink"]
                    ablated = arms[factor]["available_ablation"]["p_drink"] - arms[factor]["masked_ablation"]["p_drink"]
                per_seed[f"{factor}_primary"].append(float(full))
                per_seed[f"{factor}_attenuation"].append(float(full - ablated))
        for key in values:
            values[key].append(per_seed[key])
    components = {key: _variance_components(np.asarray(matrix, dtype=float).T)
                  for key, matrix in values.items()}
    plans = {key: _required_world_count(value) for key, value in components.items()}
    required_n = max(plan["required_world_seeds"] for plan in plans.values())
    base_locked = manifest["seed_protocol"]["locked_test_world_seeds_by_family"]["primary_starter_14x10"]
    reserves = manifest["seed_protocol"]["predeclared_precision_reserve_world_seeds"]["primary_starter_14x10"]
    selected = list(base_locked[:min(required_n, len(base_locked))])
    if required_n > len(base_locked):
        selected.extend(reserves[:required_n - len(base_locked)])
    test_path = ROOT / "tests/test_cognition_counterfactuals.py"
    decision = {
        "decision_type": "pre-locked causal precision/sample-count decision; excluded-seed variance pilot only",
        "manifest_sha256": manifest_hash,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "checkpoint_hashes": checkpoint_hashes,
        "training_summaries": {item["training_summary_path"]: item["training_summary_sha256"] for item in checkpoint_hashes.values()},
        "training_condition": "encoder_generated_wider_support_current_objective",
        "pilot_seed_count": len(pilot_seeds),
        "pilot_seed_ids": pilot_seeds,
        "pilot_seed_sha256": hashlib.sha256(json.dumps(pilot_seeds, separators=(",", ":")).encode()).hexdigest(),
        "pilot_seed_first_last": [pilot_seeds[0], pilot_seeds[-1]],
        "pilot_data_used_for": "episode-level paired endpoint variance and checkpoint/world-seed correlation only; no policy/intervention mean, sign, ranking, or favorable-contrast report",
        "pilot_interventions_100pct_faithful": pilot_fidelity,
        "pilot_shams_bit_identical": pilot_shams_identical,
        "pilot_invalid_actions": pilot_invalid_actions,
        "variance_components_and_iccs": components,
        "planned_precision": plans,
        "required_world_seed_count_per_factor": required_n,
        "selected_locked_world_seeds": {"starter_seed": selected},
        "reserve_seed_count_activated": max(0, required_n - len(base_locked)),
        "precision_decision": "PLANNED_N_MET_TARGET" if all(p["precision_attainable_with_approved_reserves"] for p in plans.values()) else "INCONCLUSIVE_MAX_APPROVED_SEEDS_STILL_UNDERPOWERED",
        "minimum_checkpoints": 10,
        "target_95_ci_half_width": 0.025,
        "no_locked_test_outcomes_accessed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision

def _validate_precision_decision(decision, manifest, *, manifest_hash, harness_hash, test_hash,
                                 checkpoint_hashes, selected_seeds):
    if decision.get("harness_sha256") != harness_hash or decision.get("manifest_sha256") != manifest_hash:
        raise ValueError("frozen pilot decision does not match current harness/manifest hash")
    if decision.get("checkpoint_hashes") != checkpoint_hashes:
        raise ValueError("frozen pilot decision checkpoint/selection provenance does not match this cohort")
    expected_summaries = {item["training_summary_path"]: item["training_summary_sha256"]
                          for item in checkpoint_hashes.values()}
    if decision.get("training_summaries") != expected_summaries:
        raise ValueError("frozen pilot decision training-summary provenance does not match current checkpoints")
    if decision.get("test_sha256") != test_hash:
        raise ValueError("frozen pilot decision test hash does not match current test suite")
    if (decision.get("training_condition") != "encoder_generated_wider_support_current_objective" or
            int(decision.get("minimum_checkpoints", 0)) < 10):
        raise ValueError("precision decision lacks the approved checkpoint condition/sample minimum")
    pilot_seeds = manifest["seed_protocol"]["independent_precision_pilot_world_seeds"]
    expected_pilot_hash = hashlib.sha256(json.dumps(pilot_seeds, separators=(",", ":")).encode()).hexdigest()
    if (decision.get("pilot_seed_ids") != pilot_seeds or decision.get("pilot_seed_sha256") != expected_pilot_hash or
            decision.get("no_locked_test_outcomes_accessed") is not True):
        raise ValueError("precision decision does not identify the frozen excluded pilot seeds/protocol")
    n = int(decision.get("required_world_seed_count_per_factor", 0))
    base = manifest["seed_protocol"]["locked_test_world_seeds_by_family"]["primary_starter_14x10"]
    reserve = manifest["seed_protocol"]["predeclared_precision_reserve_world_seeds"]["primary_starter_14x10"]
    if not 30 <= n <= 60:
        raise ValueError("precision decision sample count is outside approved 30-60 seed range")
    expected_selected = list(base[:min(n, 30)]) + (list(reserve[:max(0, n - 30)]))
    if decision.get("selected_locked_world_seeds") != {"starter_seed": expected_selected}:
        raise ValueError("precision decision selected seeds must be base 41001-41030 plus only the first authorized reserve IDs")
    if list(selected_seeds) != expected_selected:
        raise ValueError("requested locked world seeds differ from exact pilot-authorized sample decision")
    if decision.get("precision_decision") not in {"PLANNED_N_MET_TARGET", "INCONCLUSIVE_MAX_APPROVED_SEEDS_STILL_UNDERPOWERED"}:
        raise ValueError("precision decision status is missing or invalid")
    return decision["precision_decision"] == "PLANNED_N_MET_TARGET"


def run(checkpoints, scenarios, seeds, output, decision_file=None):
    # Freeze and validate every provenance boundary before opening a rollout.
    manifest_path = ROOT / ".omx/research/mvp1-remediation-baseline/experiment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    test_path = ROOT / "tests/test_cognition_counterfactuals.py"
    test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    expected_training_ids = set(manifest["seed_protocol"]["training_checkpoint_seed_ids"])
    checkpoint_hashes = {}
    training_ids_seen = set()
    policies = {}
    for checkpoint in checkpoints:
        policy, checkpoint_hash, sidecar_hash, provenance = _validate_checkpoint(checkpoint, expected_training_ids)
        if policy.seed in training_ids_seen:
            raise ValueError(f"duplicate checkpoint training seed {policy.seed}")
        training_ids_seen.add(policy.seed)
        policies[Path(checkpoint)] = policy
        checkpoint_hashes[str(Path(checkpoint).resolve())] = {"weights_sha256": checkpoint_hash,
                                              "sidecar_sha256": sidecar_hash,
                                              "training_seed": policy.seed, **provenance}
    cohort_verified = training_ids_seen == expected_training_ids
    if not cohort_verified:
        raise ValueError("causal confirmatory run requires exactly the 10 preregistered validation-selected checkpoints")

    locked_verified = (isinstance(seeds, dict) and scenarios == ["starter_seed"] and
                       all(seed in manifest["seed_protocol"]["locked_test_world_seeds_by_family"]["primary_starter_14x10"] +
                           manifest["seed_protocol"]["predeclared_precision_reserve_world_seeds"]["primary_starter_14x10"]
                           for seed in seeds.get("starter_seed", [])))
    decision = json.loads(Path(decision_file).read_text()) if decision_file else None
    precision_plan_authorizes_gate = False
    if locked_verified:
        if decision is None:
            raise ValueError("refusing confirmatory locked outcomes before a frozen precision decision file")
        precision_plan_authorizes_gate = _validate_precision_decision(
            decision, manifest, manifest_hash=manifest_hash, harness_hash=script_hash,
            test_hash=test_hash, checkpoint_hashes=checkpoint_hashes,
            selected_seeds=seeds["starter_seed"])
    elif decision is not None:
        raise ValueError("pilot decision file only authorizes its exact frozen primary starter seed cohort")

    rows = []
    for checkpoint in checkpoints:
        policy = policies[Path(checkpoint)]
        training_seed = policy.seed
        checkpoint_hash = checkpoint_hashes[str(Path(checkpoint).resolve())]["weights_sha256"]
        sidecar_hash = checkpoint_hashes[str(Path(checkpoint).resolve())]["sidecar_sha256"]
        for scenario in scenarios:
            scenario_seeds = seeds[scenario] if isinstance(seeds, dict) else seeds
            for seed in scenario_seeds:
                base = prepare_decision_fixture(scenario_world(scenario, seed), policy)
                cell = {"checkpoint": str(checkpoint), "training_seed": training_seed, "checkpoint_sha256": checkpoint_hash,
                        "checkpoint_sidecar_sha256": sidecar_hash, "world_seed": seed,
                        "environment_family": scenario, "memory": {}, "perception": {}}
                arms = collect_arms(policy, base)
                cell["memory"] = arms["memory"]
                cell["perception"] = arms["perception"]
                rows.append(cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    locked_verified = bool(locked_verified)
    summary = {
        "protocol": "approved MVP1 P4 confirmatory memory and perception interventions with matched feature-off ablations",
        "manifest_sha256": manifest_hash, "harness_sha256": script_hash,
        "frozen_precision_decision_sha256": hashlib.sha256(Path(decision_file).read_bytes()).hexdigest() if decision_file else None,
        "checkpoint_hashes": checkpoint_hashes,
        "locked_seed_sets_verified": locked_verified,
        "frozen_precision_decision_verified": decision is not None and locked_verified,
        "precision_plan_authorizes_gate": precision_plan_authorizes_gate,
        "confirmatory_checkpoint_cohort_verified": cohort_verified,
        "training_condition_required": "encoder_generated_wider_support_current_objective",
        "scope": "synthetic software cognition only; not biological validation",
        "decision_tick": DECISION_TICK, "fixture_memory_event_age": DECISION_TICK,
        "horizon_ticks_after_intervention": HORIZON, "synthetic_days_after_intervention": HORIZON // 24,
        "checkpoint_count": len(checkpoints),
        "world_seed_count_per_family": {name: len(seeds[name]) for name in scenarios} if isinstance(seeds, dict) else len(seeds),
        "family_count": len(scenarios), "paired_cells": len(rows),
        "run_classification": "confirmatory_locked_primary" if (locked_verified and cohort_verified and precision_plan_authorizes_gate) else "exploratory_no_causal_pass",
        "required_confirmatory_cells": 10 * (len(seeds["starter_seed"]) if locked_verified else 30) if isinstance(seeds, dict) else 0,
        "adequate_minimum_sample": (cohort_verified and locked_verified and all(len(seeds[name]) >= 30 for name in scenarios)
                                    if isinstance(seeds, dict) else False),
        "all_interventions_faithful": all(arm["intervention_fidelity"] for row in rows for factor in ("memory", "perception") for arm in row[factor].values()),
        "all_shams_bit_identical": all(row[factor]["sham_success" if factor == "memory" else "sham_available"]["bit_identical_to_control"] for row in rows for factor in ("memory", "perception")),
        "invalid_actions": sum(arm["invalid_actions"] for row in rows for factor in ("memory", "perception") for arm in row[factor].values()),
        "causal_estimates_by_family": paired_stats(rows, gate_authorized=(locked_verified and cohort_verified and precision_plan_authorizes_gate)),
        "paired_rows": output.name,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="trained local .npz; repeat for independent checkpoints")
    parser.add_argument("--scenario", action="append", choices=("starter_seed", "wide_grid_20x14", "resource_scarce_relocated", "obstacle_topology"), default=None)
    parser.add_argument("--seed", action="append", type=int, default=None, help="world seed; repeat; default locked primary seeds 41001..41030")
    parser.add_argument("--output", type=Path, required=True, help="pilot decision JSON or paired JSONL output")
    parser.add_argument("--pilot-only", action="store_true", help="run excluded-seed variance pilot and freeze sample count; emits no pilot effect means")
    parser.add_argument("--decision-file", type=Path, help="frozen pilot decision required for confirmatory rollouts")
    args = parser.parse_args()
    if args.pilot_only:
        decision = pilot_precision_decision([Path(p) for p in args.checkpoint], args.output)
        print(json.dumps({"decision_file": str(args.output), **decision}, indent=2, sort_keys=True))
        return
    scenarios = args.scenario or ["starter_seed"]
    if args.seed:
        # Explicit custom seeds are useful for tests/pilots but not locked-test
        # confirmation unless they exactly equal a single family's manifest list.
        seeds = {scenario: list(args.seed) for scenario in scenarios}
    else:
        manifest = json.loads((ROOT / ".omx/research/mvp1-remediation-baseline/experiment-manifest.json").read_text())
        locked = manifest["seed_protocol"]["locked_test_world_seeds_by_family"]
        manifest_names = {"starter_seed": "primary_starter_14x10",
                          "wide_grid_20x14": "wide_grid_20x14",
                          "resource_scarce_relocated": "resource_scarce_relocated"}
        if args.decision_file and scenarios == ["starter_seed"]:
            frozen_decision = json.loads(args.decision_file.read_text())
            seeds = {"starter_seed": frozen_decision["selected_locked_world_seeds"]["starter_seed"]}
        else:
            seeds = {scenario: locked[manifest_names[scenario]] for scenario in scenarios if scenario in manifest_names}
        if "obstacle_topology" in scenarios:
            raise SystemExit("obstacle_topology has no preregistered locked seed list; specify custom --seed and label exploratory")
    summary, path = run([Path(p) for p in args.checkpoint], scenarios, seeds, args.output,
                        decision_file=args.decision_file)
    print(json.dumps({"summary_path": str(path), **summary}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
