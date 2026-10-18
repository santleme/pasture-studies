#!/usr/bin/env python3
"""Instrumented CPU profile of simulator phases and scalar/batched collection.

This reports observed wall-clock timings for a reproducible synthetic workload.
The collector comparison reports descriptive scalar-per-animal versus
multi-world-batched inference timing and parity for a fixed synthetic workload.
It does not provide CPU/GPU comparisons. Phase timings are wrapper-level
measurements and add instrumentation overhead.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import farm_sim.simulation as sim_module
from farm_sim.model import Action
from farm_sim.policy import CategoricalPolicy, load_checkpoint, save_checkpoint
from farm_sim.simulation import Simulation, make_world, serialize_world


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar_torch_episode(model, world_seed: int, *, root_seed: int, episode_index: int):
    """Independent scalar-inference collector using the batch collector's uniform stream."""
    from farm_sim.model import Action, PolicyDecision
    from farm_sim.simulation import Simulation
    from farm_sim.torch_training import (ACTIONS, N_ACTIONS, N_INPUTS, Trajectory,
                                         derive_seed, require_torch, validate_probabilities)

    torch = require_torch()
    uniforms = np.asarray([
        [np.random.default_rng(derive_seed(root_seed, "action-uniform", world_seed,
                                           episode_index, tick, animal)).random()
         for animal in range(5)] for tick in range(72)
    ], dtype=np.float64)

    class ScalarPolicy:
        def __init__(self):
            self.index = 0
            self.probabilities = []

        def decide(self, features):
            row = np.asarray(features, dtype=np.float32)
            with torch.no_grad():
                p = model.probabilities(torch.as_tensor(row, dtype=torch.float32,
                                                          device=next(model.parameters()).device))
            probs = validate_probabilities(p.detach().cpu().numpy().reshape(-1))
            u = float(uniforms.flat[self.index])
            idx = min(N_ACTIONS - 1, int(np.searchsorted(np.cumsum(probs), u, side="right")))
            self.index += 1
            self.probabilities.append(probs)
            return PolicyDecision(ACTIONS[idx], tuple(float(v) for v in probs), "torch-sampled-batched")

    policy = ScalarPolicy()
    sim = Simulation(make_world(world_seed), policy)
    observations = np.empty((72, 5, N_INPUTS), dtype=np.float32)
    actions = np.empty((72, 5), dtype=np.int64)
    rewards = np.empty((72, 5), dtype=np.float32)
    for tick in range(72):
        rows = sim.decision_features()
        observations[tick] = np.asarray([features for _, features in rows], dtype=np.float32)
        result = sim.step()
        for animal_position, animal in enumerate(sorted(result.world.animals, key=lambda a: a.animal_id)):
            actions[tick, animal_position] = ACTIONS.index(animal.action)
            rewards[tick, animal_position] = animal.last_reward
    counts = {}
    for event in sim.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    trajectory = Trajectory(observations, actions, rewards, uniforms, world_seed,
                            np.asarray(policy.probabilities, dtype=np.float32).reshape(72, 5, N_ACTIONS), counts, 0)
    return trajectory, sim.world, tuple(sim.events)


def benchmark_collectors(*, seeds: tuple[int, ...] = (51337, 51338, 51339),
                         root_seed: int = 99173, repeats: int = 3, model_seed: int = 29) -> dict:
    """Compare scalar per-animal inference and multi-world batched inference on CPU."""
    from farm_sim.simulation import Simulation
    from farm_sim.torch_training import ActorCritic, collect_episodes_batch, require_torch

    if not seeds or repeats < 1:
        raise ValueError("at least one seed and a positive repeat count are required")
    torch = require_torch()
    if torch.cuda.is_available():
        # This function remains explicitly CPU-only even on a GPU host.
        device = "cpu"
    else:
        device = "cpu"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    specs = [(int(seed), index) for index, seed in enumerate(seeds)]
    model = ActorCritic(value_head=False, seed=model_seed).to(device).eval()
    scalar_samples, batch_samples = [], []
    try:
        # Warm up both execution paths on the same deterministic first seed.
        _scalar_torch_episode(model, specs[0][0], root_seed=root_seed, episode_index=specs[0][1])
        collect_episodes_batch(model, [specs[0]], root_seed=root_seed)
        for repeat in range(repeats):
            # Reverse order every other repeat to reduce simple order/cache bias.
            order = ("scalar", "batched") if repeat % 2 == 0 else ("batched", "scalar")
            for mode in order:
                start = time.perf_counter()
                if mode == "scalar":
                    for seed, index in specs:
                        _scalar_torch_episode(model, seed, root_seed=root_seed, episode_index=index)
                    scalar_samples.append(time.perf_counter() - start)
                else:
                    collect_episodes_batch(model, specs, root_seed=root_seed)
                    batch_samples.append(time.perf_counter() - start)

        scalar_rows = [_scalar_torch_episode(model, seed, root_seed=root_seed, episode_index=index)
                       for seed, index in specs]
        captured_worlds = {}
        captured_events = {}
        original_commit = Simulation.step_with_decisions

        def capture_commit(sim, decisions):
            result = original_commit(sim, decisions)
            if result.world.tick == 72:
                captured_worlds[result.world.seed] = result.world
                captured_events[result.world.seed] = tuple(sim.events)
            return result

        with patch.object(Simulation, "step_with_decisions", capture_commit):
            batch_rows = collect_episodes_batch(model, specs, root_seed=root_seed)
        parity = []
        for spec, scalar, batch in zip(specs, scalar_rows, batch_rows):
            seed, _ = spec
            scalar_traj, scalar_world, scalar_events = scalar
            exact_fields = ("observations", "actions", "rewards", "uniforms")
            exact = {field: bool(np.array_equal(getattr(scalar_traj, field), getattr(batch, field)))
                     for field in exact_fields}
            prob_max_abs = float(np.max(np.abs(scalar_traj.probabilities - batch.probabilities)))
            world_equal = (seed in captured_worlds and
                           replace_world_probabilities(scalar_world) == replace_world_probabilities(captured_worlds[seed]))
            events_equal = scalar_events == captured_events.get(seed)
            parity.append({"seed": seed, "trajectory_fields_exact": exact,
                           "probabilities_max_abs_difference": prob_max_abs,
                           "probabilities_allclose_rtol_1e-6_atol_1e-7": bool(np.allclose(
                               scalar_traj.probabilities, batch.probabilities, rtol=1e-6, atol=1e-7)),
                           "final_world_state_exact_except_probability_telemetry": bool(world_equal),
                           "events_exact": bool(events_equal),
                           "all_parity_checks_pass": bool(all(exact.values()) and world_equal and events_equal and
                                                            np.allclose(scalar_traj.probabilities, batch.probabilities,
                                                                        rtol=1e-6, atol=1e-7))})
        decisions = len(specs) * 72 * 5
        def stats(samples):
            median = float(statistics.median(samples))
            return {"samples_seconds": samples, "median_seconds": median,
                    "p95_seconds": float(np.percentile(samples, 95)),
                    "decisions": decisions,
                    "decisions_per_second_at_median": float(decisions / median) if median else None}
        return {"workload": {"seeds": [s for s, _ in specs], "episode_indices": [i for _, i in specs],
                              "root_seed": root_seed, "model_seed": model_seed,
                              "ticks_per_episode": 72, "animals_per_episode": 5,
                              "episodes_per_repeat": len(specs), "decisions_per_repeat": decisions,
                              "device": "cpu", "torch_threads": 1},
                "scalar_per_animal_inference": stats(scalar_samples),
                "batched_multiworld_inference": stats(batch_samples),
                "paired_parity": parity,
                "parity_pass": all(row["all_parity_checks_pass"] for row in parity),
                "interpretation": "Descriptive CPU collector profile only. Timing includes policy inference and simulator collection; not CUDA evidence, not a learning comparison, and not a promotion/acceptance result."}
    finally:
        torch.set_num_threads(old_threads)


def replace_world_probabilities(world):
    """Ignore only float probability telemetry; require exact physical/world state."""
    from dataclasses import replace
    return replace(world, animals=tuple(replace(animal, last_probabilities=()) for animal in world.animals))


class Timings:
    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)

    def wrap(self, name: str, function: Callable):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.samples[name].append(time.perf_counter() - start)
        return measured

    def summarize(self) -> dict:
        return {name: {
            "samples_seconds": values,
            "calls": len(values),
            "total_seconds": float(sum(values)),
            "median_seconds": float(statistics.median(values)) if values else 0.0,
            "p95_seconds": float(np.percentile(values, 95)) if values else 0.0,
        } for name, values in sorted(self.samples.items())}


def profile(*, seed: int = 51337, ticks: int = 72, repeats: int = 3, output: Path | None = None) -> dict:
    if ticks < 1 or repeats < 1:
        raise ValueError("ticks and repeats must be positive")
    timing = Timings()
    patches = {
        "perception": (sim_module, "perceive"),
        "memory_retrieval": (sim_module, "retrieve"),
        "feature_encoding": (sim_module, "encode_features"),
        "navigation_arbitration": (sim_module, "navigate"),
        "physiology_reward": (sim_module, "step_physiology"),
    }
    originals = {name: getattr(module, attr) for name, (module, attr) in patches.items()}
    for name, (module, attr) in patches.items():
        setattr(module, attr, timing.wrap(name, originals[name]))

    phases: dict[str, list[float]] = defaultdict(list)
    checkpoints: list[dict] = []
    try:
        for repeat in range(repeats):
            world_start = time.perf_counter()
            world = make_world(seed)
            phases["world_construction"].append(time.perf_counter() - world_start)
            policy = CategoricalPolicy.initialize(seed)
            inference_original = policy.decide
            policy.decide = timing.wrap("policy_inference", inference_original)
            sim = Simulation(world, policy)
            rollout_start = time.perf_counter()
            for _ in range(ticks):
                tick_start = time.perf_counter()
                sim.step()
                phases["simulator_step_inclusive"].append(time.perf_counter() - tick_start)
            phases["full_rollout"].append(time.perf_counter() - rollout_start)

            serial_start = time.perf_counter()
            serialized = serialize_world(sim.world)
            phases["world_serialization"].append(time.perf_counter() - serial_start)
            # Measure encoding and file write separately from simulator work.
            log_payload = {"seed": seed, "repeat": repeat, "world": serialized}
            serialize_start = time.perf_counter()
            encoded = json.dumps(log_payload, sort_keys=True, separators=(",", ":"))
            phases["json_log_serialization"].append(time.perf_counter() - serialize_start)
            if output is not None:
                log_path = output.with_name(f"{output.stem}.run-{repeat}.json")
                write_start = time.perf_counter()
                log_path.write_text(encoded + "\n")
                phases["json_log_file_write"].append(time.perf_counter() - write_start)
                log_path.unlink(missing_ok=True)

            # Checkpoint storage is an optional policy-artifact I/O sample, not a
            # training checkpoint (there is no optimizer/trainer state here).
            if output is not None:
                checkpoint_path = output.with_name(f"{output.stem}.policy-{repeat}.npz")
                checkpoint_start = time.perf_counter()
                save_checkpoint(policy, checkpoint_path, metadata={"profile_seed": seed, "repeat": repeat})
                save_checkpoint_seconds = time.perf_counter() - checkpoint_start
                load_start = time.perf_counter()
                load_checkpoint(checkpoint_path)
                load_seconds = time.perf_counter() - load_start
                checkpoints.append({"save_seconds": save_checkpoint_seconds, "load_seconds": load_seconds})
                phases["policy_checkpoint_save_npz_and_sidecar"].append(save_checkpoint_seconds)
                phases["policy_checkpoint_load_npz_and_sidecar"].append(load_seconds)
                checkpoint_path.unlink(missing_ok=True)
                checkpoint_path.with_suffix(".json").unlink(missing_ok=True)
    finally:
        for name, (module, attr) in patches.items():
            setattr(module, attr, originals[name])

    source_files = [Path(__file__), ROOT / "farm_sim/torch_training.py", ROOT / "farm_sim/simulation.py", ROOT / "farm_sim/navigation.py",
                    ROOT / "farm_sim/perception.py", ROOT / "farm_sim/memory.py",
                    ROOT / "farm_sim/physiology.py", ROOT / "farm_sim/policy.py"]
    timings = {}
    for name, samples in sorted(phases.items()):
        timings[name] = {"samples_seconds": samples, "calls": len(samples),
                         "total_seconds": float(sum(samples)),
                         "median_seconds": float(statistics.median(samples)),
                         "p95_seconds": float(np.percentile(samples, 95))}
    timings.update(timing.summarize())
    try:
        collector_comparison = benchmark_collectors()
    except RuntimeError as exc:
        collector_comparison = {"status": "unavailable", "reason": str(exc),
                                "interpretation": "CPU scalar-vs-batched collector timings not measured because optional PyTorch is unavailable."}
    doc = {
        "suite": "scalar-simulator-phase-profile-v1",
        "configuration": {"seed": seed, "ticks_per_repeat": ticks, "repeats": repeats,
                           "animals": 5, "action_policy": "deterministic untrained NumPy CategoricalPolicy",
                           "simulator_semantics": "unchanged scalar Simulation.step; one episode-like rollout"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "numpy": np.__version__, "device": "cpu", "cuda_tested": False},
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in source_files},
        "timings": timings,
        "collector_comparison": collector_comparison,
        "checkpoint_samples": checkpoints,
        "coverage_and_limits": {
            "measured": ["world construction", "perception", "memory retrieval", "feature encoding",
                         "policy inference", "navigation/arbitration", "physiology/reward", "simulator step",
                         "full rollout", "world/JSON serialization", "optional policy checkpoint save/load"],
            "not_measured": ["gradient training/update", "validation/evaluation workload", "trainer checkpoint recovery",
                             "GPU execution/utilization/memory", "training-quality or acceptance conclusions"],
            "phase_timing_note": "Wrapper-level CPU wall timings; instrumentation overhead is included in inclusive full-rollout/step timings, while phase sums overlap inclusively.",
            "comparison_note": "Scalar-per-animal and multi-world-batched CPU collection are compared descriptively with paired parity checks; this does not establish GPU speedup or policy quality.",
            "checkpoint_note": "Checkpoint samples use the local inference-policy NPZ+JSON bridge, not resumable optimizer/trainer generations.",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Save timing is intentionally external to the report's own JSON timing.
        start = time.perf_counter()
        output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        first_write_seconds = time.perf_counter() - start
        doc["timings"]["profile_report_json_write"] = {"samples_seconds": [first_write_seconds], "calls": 1,
                                                       "total_seconds": first_write_seconds,
                                                       "median_seconds": first_write_seconds,
                                                       "p95_seconds": first_write_seconds,
                                                       "note": "Measured first report write; subsequent self-inclusive write is not timed."}
        output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=51337)
    parser.add_argument("--ticks", type=int, default=72)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/profile-training-pipeline.json"))
    args = parser.parse_args(argv)
    result = profile(seed=args.seed, ticks=args.ticks, repeats=args.repeats, output=args.output)
    print(json.dumps({"output": str(args.output), "configuration": result["configuration"],
                      "timings": {k: {"total_seconds": v["total_seconds"], "calls": v["calls"]}
                                  for k, v in result["timings"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
