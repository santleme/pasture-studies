"""Reproducible optional Torch trajectory REINFORCE/A2C CLI (synthetic task only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np

from farm_sim.model import Action, FEATURE_NAMES
from farm_sim.policy import CategoricalPolicy, load_checkpoint
from farm_sim.exploratory_learning import calculate_exploratory_loss
from farm_sim.simulation import Simulation, make_world
from farm_sim.torch_checkpoint import load_generation, resolve_generation, save_generation
from farm_sim.torch_training import (
    ACTIONS, ActorCritic, ROLLOUT_TICKS, calculate_loss, collect_episode, collect_episodes_batch,
    derive_seed, export_actor_npz, intervention_probabilities, require_torch,
    resolve_device, set_deterministic, shard_episode_indices,
    InvalidPolicyOutput, assert_finite_training_state,
)

BATCH_EPISODES = 16
GRAD_CLIP_NORM = 2.0
LEARNING_RATE = 1e-3
CONFIG_VERSION = "trajectory-v1"
READINESS_EPISODES = 32
READINESS_UPDATES = 2
READINESS_BATCH_EPISODES = 16
EXPLORATORY_A2C_SEED = 9482817
EXPLORATORY_A2C_EPISODES = 256
EXPLORATORY_A2C_EVAL_EPISODES = 5
EXPLORATORY_BATCH_EPISODES = 512
EXPLORATORY_BATCH_EVAL_EPISODES = 5
EXPLORATORY_BATCH_STATE_REWARD_SCALE = 0.1
EXPLORATORY_BATCH_DISCOUNT_FACTOR = 0.99
EXPLORATORY_BATCH_SEEDS = (9483001, 9483002, 9483003)
EXPLORATORY_BATCH_MAX_JOBS = 24
EXPLORATORY_BATCH_VARIANTS = {
    "A": {"reward_variant": "original", "state_reward_scale": 0.1,
          "normalize_advantage": False, "entropy_coefficient": 0.0, "discount_factor": 0.99},
    "B": {"reward_variant": "original", "state_reward_scale": 0.1,
          "normalize_advantage": True, "entropy_coefficient": 0.01, "discount_factor": 0.99},
    "C": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
          "normalize_advantage": False, "entropy_coefficient": 0.0, "discount_factor": 0.99},
    "D": {"reward_variant": "state_aware", "state_reward_scale": 0.1,
          "normalize_advantage": True, "entropy_coefficient": 0.01, "discount_factor": 0.99},
}


def exploratory_batch_specs() -> list[dict]:
    """Return the fixed, paired nonformal 4-variant x 3-seed workload."""
    return [{"variant": variant, **EXPLORATORY_BATCH_VARIANTS[variant], "seed": seed,
             "episodes": EXPLORATORY_BATCH_EPISODES}
            for variant in EXPLORATORY_BATCH_VARIANTS for seed in EXPLORATORY_BATCH_SEEDS]


def profile_exploratory_step(*, device: str, output: Path, repeats: int = 2) -> dict:
    """Measure one real 16-episode rollout/forward/backward batch; never step/save."""
    torch = require_torch()
    if device not in {"cpu", "cuda:0", "cuda:1"}:
        raise ValueError("calibration device must be cpu, cuda:0, or cuda:1")
    if device.startswith("cuda") and (not torch.cuda.is_available()
            or int(device.split(":")[1]) >= torch.cuda.device_count()):
        raise RuntimeError(f"calibration device {device} is unavailable")
    if repeats not in {1, 2}:
        raise ValueError("calibration uses one or two complete episode batches")
    torch_device = torch.device(device)
    model = ActorCritic(value_head=True, seed=EXPLORATORY_BATCH_SEEDS[0], dtype=torch.float32).to(torch_device)
    model.eval()
    rows = []
    for repeat in range(repeats):
        root_seed = EXPLORATORY_BATCH_SEEDS[0]
        specs = [(derive_seed(root_seed, "profile-world", i), i) for i in range(BATCH_EPISODES)]
        start = time.perf_counter()
        trajectories = collect_episodes_batch(model, specs, root_seed=root_seed)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - start
        start = time.perf_counter()
        loss, stats = calculate_exploratory_loss(model, trajectories, "a2c",
            reward_variant="original", state_reward_scale=EXPLORATORY_BATCH_STATE_REWARD_SCALE,
            entropy_coefficient=0.0, normalize_advantage=False)
        loss.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        forward_backward_seconds = time.perf_counter() - start
        rows.append({"repeat": repeat, "episodes": BATCH_EPISODES,
                     "decisions": BATCH_EPISODES * ROLLOUT_TICKS * 5,
                     "rollout_seconds": rollout_seconds,
                     "forward_backward_seconds": forward_backward_seconds,
                     "batch_seconds": rollout_seconds + forward_backward_seconds,
                     "mean_environment_reward": float(sum(x.rewards.sum() for x in trajectories)
                                                       / BATCH_EPISODES),
                     "mean_training_target_return_to_go": stats["mean_return_to_go"],
                     "optimizer_step": False, "checkpoint_write": False})
        model.zero_grad(set_to_none=True)
    report = {"schema_version": 1, "profile": "exploratory-real-rollout-forward-backward-v1",
              "device": str(torch_device), "torch_version": torch.__version__,
              "episodes_per_batch": BATCH_EPISODES, "ticks_per_episode": ROLLOUT_TICKS,
              "animals_per_episode": 5, "repeat_count": repeats,
              "timing_scope": "simulator collection incl. policy inference; then loss forward/backward",
              "learning_update": False, "optimizer_step": False, "checkpoint_write": False,
              "formal_eligible": False, "benchmark_eligible": False, "locked_outcomes_accessed": False,
              "rows": rows, "total_seconds": sum(row["batch_seconds"] for row in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def _metrics_write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def _model_digest(model) -> str:
    """Hash model tensors in a stable name/order representation."""
    digest = hashlib.sha256()
    target = model.module if hasattr(model, "module") else model
    for name, value in sorted(target.state_dict().items()):
        tensor = value.detach().contiguous().cpu().numpy()
        digest.update(name.encode("utf8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def run_readiness_ddp(algorithm: str, *, seed: int, output: Path, device: str,
                      resume: str | None = None, max_updates: int | None = None) -> dict:
    """Bounded diagnostic DDP exercise; deliberately emits no experiment metrics/eval.

    The command-line entry point requires CUDA/NCCL. CPU/Gloo is supported here only
    so CI can execute the same collect/gradient/checkpoint/recovery path without GPUs.
    """
    torch = require_torch()
    if algorithm not in {"reinforce", "a2c"}:
        raise ValueError("readiness DDP requires one algorithm: reinforce or a2c")
    if max_updates not in {None, 1}:
        raise ValueError("readiness DDP allows at most one update per invocation")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("readiness DDP requires an initialized process group")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise ValueError("readiness DDP requires exactly two ranks")
    device = str(device)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for readiness DDP but unavailable")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif device != "cpu":
        raise ValueError("readiness DDP device must be CPU (test only) or CUDA")

    set_deterministic(seed, device)
    run_dir = output / "readiness-only" / algorithm
    config = {"protocol": "readiness-ddp-v1", "algorithm": algorithm, "seed": int(seed),
              "episodes": READINESS_EPISODES, "batch_episodes": READINESS_BATCH_EPISODES,
              "updates": READINESS_UPDATES, "world_size": 2, "device_class": "cuda",
              "evaluation": False, "locked_split_access": False, "metrics_promotion": False}
    base_model = ActorCritic(value_head=algorithm == "a2c", seed=seed, dtype=torch.float32).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        base_model, device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    state = {"generation": 0, "update": 0, "episode_cursor": 0,
             "baseline_return_sum": 0.0, "baseline_return_count": 0,
             "readiness_only": True, "benchmark_eligible": False}
    if resume:
        restored = load_generation(resume if resume != "latest" else run_dir,
                                   model=model, optimizer=optimizer, map_location=device,
                                   expected_algorithm=algorithm, expected_config=config)
        state = restored["state"]
        if state.get("readiness_only") is not True or state.get("benchmark_eligible") is not False:
            raise ValueError("checkpoint is not readiness-only")
    start_update = int(state["update"])
    if start_update not in {0, 1} or int(state["episode_cursor"]) != start_update * READINESS_BATCH_EPISODES:
        raise ValueError("readiness checkpoint cursor must be at update 0 or 1 boundary")
    stop_update = READINESS_UPDATES if max_updates is None else min(READINESS_UPDATES, start_update + max_updates)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "readiness-evidence.json"
    if rank == 0:
        if start_update == 0 and evidence_path.exists():
            evidence = {"error": f"readiness output already exists: {evidence_path}"}
        else:
            evidence = {"protocol": "readiness-ddp-v1", "algorithm": algorithm, "seed": int(seed),
                        "episodes": READINESS_EPISODES, "world_size": 2,
                        "episode_seed_derivation": "sha256(root,train-world,episode-index)",
                        "benchmark_eligible": False, "evaluation_performed": False,
                        "locked_split_accessed": False, "updates": []}
    else:
        evidence = None
    if rank == 0 and resume:
        # Keep recovery evidence in a separate readiness-only document; it is not
        # produced by experimental evaluation and contains no outcome metrics.
        previous_evidence = run_dir / "readiness-evidence.json"
        if not previous_evidence.is_file():
            evidence = {"error": "readiness evidence missing; refusing an unverifiable resume"}
        else:
            evidence = json.loads(previous_evidence.read_text())
            updates = evidence.get("updates", []) if isinstance(evidence, dict) else None
            if (evidence.get("protocol") != "readiness-ddp-v1" or
                    evidence.get("algorithm") != algorithm or evidence.get("seed") != int(seed) or
                    not isinstance(updates, list) or len(updates) not in {int(state["update"]), int(state["update"]) + 1}):
                evidence = {"error": "readiness evidence does not match checkpoint resume state"}
            elif len(updates) == int(state["update"]) + 1:
                pending = updates[-1]
                if (not isinstance(pending, dict) or pending.get("update") != int(state["update"]) + 1 or
                        pending.get("checkpoint_manifest_sha256") is not None):
                    evidence = {"error": "readiness evidence has an inconsistent uncommitted update"}
                else:
                    # The evidence prepare record is durable before the checkpoint
                    # commit. Drop it when the optimizer/checkpoint update did not commit.
                    updates.pop()
                    previous_evidence.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
            if not evidence.get("error") and int(state["update"]) > 0:
                committed = evidence["updates"][-1]
                if not isinstance(committed, dict):
                    evidence = {"error": "readiness evidence update record is invalid"}
                elif committed.get("checkpoint_manifest_sha256") is None:
                    generation_path = run_dir / f"generation-{int(state['generation']):08d}"
                    manifest_path = generation_path / "manifest.json"
                    if not manifest_path.is_file():
                        evidence = {"error": "committed readiness checkpoint is missing its manifest"}
                    else:
                        checkpoint_manifest = json.loads(manifest_path.read_text())
                        committed["checkpoint_generation"] = generation_path.name
                        committed["checkpoint_manifest_sha256"] = _sha(manifest_path)
                        committed["training_state_sha256"] = checkpoint_manifest["state_sha256"]
                        previous_evidence.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    evidence_holder = [evidence]
    torch.distributed.broadcast_object_list(evidence_holder, src=0)
    evidence = evidence_holder[0]
    if isinstance(evidence, dict) and evidence.get("error"):
        raise ValueError(evidence["error"])

    while int(state["update"]) < stop_update:
        update_index = int(state["update"])
        episode_ids = shard_episode_indices(update_index * READINESS_BATCH_EPISODES,
                                             READINESS_BATCH_EPISODES, world_size, rank)
        episode_specs = [(derive_seed(seed, "train-world", ep), ep) for ep in episode_ids]
        trajectories = collect_episodes_batch(model, episode_specs, root_seed=seed)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = calculate_loss(model, trajectories, algorithm,
            previous_return_sum=state["baseline_return_sum"],
            previous_return_count=state["baseline_return_count"])
        assert_finite_training_state(loss, base_model)
        loss.backward()
        assert_finite_training_state(loss, base_model, gradients=True)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP_NORM)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError("readiness update requires a finite nonzero gradient norm")
        digest_before = _model_digest(base_model)
        optimizer.step()
        assert_finite_training_state(loss.detach(), base_model)
        digest_after = _model_digest(base_model)
        if digest_before == digest_after:
            raise RuntimeError("readiness optimizer step did not change model parameters")
        local_record = {"rank": rank, "episode_ids": episode_ids, "device": device,
                        "cuda_device_index": int(os.environ.get("LOCAL_RANK", rank)) if device.startswith("cuda") else None,
                        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else None,
                        "gradient_norm": grad_norm, "gradient_finite": True,
                        "optimizer_step_changed_model": True, "model_digest": digest_after}
        records = [None, None]
        torch.distributed.all_gather_object(records, local_record)
        # DDP must leave each rank with the same synchronized parameters.
        if len({record["model_digest"] for record in records}) != 1:
            raise RuntimeError("DDP model digests diverged after optimizer step")
        state["update"] += 1
        state["episode_cursor"] += READINESS_BATCH_EPISODES
        state["generation"] += 1
        return_arrays = [x.returns_to_go().reshape(-1) for x in trajectories]
        reward_sum = float(sum(float(x.rewards.sum()) for x in trajectories))
        return_sum = float(sum(float(values.sum()) for values in return_arrays))
        return_count = int(sum(values.size for values in return_arrays))
        local_totals = torch.tensor([reward_sum, return_sum, return_count], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(local_totals)
        global_reward_sum = float(local_totals[0].item())
        state["baseline_return_sum"] += float(local_totals[1].item())
        state["baseline_return_count"] += int(local_totals[2].item())
        update_evidence = {"update": state["update"], "episode_ids_by_rank": records,
                           "global_episode_count": READINESS_BATCH_EPISODES,
                           "diagnostic_reward_sum_not_evaluation": global_reward_sum,
                           "model_digest": digest_after, "benchmark_eligible": False,
                           "checkpoint_generation": f"generation-{int(state['generation']):08d}",
                           "checkpoint_manifest_sha256": None, "training_state_sha256": None}
        if rank == 0:
            evidence["updates"].append(update_evidence)
            # Persist the exact rank/device work record before the checkpoint
            # commit. On restart, an uncommitted prepare row is safely discarded.
            evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
            save_generation(run_dir, model=model, optimizer=optimizer, state=state,
                            config=config, algorithm=algorithm, rank=rank)
            generation_path = run_dir / f"generation-{int(state['generation']):08d}"
            checkpoint_manifest = json.loads((generation_path / "manifest.json").read_text())
            update_evidence["checkpoint_manifest_sha256"] = _sha(generation_path / "manifest.json")
            update_evidence["training_state_sha256"] = checkpoint_manifest["state_sha256"]
            evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
        torch.distributed.barrier()
    if rank == 0:
        result = {"readiness_only": True, "algorithm": algorithm,
                  "checkpoint": str(run_dir / "LATEST"), "evidence": str(evidence_path),
                  "updates": int(state["update"]), "episode_cursor": int(state["episode_cursor"]),
                  "benchmark_eligible": False}
    else:
        result = {"rank": rank, "updates": int(state["update"]), "readiness_only": True}
    torch.distributed.barrier()
    return result


def deterministic_rollout(model, seed: int, *, root_seed: int, episode_index: int):
    """Argmax deployment/evaluation run; emits detailed synthetic outcomes."""
    import torch
    class Greedy:
        def __init__(self): self.invalid_outputs = 0; self.decisions = 0
        def decide(self, features):
            self.decisions += 1
            try:
                return model.decide(features)
            except InvalidPolicyOutput:
                self.invalid_outputs += 1
                raise
    greedy = Greedy()
    sim = Simulation(make_world(seed), greedy)
    actions = {a.value: 0 for a in Action}
    returns = []
    drive_rows = []
    per_animal = {}
    for animal in sim.world.animals:
        per_animal[animal.animal_id] = {"action_counts": {a.value: 0 for a in Action},
                                        "reward_sum": 0.0, "ticks": 0}
    for _ in range(ROLLOUT_TICKS):
        try:
            sim.step()
        except InvalidPolicyOutput:
            break
        for animal in sim.world.animals:
            actions[animal.action.value] += 1
            returns.append(animal.last_reward)
            drive_rows.append((animal.physiology.hunger, animal.physiology.thirst,
                               animal.physiology.fatigue, animal.physiology.social_need))
            row = per_animal[animal.animal_id]
            row["action_counts"][animal.action.value] += 1
            row["reward_sum"] += float(animal.last_reward)
            row["ticks"] += 1
    animal_rows = sim.world.animals
    for animal in animal_rows:
        row = per_animal[animal.animal_id]
        row["final_state"] = {"position": {"x": animal.position.x, "y": animal.position.y},
                              "hunger": animal.physiology.hunger, "thirst": animal.physiology.thirst,
                              "fatigue": animal.physiology.fatigue, "social_need": animal.physiology.social_need,
                              "memory_events": len(animal.memory.events)}
    feature_rows = np.asarray([a.last_features for a in animal_rows], dtype=np.float32)
    influence = intervention_probabilities(model, feature_rows[0]) if sim.world.tick == ROLLOUT_TICKS else None
    resources = {"forage_available": sum(p.available for p in sim.world.forage),
                 "water_available": sum(p.available for p in sim.world.water)}
    return {"seed": int(seed), "return": float(sum(returns)),
            "mean_tick_reward": float(np.mean(returns)) if returns else None,
            "action_counts": actions, "action_frequencies": {k: (v / len(returns) if returns else 0.0) for k,v in actions.items()},
            "invalid_actions": greedy.invalid_outputs, "decision_attempts": greedy.decisions,
            "complete": sim.world.tick == ROLLOUT_TICKS,
            "final_drive_means": np.asarray(drive_rows[-5:]).mean(0).tolist() if drive_rows else [],
            "per_animal": per_animal,
            "resources_remaining": resources, "memory_perception_influence": influence,
            "ticks": sim.world.tick}


def _rng_snapshot(torch):
    numpy_state = np.random.get_state()
    # weights_only checkpoints accept primitive values/tensors, not NumPy ndarray
    # pickle globals; encode MT state as a primitive list and restore explicitly.
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _rng_restore(torch, state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
    torch.set_rng_state(state["torch_cpu"])
    saved_cuda = state.get("torch_cuda", [])
    if torch.cuda.is_available() and saved_cuda and len(saved_cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved_cuda)
    # If device count changed, current process/device seed was set from the root
    # seed above; no stochastic training operation depends on rank-local Torch RNG.


def train_algorithm(algorithm: str, *, seed: int, episodes: int, output: Path,
                    device: str, resume: str | None, metrics_path: Path,
                    checkpoint_every: int, smoke: bool = False, max_updates: int | None = None, eval_count: int = 10,
                    eval_only: bool = False, allow_execution_migration: bool = False, exploratory: bool = False,
                    exploratory_variant: str | None = None, reward_variant: str = "original",
                    state_reward_scale: float = 0.1, normalize_advantage: bool = False,
                    entropy_coefficient: float = 0.0, discount_factor: float = 0.99) -> dict:
    torch = require_torch()
    device = resolve_device(device)
    set_deterministic(seed, device)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if device == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if device == "cuda":
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        if BATCH_EPISODES % world_size or (episodes % BATCH_EPISODES) % world_size:
            raise ValueError("DDP requires equal episode shards in full and tail batches")
    model = ActorCritic(value_head=algorithm == "a2c", seed=seed, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model = model
    if distributed:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None)
    run_dir = output / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"generation": 0, "update": 0, "episode_cursor": 0,
             "baseline_return_sum": 0.0, "baseline_return_count": 0,
             "history": [], "rng": _rng_snapshot(torch), "execution_migration": None,
             "smoke_run": bool(smoke), "exploratory_only": bool(exploratory),
             "formal_eligible": False if exploratory else None,
             "benchmark_eligible": not smoke and not exploratory}
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), source_root / "farm_sim" / "torch_training.py",
                    source_root / "farm_sim" / "torch_checkpoint.py", source_root / "requirements-training.txt",
                    *(source_root / "farm_sim" / name for name in
                    ("simulation.py", "model.py", "navigation.py", "perception.py", "memory.py", "physiology.py", "policy.py"))]
    if exploratory_variant is not None:
        source_paths.append(source_root / "farm_sim" / "exploratory_learning.py")
    source_hashes = {str(path.relative_to(source_root)): _sha(path) for path in source_paths}
    config = {"version": CONFIG_VERSION, "algorithm": algorithm, "seed": int(seed),
              "source_hashes": source_hashes, "numpy_version": np.__version__,
              "seed_manifest": {"root_seed": int(seed), "split": "dynamic-train", "world_seed_derivation": "sha256(root,train-world,episode-index)",
                                "action_uniform_derivation": "sha256(root,action-uniform,world-seed,episode,tick,animal-index)"},
              "torch_version": torch.__version__, "optimizer": "Adam",
              "reinforce_baseline": "prior_completed_batch_return_to_go_mean",
              "a2c_targets": "monte_carlo_no_bootstrap_no_gae", "entropy_bonus": 0.0,
              "episodes": int(episodes), "rollout_ticks": ROLLOUT_TICKS, "batch_episodes": BATCH_EPISODES,
              "lr": LEARNING_RATE, "grad_clip_norm": GRAD_CLIP_NORM,
              "feature_schema": "cattle-observation-v1", "feature_names": list(FEATURE_NAMES),
              "actions": [x.value for x in ACTIONS], "dtype": "float32", "amp": False, "tf32": False,
              "world_size": world_size}
    if exploratory:
        config.update({"protocol": "exploratory-comparison-v1" if exploratory_variant else "exploratory-a2c-v1",
                       "exploratory_variant": exploratory_variant, "exploratory_only": True,
                       "formal_eligible": False, "locked_outcomes_accessed": False})
    if exploratory_variant is not None:
        config.update({"reward_variant": reward_variant, "state_reward_scale": float(state_reward_scale),
                       "normalize_advantage": bool(normalize_advantage),
                       "entropy_coefficient": float(entropy_coefficient),
                       "discount_factor": float(discount_factor)})
    if resume:
        restored = load_generation(resume if resume != "latest" else run_dir, model=train_model,
                                   optimizer=optimizer, map_location=device,
                                   expected_algorithm=algorithm, expected_config=config,
                                   allow_world_size_migration=allow_execution_migration)
        state = restored["state"]
        # Preserve diagnostic-only lineage through later train/eval invocations.
        # A normal invocation must never promote a smoke checkpoint to a benchmark.
        state["smoke_run"] = bool(state.get("smoke_run", False) or smoke)
        if bool(state.get("exploratory_only", False)) != bool(exploratory):
            raise ValueError("checkpoint exploratory/formal lineage differs from requested mode")
        if exploratory and state.get("formal_eligible") is not False:
            raise ValueError("exploratory checkpoint is missing its nonformal eligibility marker")
        if restored.get("execution_migration") is not None:
            # Keep migration provenance durable across later same-world-size
            # resumes so metrics/evaluations cannot be mistaken for direct runs.
            state["execution_migration"] = restored["execution_migration"]
        _rng_restore(torch, state["rng"])
    if eval_only and not resume:
        raise ValueError("eval-only mode requires a verified resumable checkpoint")
    start_episode = int(state["episode_cursor"])
    while not eval_only and start_episode < episodes:
        batch_size = min(BATCH_EPISODES, episodes - start_episode)
        if distributed and batch_size % world_size:
            raise ValueError("DDP tail must shard equally across ranks")
        episode_indices = shard_episode_indices(start_episode, batch_size, world_size, rank)
        episode_specs = [(derive_seed(seed, "train-world", ep_index), ep_index) for ep_index in episode_indices]
        batch_started = time.perf_counter()
        stage_started = time.perf_counter()
        trajectories = collect_episodes_batch(train_model, episode_specs, root_seed=seed)
        rollout_seconds = time.perf_counter() - stage_started
        optimizer.zero_grad(set_to_none=True)
        stage_started = time.perf_counter()
        if exploratory_variant is not None:
            loss, stats = calculate_exploratory_loss(train_model, trajectories, algorithm,
                previous_return_sum=state["baseline_return_sum"],
                previous_return_count=state["baseline_return_count"],
                reward_variant=reward_variant, state_reward_scale=state_reward_scale,
                normalize_advantage=normalize_advantage, discount_factor=discount_factor,
                entropy_coefficient=entropy_coefficient)
        else:
            loss, stats = calculate_loss(train_model, trajectories, algorithm,
                previous_return_sum=state["baseline_return_sum"],
                previous_return_count=state["baseline_return_count"])
        forward_loss_seconds = time.perf_counter() - stage_started
        assert_finite_training_state(loss, model)
        stage_started = time.perf_counter()
        loss.backward()
        assert_finite_training_state(loss, model, gradients=True)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).detach().cpu())
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm; refusing optimizer/checkpoint update")
        backward_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        optimizer.step()
        assert_finite_training_state(loss.detach(), model)
        optimizer_seconds = time.perf_counter() - stage_started
        returns = np.concatenate([x.returns_to_go().reshape(-1) for x in trajectories])
        # With equal per-rank episodes, DDP all-reduce mean is the global sample mean.
        batch_return_sum = float(returns.sum())
        batch_return_count = int(returns.size)
        episode_reward_sum = float(sum(float(x.rewards.sum()) for x in trajectories))
        if distributed:
            totals = torch.tensor([batch_return_sum, batch_return_count, episode_reward_sum], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            batch_return_sum, batch_return_count = float(totals[0].item()), int(totals[1].item())
            episode_reward_sum = float(totals[2].item())
            metric_keys = ("actor_loss", "value_loss", "entropy", "mean_return_to_go", "mean_value", "mean_advantage")
            metric_values = torch.tensor([stats[k] for k in metric_keys], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
            for k, v in zip(metric_keys, metric_values.tolist()): stats[k] = float(v / world_size)
            stats["return_sum"] = batch_return_sum
            stats["return_count"] = batch_return_count
        state["baseline_return_sum"] += batch_return_sum
        state["baseline_return_count"] += batch_return_count
        state["update"] += 1
        state["episode_cursor"] += batch_size
        state["generation"] += 1
        state["rng"] = _rng_snapshot(torch)
        action_counts = np.asarray([int(np.sum(np.concatenate([x.actions.reshape(-1) for x in trajectories]) == i)) for i in range(len(ACTIONS))], dtype=np.int64)
        prob_means = np.asarray([float(np.concatenate([x.probabilities.reshape(-1, len(ACTIONS)) for x in trajectories])[:, i].mean()) for i in range(len(ACTIONS))], dtype=np.float64)
        component_names = ("forage_success", "forage_depleted", "water_success", "water_blocked", "rested", "ruminated", "herd_contact", "herd_alone", "walked")
        component_counts = np.asarray([sum((x.event_counts or {}).get(kind, 0) for x in trajectories) for kind in component_names], dtype=np.int64)
        if distributed:
            ac = torch.as_tensor(action_counts, device=device); torch.distributed.all_reduce(ac); action_counts = ac.cpu().numpy()
            pm = torch.as_tensor(prob_means, device=device); torch.distributed.all_reduce(pm); prob_means = (pm / world_size).cpu().numpy()
            cc = torch.as_tensor(component_counts, device=device); torch.distributed.all_reduce(cc); component_counts = cc.cpu().numpy()
        row = {"algorithm": algorithm, "seed": seed, "update": state["update"],
               "episode_start": start_episode, "episodes_in_batch": batch_size,
               "transitions": batch_size * ROLLOUT_TICKS * 5, **stats,
               "batch_return": episode_reward_sum / batch_size,
               "mean_trajectory_return": episode_reward_sum / batch_size,
               "raw_environment_batch_return": episode_reward_sum / batch_size,
               "training_target_mean_return_to_go": float(stats["mean_return_to_go"]),
               "reward_components": {kind: int(component_counts[i]) for i, kind in enumerate(component_names)},
               "mean_action_probability": {a.value: float(prob_means[i]) for i, a in enumerate(ACTIONS)},
               "batch_action_counts": {a.value: int(action_counts[i]) for i,a in enumerate(ACTIONS)},
               "invalid_actions": int(sum(x.invalid_outputs for x in trajectories)),
               "grad_norm_preclip": grad_norm,
               "clip_fraction": float(grad_norm > GRAD_CLIP_NORM),
               "parameter_norm": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).cpu()),
               "elapsed_seconds": None, "device": device, "world_size": world_size,
               "execution_migration": state.get("execution_migration"),
               "exploratory_only": bool(exploratory), "formal_eligible": False if exploratory else None,
               "benchmark_eligible": not state.get("smoke_run", False) and not exploratory and state.get("execution_migration") is None,
               "exploratory_variant": exploratory_variant,
               "timing": {"rollout_seconds": rollout_seconds,
                          "forward_loss_seconds": forward_loss_seconds,
                          "backward_gradient_seconds": backward_seconds,
                          "optimizer_seconds": optimizer_seconds,
                          "checkpoint_io_seconds": 0.0,
                          "batch_wall_seconds": None,
                          "execution_migration": state.get("execution_migration"),
                          "exploratory_only": bool(exploratory),
                          "benchmark_eligible": not state.get("smoke_run", False) and not exploratory and state.get("execution_migration") is None}}
        if rank == 0:
            if state["update"] % checkpoint_every == 0 or state["episode_cursor"] >= episodes:
                checkpoint_started = time.perf_counter()
                save_generation(run_dir, model=model, optimizer=optimizer, state=state,
                                config=config, algorithm=algorithm, rank=rank)
                row["timing"]["checkpoint_io_seconds"] = time.perf_counter() - checkpoint_started
            row["timing"]["batch_wall_seconds"] = time.perf_counter() - batch_started
            row["elapsed_seconds"] = row["timing"]["batch_wall_seconds"]
            row["timing"]["elapsed_seconds"] = row["timing"]["batch_wall_seconds"]
            _metrics_write(metrics_path, row)
        if distributed:
            torch.distributed.barrier()
        start_episode = int(state["episode_cursor"])
        if max_updates is not None and state["update"] >= max_updates:
            break
    if distributed:
        torch.distributed.barrier()
    if rank == 0 and not eval_only and state["episode_cursor"] < episodes:
        result = {"checkpoint": str(run_dir / "LATEST"), "complete": False,
                  "episode_cursor": state["episode_cursor"], "updates": state["update"]}
        if distributed: torch.distributed.destroy_process_group()
        return result
    if rank == 0 and (eval_only or state["episode_cursor"] >= episodes):
        # Preserve a model-only local-inference artifact only after training; eval-only is read-only.
        exported = None
        if not eval_only:
            exported = export_actor_npz(model, run_dir / f"torch-{algorithm}-seed-{seed}.npz", seed=seed,
            episodes=int(state["episode_cursor"]), metadata={"algorithm": algorithm, "config": config,
            "synthetic_task_only": True, "training_checkpoint": str(run_dir / "LATEST")})
        generation_path = resolve_generation(resume if eval_only else run_dir)
        training_manifest = json.loads((generation_path / "manifest.json").read_text())
        if eval_only:
            sidecar = None
        else:
            sidecar = exported.with_suffix(".json")
        if not eval_only:
            sidecar_doc = json.loads(sidecar.read_text())
            sidecar_doc["metadata"].update({"training_generation": generation_path.name,
                                            "training_state_sha256": training_manifest["state_sha256"],
                                            "model_npz_sha256": _sha(exported)})
            sidecar.write_text(json.dumps(sidecar_doc, sort_keys=True, indent=2) + "\n")
            artifact_manifest = {"algorithm": algorithm, "training_generation": generation_path.name,
                "training_state_sha256": training_manifest["state_sha256"],
                "policy_npz": exported.name, "policy_npz_sha256": _sha(exported),
                "policy_json": sidecar.name, "policy_json_sha256": _sha(sidecar),
                "feature_schema": "cattle-observation-v1", "actions": [x.value for x in ACTIONS],
                "synthetic_task_only": True}
            artifact_manifest["execution_migration"] = state.get("execution_migration")
            artifact_manifest["exploratory_only"] = bool(exploratory)
            artifact_manifest["formal_eligible"] = False if exploratory else None
            artifact_manifest["benchmark_eligible"] = not state.get("smoke_run", False) and not exploratory and state.get("execution_migration") is None
            (run_dir / "artifact-manifest.json").write_text(json.dumps(artifact_manifest, sort_keys=True, indent=2) + "\n")
        eval_count = 2 if smoke else eval_count
        eval_rows = [deterministic_rollout(model, derive_seed(seed, "eval-world", i), root_seed=seed, episode_index=i)
                     for i in range(eval_count)]
        evaluation = {"algorithm": algorithm, "seed": seed, "device": device,
                      "mean_return": float(np.mean([x["return"] for x in eval_rows])),
                      "episodes": eval_rows,
                      "execution_migration": state.get("execution_migration"),
                      "exploratory_only": bool(exploratory), "formal_eligible": False if exploratory else None,
                      "benchmark_eligible": not state.get("smoke_run", False) and not exploratory and state.get("execution_migration") is None,
                      "claims_boundary": "Synthetic software-task results only; not biological validation."}
        evaluation_path = run_dir / (f"evaluation-readonly-seed-{seed}.json" if eval_only else "evaluation.json")
        evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        result = {"checkpoint": str(generation_path / "manifest.json"),
                  "export": str(exported) if exported is not None else None,
                  "evaluation": str(evaluation_path), "updates": state["update"], "eval_only": eval_only,
                  "execution_migration": state.get("execution_migration"),
                  "exploratory_only": bool(exploratory), "formal_eligible": False if exploratory else None,
                  "benchmark_eligible": not state.get("smoke_run", False) and not exploratory and state.get("execution_migration") is None}
        if distributed: torch.distributed.destroy_process_group()
        return result
    if distributed: torch.distributed.destroy_process_group()
    return {"rank": rank, "updates": state["update"]}


def run_exploratory_batch(*, output: Path, device: str, shard_index: int = 0,
                          shard_count: int = 1) -> dict:
    """Run one shard of the immutable 12-job exploratory comparison suite.

    The notebook uses one shard for the single-GPU baseline and two independent
    shards, each masked to one T4, for its bounded dual-GPU calibration pass.
    """
    specs = exploratory_batch_specs()
    if len(specs) > EXPLORATORY_BATCH_MAX_JOBS:
        raise ValueError("exploratory comparison exceeds the 24-job safety limit")
    if device not in {"cpu", "cuda"}:
        raise ValueError("exploratory comparison requires an explicit cpu or cuda device")
    if shard_count not in {1, 2} or not 0 <= shard_index < shard_count:
        raise ValueError("exploratory comparison supports one or two valid worker shards")
    if shard_count == 2 and device != "cuda":
        raise ValueError("two-worker exploratory comparison shards require CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("exploratory comparison uses independent single-process jobs, not DDP")
    selected = specs[shard_index::shard_count]
    results = []
    for spec in selected:
        run_root = output / f"variant-{spec['variant']}" / f"seed-{spec['seed']}"
        checkpoint = run_root / "a2c"
        result = train_algorithm(
            "a2c", seed=spec["seed"], episodes=EXPLORATORY_BATCH_EPISODES,
            output=run_root, device=device,
            resume=str(checkpoint) if (checkpoint / "LATEST").is_file() else None,
            metrics_path=run_root / "metrics.jsonl", checkpoint_every=1,
            eval_count=EXPLORATORY_BATCH_EVAL_EPISODES, exploratory=True,
            exploratory_variant=spec["variant"], reward_variant=spec["reward_variant"],
            state_reward_scale=spec["state_reward_scale"],
            normalize_advantage=spec["normalize_advantage"],
            entropy_coefficient=spec["entropy_coefficient"])
        results.append({**spec, **result, "run_root": str(run_root)})
    summary = {"schema_version": 1, "protocol": "exploratory-comparison-v1",
               "exploratory_only": True, "formal_eligible": False,
               "locked_outcomes_accessed": False, "device": device,
               "planned_jobs": len(specs), "max_jobs": EXPLORATORY_BATCH_MAX_JOBS,
               "shard_index": shard_index, "shard_count": shard_count,
               "jobs": results}
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"batch-shard-{shard_index}-of-{shard_count}.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="two-update CPU smoke run with export/eval")
    p.add_argument("--smoke-update", action="store_true",
                   help="one disposable BATCH_EPISODES update on explicit CPU or CUDA; single algorithm only")
    p.add_argument("--readiness-ddp", action="store_true",
                   help="bounded 2-rank CUDA readiness test (32 episodes, 2 updates), no evaluation/promotion")
    p.add_argument("--exploratory-a2c", action="store_true",
                   help="separate, fixed-budget nonformal A2C learning run (256 episodes, CUDA, 5 exploratory rollouts)")
    p.add_argument("--exploratory-batch", action="store_true",
                   help="fixed A-D x 3-seed nonformal comparison (512 episodes per job); separate from --exploratory-a2c")
    p.add_argument("--profile-only", action="store_true",
                   help="performance-only real 16-episode calibration; no optimizer step or checkpoint")
    p.add_argument("--profile-device", choices=("cpu", "cuda:0", "cuda:1"), default=None)
    p.add_argument("--profile-output", type=Path, default=None)
    p.add_argument("--profile-repeats", type=int, default=2)
    p.add_argument("--batch-shard-index", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--batch-shard-count", type=int, default=1,
                   help=argparse.SUPPRESS)
    p.add_argument("--algorithm", choices=("reinforce", "a2c", "both"), default="both")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=810001)
    p.add_argument("--episodes", type=int, default=278)
    p.add_argument("--output", type=Path, default=Path("artifacts/torch-training"))
    p.add_argument("--resume", nargs="?", const="latest", default=None)
    p.add_argument("--metrics", type=Path, default=None)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--mode", choices=("train", "eval", "run"), default="train")
    p.add_argument("--eval-seeds", type=int, default=10)
    p.add_argument("--max-updates", type=int, default=None, help="stop after N committed updates (for recovery testing)")
    p.add_argument("--allow-execution-migration", action="store_true",
                   help="allow resume across world_size only; all experiment config must match")
    args = p.parse_args(argv)
    if args.episodes <= 0 or args.checkpoint_every <= 0:
        p.error("--episodes and --checkpoint-every must be positive")
    if args.smoke and args.smoke_update:
        p.error("--smoke and --smoke-update are mutually exclusive")
    if args.exploratory_a2c and args.exploratory_batch:
        p.error("--exploratory-a2c and --exploratory-batch are separate fixed protocols")
    if args.profile_only:
        if (args.profile_device is None or args.profile_output is None or args.exploratory_a2c
                or args.exploratory_batch or args.smoke or args.smoke_update or args.readiness_ddp
                or args.resume is not None or args.max_updates is not None):
            p.error("--profile-only requires --profile-device and --profile-output, and cannot train/resume")
        try:
            result = profile_exploratory_step(device=args.profile_device, output=args.profile_output,
                                               repeats=args.profile_repeats)
        except (RuntimeError, ValueError) as exc:
            p.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.exploratory_a2c:
        if (args.algorithm != "a2c" or args.device != "cuda" or args.mode != "train"
                or args.seed != EXPLORATORY_A2C_SEED or args.episodes != EXPLORATORY_A2C_EPISODES
                or args.eval_seeds != EXPLORATORY_A2C_EVAL_EPISODES or args.checkpoint_every != 1
                or args.smoke or args.smoke_update or args.readiness_ddp or args.max_updates is not None
                or int(os.environ.get("WORLD_SIZE", "1")) != 1):
            p.error("--exploratory-a2c requires its fixed A2C/CUDA/seed/budget config and one process")
        torch = require_torch()
        if not torch.cuda.is_available():
            p.error("--exploratory-a2c requires CUDA")
        if torch.cuda.device_count() < 1:
            p.error("--exploratory-a2c requires at least one visible CUDA device")
    if args.exploratory_batch:
        if (args.algorithm != "a2c" or args.device not in {"cpu", "cuda"} or args.mode != "train"
                or args.episodes not in {278, EXPLORATORY_BATCH_EPISODES}
                or args.seed != 810001 or args.eval_seeds not in {10, EXPLORATORY_BATCH_EVAL_EPISODES}
                or args.checkpoint_every != 1 or args.smoke or args.smoke_update or args.readiness_ddp
                or args.resume is not None or args.max_updates is not None or args.allow_execution_migration):
            p.error("--exploratory-batch requires fixed A2C/CPU-or-CUDA/512-episode config; resume is automatic per job")
        if not 0 <= args.batch_shard_index < args.batch_shard_count or args.batch_shard_count not in {1, 2}:
            p.error("--exploratory-batch supports one or two valid worker shards")
        torch = require_torch()
        if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 1):
            p.error("--exploratory-batch requires a visible CUDA device")
        if args.batch_shard_count == 2 and (args.device != "cuda" or torch.cuda.device_count() != 1):
            p.error("two-shard execution must mask each independent job to exactly one visible CUDA device")
        result = run_exploratory_batch(output=args.output / "exploratory-comparison",
            device=args.device, shard_index=args.batch_shard_index, shard_count=args.batch_shard_count)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if (args.batch_shard_index != 0 or args.batch_shard_count != 1):
        p.error("batch shard options require --exploratory-batch")
    if args.readiness_ddp:
        if args.smoke or args.smoke_update or args.algorithm == "both":
            p.error("--readiness-ddp requires one algorithm and cannot combine with smoke modes")
        if args.device != "cuda":
            p.error("--readiness-ddp requires --device cuda")
        if args.mode != "train" or args.eval_seeds != 10 or args.allow_execution_migration:
            p.error("--readiness-ddp is training-only and does not support evaluation or migration")
        if args.metrics is not None:
            p.error("--readiness-ddp writes readiness evidence only; --metrics is not supported")
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            p.error("--readiness-ddp requires torchrun with exactly two ranks")
        if args.max_updates not in {None, 1}:
            p.error("--readiness-ddp permits at most one update per invocation")
        if args.episodes not in {278, READINESS_EPISODES}:
            p.error("--readiness-ddp has a fixed 32-episode budget")
        if args.checkpoint_every != 1:
            p.error("--readiness-ddp checkpoints every update")
        torch = require_torch()
        if not torch.cuda.is_available():
            p.error("--readiness-ddp requires CUDA availability")
        try:
            if torch.cuda.device_count() < 2:
                p.error("--readiness-ddp requires two visible CUDA devices")
        except RuntimeError as exc:
            p.error(f"CUDA device query failed: {exc}")
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank not in {0, 1}:
            p.error("--readiness-ddp requires LOCAL_RANK 0 or 1")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        try:
            result = run_readiness_ddp(args.algorithm, seed=args.seed, output=args.output,
                                       device="cuda", resume=args.resume, max_updates=args.max_updates)
            if int(os.environ.get("RANK", "0")) == 0:
                print(json.dumps(result, indent=2))
        finally:
            torch.distributed.destroy_process_group()
        return
    if not args.smoke and not args.smoke_update and not args.exploratory_a2c and args.mode != "eval":
        p.error("ordinary CLI training is disabled: no independently trusted experimental authorization verifier is configured; use the reviewed release workflow after authorization is established")
    if args.smoke:
        args.algorithm = "both"
        args.device = "cpu"
        args.episodes = 2 * BATCH_EPISODES
        args.checkpoint_every = 1
        args.output = args.output / "smoke"
    if args.smoke_update:
        if args.algorithm == "both":
            p.error("--smoke-update requires exactly one --algorithm (reinforce or a2c)")
        if args.device not in {"cpu", "cuda"}:
            p.error("--smoke-update requires an explicit --device cpu or --device cuda")
        if args.mode != "train" or args.resume is not None:
            p.error("--smoke-update supports only a fresh --mode train run; resume/eval are not permitted")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            p.error("--smoke-update does not support distributed execution")
        args.episodes = BATCH_EPISODES
        args.checkpoint_every = 1
        args.max_updates = 1
        args.output = args.output / "smoke"
    if args.exploratory_a2c:
        args.output = args.output / "exploratory-only"
    if args.mode == "eval" and args.resume is None:
        p.error("--mode eval requires --resume pointing at a committed checkpoint")
    metrics = args.metrics or args.output / "metrics.jsonl"
    algorithms = ("reinforce", "a2c") if args.algorithm == "both" else (args.algorithm,)
    results = []
    for algo in algorithms:
        resume = args.resume
        if resume == "latest": resume = str(args.output / algo)
        result = train_algorithm(algo, seed=args.seed, episodes=args.episodes,
                                 output=args.output, device=args.device, resume=resume,
                                 metrics_path=metrics, checkpoint_every=args.checkpoint_every,
                                 smoke=(args.smoke or args.smoke_update), max_updates=args.max_updates,
                                 eval_count=args.eval_seeds, eval_only=args.mode == "eval",
                                 allow_execution_migration=args.allow_execution_migration,
                                 exploratory=args.exploratory_a2c)
        results.append(result)
    if not int(os.environ.get("RANK", "0")):
        print(json.dumps({"results": results, "device_requested": args.device,
                          "torch_version": require_torch().__version__,
                          "cuda_available": require_torch().cuda.is_available(),
                          "python": platform.python_version()}, indent=2))

if __name__ == "__main__":
    main()
