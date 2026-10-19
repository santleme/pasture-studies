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
from farm_sim.simulation import Simulation, make_world
from farm_sim.torch_checkpoint import load_generation, resolve_generation, save_generation
from farm_sim.torch_training import (
    ACTIONS, ActorCritic, ROLLOUT_TICKS, calculate_loss, collect_episode, collect_episodes_batch,
    derive_seed, export_actor_npz, intervention_probabilities, require_torch,
    resolve_device, set_deterministic, shard_episode_indices,
    InvalidPolicyOutput, assert_finite_training_state,
)
from farm_sim.execution_policy import require_kaggle_for_full_workload

BATCH_EPISODES = 16
GRAD_CLIP_NORM = 2.0
LEARNING_RATE = 1e-3
CONFIG_VERSION = "trajectory-v1"
READINESS_EPISODES = 32
READINESS_UPDATES = 2
READINESS_BATCH_EPISODES = 16


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
    animal_rows = sim.world.animals
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
                    checkpoint_every: int, smoke: bool = False, max_updates: int | None = None, eval_count: int = 10, eval_only: bool = False, allow_execution_migration: bool = False) -> dict:
    require_kaggle_for_full_workload(
        f"{algorithm} training/evaluation",
        smoke=smoke or (eval_only and episodes <= 32 and eval_count <= 10) or (not eval_only and episodes <= 32),
    )
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
             "smoke_run": bool(smoke)}
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), source_root / "farm_sim" / "torch_training.py",
                    source_root / "farm_sim" / "torch_checkpoint.py", source_root / "requirements-training.txt",
                    *(source_root / "farm_sim" / name for name in
                    ("simulation.py", "model.py", "navigation.py", "perception.py", "memory.py", "physiology.py", "policy.py"))]
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
    if resume:
        restored = load_generation(resume if resume != "latest" else run_dir, model=train_model,
                                   optimizer=optimizer, map_location=device,
                                   expected_algorithm=algorithm, expected_config=config,
                                   allow_world_size_migration=allow_execution_migration)
        state = restored["state"]
        # Preserve diagnostic-only lineage through later train/eval invocations.
        # A normal invocation must never promote a smoke checkpoint to a benchmark.
        state["smoke_run"] = bool(state.get("smoke_run", False) or smoke)
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
        trajectories = collect_episodes_batch(train_model, episode_specs, root_seed=seed)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = calculate_loss(train_model, trajectories, algorithm,
            previous_return_sum=state["baseline_return_sum"], previous_return_count=state["baseline_return_count"])
        assert_finite_training_state(loss, model)
        loss.backward()
        assert_finite_training_state(loss, model, gradients=True)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).detach().cpu())
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm; refusing optimizer/checkpoint update")
        optimizer.step()
        assert_finite_training_state(loss.detach(), model)
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
               "reward_components": {kind: int(component_counts[i]) for i, kind in enumerate(component_names)},
               "mean_action_probability": {a.value: float(prob_means[i]) for i, a in enumerate(ACTIONS)},
               "batch_action_counts": {a.value: int(action_counts[i]) for i,a in enumerate(ACTIONS)},
               "invalid_actions": int(sum(x.invalid_outputs for x in trajectories)),
               "grad_norm_preclip": grad_norm,
               "clip_fraction": float(grad_norm > GRAD_CLIP_NORM),
               "parameter_norm": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).cpu()),
               "elapsed_seconds": None, "device": device, "world_size": world_size,
               "execution_migration": state.get("execution_migration"),
               "benchmark_eligible": not state.get("smoke_run", False) and state.get("execution_migration") is None,
               "timing": {"elapsed_seconds": None,
                          "execution_migration": state.get("execution_migration"),
                          "benchmark_eligible": not state.get("smoke_run", False) and state.get("execution_migration") is None}}
        if rank == 0:
            _metrics_write(metrics_path, row)
            if state["update"] % checkpoint_every == 0 or state["episode_cursor"] >= episodes:
                save_generation(run_dir, model=model, optimizer=optimizer, state=state,
                                config=config, algorithm=algorithm, rank=rank)
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
            artifact_manifest["benchmark_eligible"] = not state.get("smoke_run", False) and state.get("execution_migration") is None
            (run_dir / "artifact-manifest.json").write_text(json.dumps(artifact_manifest, sort_keys=True, indent=2) + "\n")
        eval_count = 2 if smoke else eval_count
        eval_rows = [deterministic_rollout(model, derive_seed(seed, "eval-world", i), root_seed=seed, episode_index=i)
                     for i in range(eval_count)]
        evaluation = {"algorithm": algorithm, "seed": seed, "device": device,
                      "mean_return": float(np.mean([x["return"] for x in eval_rows])),
                      "episodes": eval_rows,
                      "execution_migration": state.get("execution_migration"),
                      "benchmark_eligible": not state.get("smoke_run", False) and state.get("execution_migration") is None,
                      "claims_boundary": "Synthetic software-task results only; not biological validation."}
        evaluation_path = run_dir / (f"evaluation-readonly-seed-{seed}.json" if eval_only else "evaluation.json")
        evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        result = {"checkpoint": str(generation_path / "manifest.json"),
                  "export": str(exported) if exported is not None else None,
                  "evaluation": str(evaluation_path), "updates": state["update"], "eval_only": eval_only,
                  "execution_migration": state.get("execution_migration"),
                  "benchmark_eligible": not state.get("smoke_run", False) and state.get("execution_migration") is None}
        if distributed: torch.distributed.destroy_process_group()
        return result
    if distributed: torch.distributed.destroy_process_group()
    return {"rank": rank, "updates": state["update"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="two-update CPU smoke run with export/eval")
    p.add_argument("--smoke-update", action="store_true",
                   help="one disposable BATCH_EPISODES update on explicit CPU or CUDA; single algorithm only")
    p.add_argument("--readiness-ddp", action="store_true",
                   help="bounded 2-rank CUDA readiness test (32 episodes, 2 updates), no evaluation/promotion")
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
    if args.readiness_ddp:
        try:
            require_kaggle_for_full_workload("A2C readiness training")
        except RuntimeError as exc:
            p.error(str(exc))
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
    try:
        require_kaggle_for_full_workload(
            "A2C/REINFORCE training or evaluation",
            smoke=args.smoke or args.smoke_update,
        )
    except RuntimeError as exc:
        p.error(str(exc))
    if not args.smoke and not args.smoke_update and args.mode != "eval":
        p.error("ordinary CLI training is disabled: no independently trusted experimental authorization verifier is configured; use the reviewed release workflow after authorization is established")
    if args.smoke:
        if args.mode == "eval":
            if args.algorithm == "both" or args.episodes != BATCH_EPISODES or args.eval_seeds != 10:
                p.error("smoke evaluation requires one algorithm, one 16-episode batch, and the fixed ten-seed diagnostic schedule")
            if args.resume is None or args.allow_execution_migration or args.max_updates is not None:
                p.error("smoke evaluation requires a committed smoke checkpoint and does not accept migration/update overrides")
        else:
            if args.mode != "train" or args.resume is not None or args.allow_execution_migration or args.max_updates is not None:
                p.error("--smoke training is fresh-run only and does not accept resume, migration, or update overrides")
            args.algorithm = "both"
            args.device = "cpu"
            args.episodes = 2 * BATCH_EPISODES
            args.checkpoint_every = 1
            args.max_updates = 1
            args.eval_seeds = 1
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
                                 allow_execution_migration=args.allow_execution_migration)
        results.append(result)
    if not int(os.environ.get("RANK", "0")):
        print(json.dumps({"results": results, "device_requested": args.device,
                          "torch_version": require_torch().__version__,
                          "cuda_available": require_torch().cuda.is_available(),
                          "python": platform.python_version()}, indent=2))

if __name__ == "__main__":
    main()
