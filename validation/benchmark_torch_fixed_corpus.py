#!/usr/bin/env python3
"""Frozen-corpus PyTorch compute benchmark (not a learning-quality experiment).

Full workload is the approved kaggle-a2c-fixed-corpus-v1 schedule: ten training
partitions, each with 278 complete episodes and 18 real optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farm_sim.model import Action, FEATURE_NAMES
from farm_sim.torch_training import (ActorCritic, ROLLOUT_TICKS, collect_episode,
                                     derive_seed, require_torch, set_deterministic,
                                     shard_episode_indices, validate_equal_all_one_masks)

PARTITIONS = 10
FULL_EPISODES = 278
GLOBAL_BATCH_EPISODES = 16
UPDATES_PER_PARTITION = 18
CORPUS_VERSION = "kaggle-a2c-fixed-corpus-v1"
SOURCE_FILES = ("farm_sim/torch_training.py", "farm_sim/torch_checkpoint.py", "farm_sim/simulation.py",
                "farm_sim/model.py", "farm_sim/navigation.py", "farm_sim/perception.py",
                "farm_sim/memory.py", "farm_sim/physiology.py", "farm_sim/policy.py",
                "validation/benchmark_torch_fixed_corpus.py")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def corpus_config(*, root_seed: int, smoke: bool) -> dict[str, Any]:
    return {"version": CORPUS_VERSION, "root_seed": int(root_seed),
            "partition_count": 2 if smoke else PARTITIONS,
            "episodes_per_partition": 2 if smoke else FULL_EPISODES,
            "ticks_per_episode": ROLLOUT_TICKS, "animals_per_episode": 5,
            "observations_shape_per_episode": [ROLLOUT_TICKS, 5, len(FEATURE_NAMES)],
            "actions_shape_per_episode": [ROLLOUT_TICKS, 5], "all_one_masks": True,
            "behavior_policy": "frozen ActorCritic A2C-compatible actor, arg sampling via counter-uniforms",
            "behavior_policy_seed": derive_seed(root_seed, "fixed-corpus-behavior-policy"),
            "world_seed_derivation": "sha256(root, fixed-corpus-train-partition, partition, train-world, episode-index)",
            "split": "training-only fixed workload; never held-out; stale off-policy compute replay",
            "smoke_mode": bool(smoke)}


def build_corpus(root_seed: int, *, smoke: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect complete simulator episodes once, outside any timed replay."""
    torch = require_torch()
    config = corpus_config(root_seed=root_seed, smoke=smoke)
    n_partitions = config["partition_count"]
    episodes = config["episodes_per_partition"]
    set_deterministic(derive_seed(root_seed, "corpus-generation"), "cpu")
    behavior = ActorCritic(value_head=True, seed=config["behavior_policy_seed"], dtype=torch.float32).cpu().eval()
    shape = (n_partitions, episodes, ROLLOUT_TICKS, 5)
    observations = np.empty(shape + (len(FEATURE_NAMES),), dtype=np.float32)
    actions = np.empty(shape, dtype=np.int64)
    rewards = np.empty(shape, dtype=np.float32)
    returns = np.empty(shape, dtype=np.float32)
    masks = np.ones(shape, dtype=np.uint8)
    world_ids = np.empty((n_partitions, episodes), dtype=np.int64)
    for partition in range(n_partitions):
        partition_seed = derive_seed(root_seed, "fixed-corpus-train-partition", partition)
        for episode_index in range(episodes):
            world_seed = derive_seed(partition_seed, "train-world", episode_index)
            trajectory = collect_episode(behavior, world_seed, root_seed=partition_seed, episode_index=episode_index)
            observations[partition, episode_index] = trajectory.observations
            actions[partition, episode_index] = trajectory.actions
            rewards[partition, episode_index] = trajectory.rewards
            returns[partition, episode_index] = trajectory.returns_to_go()
            world_ids[partition, episode_index] = world_seed
    arrays = {"observations": observations, "actions": actions, "rewards": rewards,
              "returns_to_go": returns, "masks": masks, "world_ids": world_ids}
    if not np.all(masks == 1):
        raise AssertionError("corpus must contain all-one masks")
    if observations.shape[2:] != (72, 5, 20) or actions.shape[2:] != (72, 5):
        raise AssertionError("fixed corpus contains incomplete or malformed episodes")
    config["array_sha256"] = {key: array_sha(value) for key, value in arrays.items()}
    config["corpus_sha256"] = hashlib.sha256(json.dumps(config["array_sha256"], sort_keys=True).encode()).hexdigest()
    config["source_sha256"] = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    config["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    config["package_lock_sha256"] = file_sha(ROOT / "requirements-training.txt")
    with torch.no_grad():
        behavior_bytes = b"".join(p.detach().cpu().numpy().tobytes(order="C") for p in behavior.parameters())
    config["behavior_policy_sha256"] = hashlib.sha256(behavior_bytes).hexdigest()
    config["environment_semantics"] = {"ticks_per_day": 24, "ticks_per_episode": 72,
                                       "animals_per_world": 5, "reward_source": "unchanged farm_sim.physiology.step_physiology"}
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return arrays, config


def save_corpus(path: str | Path, arrays: dict[str, np.ndarray], config: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    target.with_suffix(".json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return target


def load_corpus(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = Path(path)
    config = json.loads(target.with_suffix(".json").read_text())
    with np.load(target, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("observations", "actions", "rewards", "returns_to_go", "masks", "world_ids")}
    computed = {key: array_sha(value) for key, value in arrays.items()}
    if computed != config.get("array_sha256"):
        raise ValueError("frozen corpus array hash mismatch")
    config_body = {key: value for key, value in config.items() if key != "config_sha256"}
    if hashlib.sha256(json.dumps(config_body, sort_keys=True).encode()).hexdigest() != config.get("config_sha256"):
        raise ValueError("frozen corpus config hash mismatch")
    live_sources = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    if live_sources != config.get("source_sha256"):
        changed = sorted(k for k in live_sources if live_sources[k] != config.get("source_sha256", {}).get(k))
        raise ValueError(f"frozen corpus source provenance changed: {changed}")
    if hashlib.sha256(json.dumps(config["array_sha256"], sort_keys=True).encode()).hexdigest() != config.get("corpus_sha256"):
        raise ValueError("frozen corpus manifest digest mismatch")
    if not np.all(arrays["masks"] == 1):
        raise ValueError("fixed-corpus replay rejects zero/padded rows")
    if arrays["observations"].shape[2:] != (72, 5, 20) or arrays["actions"].shape[2:] != (72, 5):
        raise ValueError("fixed-corpus replay requires complete 72-tick five-animal episodes")
    if arrays["world_ids"].shape != arrays["observations"].shape[:2] or len(np.unique(arrays["world_ids"])) != arrays["world_ids"].size:
        raise ValueError("fixed corpus must have unique world IDs aligned to every episode")
    if np.any(arrays["actions"] < 0) or np.any(arrays["actions"] >= len(Action)):
        raise ValueError("frozen corpus contains invalid action indices")
    return arrays, config


def _sync(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _one_update(model, optimizer, observations, actions, returns, device, torch):
    x = torch.as_tensor(observations.reshape(-1, observations.shape[-1]), dtype=torch.float32, device=device)
    a = torch.as_tensor(actions.reshape(-1), dtype=torch.long, device=device)
    g = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
    logits, values = model(x)
    log_probs = torch.log_softmax(logits, dim=-1).gather(1, a[:, None]).squeeze(1)
    advantages = g - values
    actor_loss = -(log_probs * advantages.detach()).mean()
    value_loss = .5 * ((values - g) ** 2).mean()
    entropy = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean()
    loss = actor_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return {"actor_loss": float(actor_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()), "gradient_norm_preclip": float(grad_norm.detach().cpu())}


def _batches(partition: int, episodes: int):
    start = 0
    while start < episodes:
        count = min(GLOBAL_BATCH_EPISODES, episodes - start)
        if episodes == FULL_EPISODES and count not in (16, 6):
            raise AssertionError("full workload must use exactly 17×16 episodes then a six-episode tail")
        yield partition, start, count
        start += count


def replay_once(arrays: dict[str, np.ndarray], *, seed: int, device_name: str,
                repeat_index: int, smoke: bool = False) -> dict[str, Any]:
    """Time actual forward/loss/backward/clip/optimizer steps on frozen rows."""
    torch = require_torch()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if distributed and world_size != 2:
        raise ValueError("fixed-corpus DDP benchmark requires exactly two ranks (T4x2)")
    device = torch.device(device_name)
    if distributed and device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    set_deterministic(derive_seed(seed, "fixed-replay", repeat_index), str(device))
    model = ActorCritic(value_head=True, seed=derive_seed(seed, "fixed-replay-model"), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None) if distributed else model
    episodes = arrays["observations"].shape[1]
    n_partitions = arrays["observations"].shape[0]
    observed_masks = [arrays["masks"][p].reshape(-1) for p in range(n_partitions)]
    if not smoke:
        validate_equal_all_one_masks([observed_masks[0], observed_masks[0]])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    step_rows = []
    # Warmup runs perform actual model updates but are excluded; reset model/optimizer
    # below so timed runs have a fixed initial actor.
    if smoke:
        warmups = 0
    else:
        warmups = 1
    batches = list(item for p in range(n_partitions) for item in _batches(p, episodes))
    for warmup_idx in range(warmups):
        p, start, count = batches[0]
        local = list(range(start, start + count))
        if distributed:
            local = [i for i in local if i % world_size == rank]
        _one_update(train_model, optimizer, arrays["observations"][p, local], arrays["actions"][p, local], arrays["returns_to_go"][p, local], device, torch)
    # The benchmark excludes setup/warmup, but starts the timed workload at a
    # reproducible initialized state with fresh Adam moments.
    set_deterministic(derive_seed(seed, "fixed-replay", repeat_index), str(device))
    model = ActorCritic(value_head=True, seed=derive_seed(seed, "fixed-replay-model"), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None) if distributed else model
    if distributed:
        torch.distributed.barrier()
    _sync(torch, device)
    started = time.perf_counter()
    update_index = 0
    for partition, start, count in batches:
        if distributed and count % world_size:
            raise ValueError("full and tail episode batches must divide evenly across two ranks")
        all_indices = np.arange(start, start + count, dtype=np.int64)
        if distributed:
            local_indices = all_indices[rank::world_size]
            counts = [len(all_indices[r::world_size]) for r in range(world_size)]
            if len(set(counts)) != 1:
                raise ValueError("DDP local episode counts must be equal")
        else:
            local_indices = all_indices
        stats = _one_update(train_model, optimizer,
                            arrays["observations"][partition, local_indices],
                            arrays["actions"][partition, local_indices],
                            arrays["returns_to_go"][partition, local_indices], device, torch)
        update_index += 1
        _sync(torch, device)
        step_rows.append({"partition": partition, "episode_start": start, "episodes": count,
                          "update_index": update_index, **stats})
    _sync(torch, device)
    elapsed = time.perf_counter() - started
    if distributed:
        values = torch.tensor([elapsed, float(update_index)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        elapsed = float(values[0].item())
        if int(values[1].item()) != update_index * world_size:
            raise AssertionError("unexpected update accounting in DDP")
    if not smoke and (update_index != PARTITIONS * UPDATES_PER_PARTITION):
        raise AssertionError(f"expected 180 updates; found {update_index}")
    if distributed and rank != 0:
        return {"rank": rank, "duration_seconds": elapsed, "updates": update_index}
    decisions = n_partitions * episodes * ROLLOUT_TICKS * 5
    return {"repeat_index": repeat_index, "device": str(device), "world_size": world_size,
            "duration_seconds": elapsed, "updates": update_index, "decisions": int(decisions),
            "decisions_per_second": float(decisions / elapsed), "step_rows": step_rows,
            "computation": "real float32 model forward, policy/value loss, backward, DDP reduction if active, global grad clip, Adam update",
            "corpus_transition_ids_hash": array_sha(arrays["world_ids"]),
            "warmup_updates_excluded": warmups, "smoke_mode": bool(smoke)}


def bootstrap_interval(values: list[float], *, seed: int = 8102, reps: int = 10000) -> dict[str, float | None]:
    if not values:
        return {"lower_95": None, "upper_95": None}
    if len(values) == 1:
        return {"lower_95": values[0], "upper_95": values[0]}
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values, dtype=np.float64), size=(reps, len(values)), replace=True).mean(axis=1)
    return {"lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["duration_seconds"]) for row in rows]
    throughput = [float(row["decisions_per_second"]) for row in rows]
    if not rows:
        return {"repeats": 0,
                "duration_seconds": {"raw": [], "median": None, "p95": None, "bootstrap_ci95_mean": bootstrap_interval([])},
                "decisions_per_second": {"raw": [], "median": None, "p95": None, "bootstrap_ci95_mean": bootstrap_interval([])}}
    return {"repeats": len(rows), "duration_seconds": {"raw": durations, "median": float(np.median(durations)),
            "p95": float(np.quantile(durations, .95)), "bootstrap_ci95_mean": bootstrap_interval(durations)},
            "decisions_per_second": {"raw": throughput, "median": float(np.median(throughput)),
            "p95": float(np.quantile(throughput, .95)), "bootstrap_ci95_mean": bootstrap_interval(throughput)}}


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_hash = left.get("corpus_config", {}).get("corpus_sha256")
    right_hash = right.get("corpus_config", {}).get("corpus_sha256")
    if not left_hash or left_hash != right_hash:
        raise ValueError("device reports must use the same frozen corpus hash")
    if left.get("workload") != right.get("workload"):
        raise ValueError("device reports must use the same full/smoke workload")
    a = sorted(left["runs"], key=lambda x: x["repeat_index"])
    b = sorted(right["runs"], key=lambda x: x["repeat_index"])
    paired = []
    for first, second in zip(a, b):
        if first["repeat_index"] != second["repeat_index"]:
            raise ValueError("comparison reports have unmatched repeat indices")
        paired.append(float(first["duration_seconds"]) / float(second["duration_seconds"]))
    if len(a) != len(b):
        raise ValueError("comparison reports must have the same number of repeats")
    return {"numerator_device": left["device"], "denominator_device": right["device"],
            "paired_elapsed_speedup": {"raw": paired, "median": float(np.median(paired)) if paired else None,
                                       "bootstrap_ci95_mean": bootstrap_interval(paired)}}


def run_benchmark(corpus_path: str | Path, *, device: str, repeats: int, seed: int,
                  output: str | Path, smoke: bool = False) -> dict[str, Any]:
    torch = require_torch()
    arrays, config = load_corpus(corpus_path)
    if bool(config.get("smoke_mode")) != bool(smoke):
        raise ValueError("corpus smoke/full mode does not match benchmark mode")
    if not smoke and (config["partition_count"] != PARTITIONS or config["episodes_per_partition"] != FULL_EPISODES):
        raise ValueError("full benchmark requires 10 partitions x 278 episodes")
    rows = [replay_once(arrays, seed=seed, device_name=device, repeat_index=i, smoke=smoke) for i in range(repeats)] if int(os.environ.get("RANK", "0")) == 0 else []
    cuda_metadata = None
    if device == "cuda" and torch.cuda.is_available():
        cuda_metadata = [{"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i),
                          "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    report = {"schema_version": 1, "workload": CORPUS_VERSION if not smoke else f"{CORPUS_VERSION}-explicit-smoke",
              "scope": "fixed frozen training-only corpus compute benchmark; stale/off-policy corpus; no learning-quality or policy-promotion claims",
              "device": device, "torchrun_world_size": int(os.environ.get("WORLD_SIZE", "1")),
              "ddp_active": int(os.environ.get("WORLD_SIZE", "1")) > 1,
              "ddp_t4x2_claim": "eligible only when two CUDA ranks actually execute DDP" if int(os.environ.get("WORLD_SIZE", "1")) == 2 and device == "cuda" else "not a T4x2 result",
              "corpus_path": str(corpus_path), "corpus_config": config,
              "corpus_file_sha256": file_sha(Path(corpus_path)), "replay_seed": seed,
              "dtype": "float32", "amp": False, "tf32": False, "torch_version": torch.__version__,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "cuda_available": bool(torch.cuda.is_available()), "cuda_devices": cuda_metadata,
              "schedule": {"partitions": config["partition_count"], "episodes_per_partition": config["episodes_per_partition"],
                           "updates": sum(row["updates"] for row in rows) if rows else None,
                           "update_sequence": "per partition 16-episode global batches plus complete tail; smoke is explicitly reduced"},
              "runs": rows, "summary": summarize(rows), "interpretation": {"compute_verdict": "UNMEASURED" if not rows else "DESCRIPTIVE_ONLY_NO_THRESHOLD_EVALUATION",
                    "gpu_speedup_claim": False, "full_acceptance_requires": "five paired full repeats per device, corpus parity, DDP parity/recovery tests, and approved speed thresholds"}}
    if cuda_metadata is None and device == "cuda":
        report["runs"] = []
        report["summary"] = summarize([])
        report["blocker"] = "CUDA was requested but is not available in this runtime"
    if int(os.environ.get("RANK", "0")) == 0:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-corpus")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--smoke", action="store_true", help="explicit two-partition/two-episode smoke corpus; not acceptance evidence")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--corpus", type=Path, required=True)
    bench.add_argument("--device", choices=("cpu", "cuda"), required=True)
    bench.add_argument("--seed", type=int, default=51337)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--smoke", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate-corpus":
        arrays, config = build_corpus(args.seed, smoke=args.smoke)
        path = save_corpus(args.output, arrays, config)
        print(json.dumps({"corpus": str(path), "corpus_sha256": config["corpus_sha256"],
                          "episodes": int(arrays["world_ids"].size), "smoke_mode": args.smoke}, indent=2))
    elif args.command == "benchmark":
        if args.repeats <= 0:
            parser.error("--repeats must be positive")
        report = run_benchmark(args.corpus, device=args.device, repeats=args.repeats,
                               seed=args.seed, output=args.output, smoke=args.smoke)
        print(json.dumps({"output": str(args.output), "device": args.device,
                          "runs": len(report["runs"]), "ddp_active": report["ddp_active"],
                          "interpretation": report["interpretation"]}, indent=2))
    else:
        first = json.loads(args.first.read_text()); second = json.loads(args.second.read_text())
        result = compare_reports(first, second)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
