#!/usr/bin/env python3
"""Fixed, resumable nonformal A2C cohorts for the autoresearch mission.

This module is called only by the explicitly authorized Kaggle AUTORESEARCH
notebook mode. It is not a general training CLI and cannot access formal data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.train_torch_mvp1 import train_algorithm

MAX_ADDITIONAL_RUNS = 24
EPISODES = 512
SEEDS_BY_BATCH = {
    1: (9484101, 9484102, 9484103),
    2: (9484104, 9484105, 9484106),
}
VARIANTS_BY_BATCH = {
    1: {
        "G": {"reward_variant": "original", "state_reward_scale": 0.0,
              "normalize_advantage": False, "entropy_coefficient": 0.0},
        "H": {"reward_variant": "original", "state_reward_scale": 0.0,
              "normalize_advantage": True, "entropy_coefficient": 0.01},
    },
    2: {
        "I": {"reward_variant": "state_aware", "state_reward_scale": 0.5,
              "normalize_advantage": False, "entropy_coefficient": 0.0},
        "J": {"reward_variant": "state_aware", "state_reward_scale": 0.5,
              "normalize_advantage": True, "entropy_coefficient": 0.01},
    },
}


def batch_specs(batch: int) -> list[dict[str, Any]]:
    if batch not in VARIANTS_BY_BATCH:
        raise ValueError("only preregistered autoresearch batch 1 or 2 is currently enabled")
    seeds = SEEDS_BY_BATCH[batch]
    return [{"variant": variant, **config, "seed": seed, "episodes": EPISODES,
             "perception_mode": "path-aware-v2", "resource_profile": "resource-rich-v1",
             "protocol": "autoresearch-world-perception-v1"}
            for variant, config in VARIANTS_BY_BATCH[batch].items() for seed in seeds]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_candidate_manifest(*, batch: int, output: Path) -> Path:
    """Create a hash-bound inference checkpoint manifest after all cohort jobs complete."""
    specs = batch_specs(batch)
    entries = []
    for spec in specs:
        base = output / f"variant-{spec['variant']}" / f"seed-{spec['seed']}" / "a2c"
        policy = base / f"torch-a2c-seed-{spec['seed']}.npz"
        sidecar = policy.with_suffix(".json")
        if not policy.is_file() or not sidecar.is_file():
            raise ValueError(f"inference export is missing for {spec['variant']} seed {spec['seed']}")
        doc = json.loads(sidecar.read_text())
        meta = doc.get("metadata", {})
        config = meta.get("config", {}) if isinstance(meta, dict) else {}
        if (doc.get("episodes") != EPISODES or doc.get("seed") != spec["seed"]
                or config.get("exploratory_only") is not True
                or config.get("formal_eligible") is not False
                or config.get("locked_outcomes_accessed") is not False
                or config.get("perception_mode") != spec["perception_mode"]
                or config.get("resource_profile") != spec["resource_profile"]
                or config.get("exploratory_variant") != spec["variant"]):
            raise ValueError(f"export metadata does not match the authorized cohort: {policy}")
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        entries.append({"variant": spec["variant"], "seed": spec["seed"],
                        "policy_npz": policy.relative_to(output).as_posix(), "sha256": digest})
    manifest = {"schema_version": 1, "mode": "autoresearch_exploratory",
                "batch": batch, "protocol": "autoresearch-world-perception-v1",
                "formal_eligible": False, "benchmark_eligible": False,
                "locked_test_access": False, "locked_outcomes_accessed": False,
                "candidate_checkpoints": entries}
    path = output / "autoresearch-candidate-manifest.json"
    _atomic_json(path, manifest)
    return path


def run_batch(*, batch: int, output: Path, device: str, shard_index: int = 0,
              shard_count: int = 1, train_callable=None) -> dict[str, Any]:
    """Run the fixed six-model cohort, two independent single-process shards at most."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be explicitly cpu or cuda")
    specs = batch_specs(batch)
    if len(specs) > MAX_ADDITIONAL_RUNS:
        raise ValueError("planned autoresearch cohort exceeds the 24-run mission limit")
    if shard_count not in {1, 2} or not 0 <= shard_index < shard_count:
        raise ValueError("supported shard_count is one or two")
    if shard_count == 2 and device != "cuda":
        raise ValueError("dual-worker shards require explicit CUDA")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("autoresearch cohorts use independent jobs, not DDP")
    selected = specs[shard_index::shard_count]
    train = train_callable or train_algorithm
    results = []
    for spec in selected:
        run_root = output / f"variant-{spec['variant']}" / f"seed-{spec['seed']}"
        checkpoint = run_root / "a2c"
        job_id = f"batch-{batch}-{spec['variant']}-seed-{spec['seed']}"
        ledger_path = output / f"{job_id}-run.json"
        previous = json.loads(ledger_path.read_text()) if ledger_path.is_file() else None
        if previous and previous.get("status") == "complete":
            results.append({**spec, **previous, "run_root": str(run_root), "resumed": False})
            continue
        record = {"job_id": job_id, "batch": batch, "variant": spec["variant"],
                  "seed": spec["seed"], "episodes": EPISODES, "device": device,
                  "status": "running", "started_at_unix": time.time(),
                  "formal_eligible": False, "locked_outcomes_accessed": False}
        _atomic_json(ledger_path, record)  # durable before the training call
        try:
            result = train(
                "a2c", seed=spec["seed"], episodes=EPISODES, output=run_root,
                device=device, resume=str(checkpoint) if (checkpoint / "LATEST").is_file() else None,
                metrics_path=run_root / "metrics.jsonl", checkpoint_every=1,
                eval_count=5, exploratory=True, exploratory_variant=spec["variant"],
                reward_variant=spec["reward_variant"],
                state_reward_scale=spec["state_reward_scale"],
                normalize_advantage=spec["normalize_advantage"],
                entropy_coefficient=spec["entropy_coefficient"],
                discount_factor=1.0, perception_mode=spec["perception_mode"],
                resource_profile=spec["resource_profile"],
                paired_seed_set=list(SEEDS_BY_BATCH[batch]))
            record.update({"status": "complete", "result": result,
                           "finished_at_unix": time.time()})
            _atomic_json(ledger_path, record)
            results.append({**spec, **result, "run_root": str(run_root), "resumed": previous is not None})
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__,
                           "error": str(exc), "finished_at_unix": time.time()})
            _atomic_json(ledger_path, record)
            raise
    summary = {"schema_version": 1, "protocol": "autoresearch-world-perception-v1",
               "batch": batch, "exploratory_only": True, "formal_eligible": False,
               "benchmark_eligible": False, "locked_outcomes_accessed": False,
               "locked_test_access": False, "device": device,
               "planned_jobs": len(specs), "max_additional_runs": MAX_ADDITIONAL_RUNS,
               "shard_index": shard_index, "shard_count": shard_count,
               "training_seeds": list(SEEDS_BY_BATCH[batch]), "jobs": results}
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"autoresearch-batch-{batch}-shard-{shard_index}-of-{shard_count}.json"
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, choices=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(run_batch(batch=args.batch, output=args.output, device=args.device,
                               shard_index=args.shard_index, shard_count=args.shard_count),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
