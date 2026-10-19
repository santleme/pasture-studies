from pathlib import Path

import pytest

from validation.run_autoresearch_training import batch_specs, create_candidate_manifest, run_batch
import json


def test_batch_specs_are_fixed_disjoint_nonformal_cohorts():
    first = batch_specs(1)
    second = batch_specs(2)
    third = batch_specs(3)
    assert len(first) == len(second) == len(third) == 6
    assert {x["variant"] for x in first} == {"G", "H"}
    assert {x["variant"] for x in second} == {"I", "J"}
    assert {x["seed"] for x in first}.isdisjoint({x["seed"] for x in second})
    assert {x["variant"] for x in third} == {"K", "L"}
    assert {x["seed"] for x in third}.isdisjoint({x["seed"] for x in first + second})
    assert all(x["episodes"] == 512 for x in first + second + third)
    assert all(x["perception_mode"] == "path-aware-v2" for x in first + second + third)
    assert all(x["resource_profile"] == "resource-rich-v1" for x in first + second + third)
    assert all(x["reward_variant"] == "outcome_aware" for x in third)
    with pytest.raises(ValueError, match="preregistered"):
        batch_specs(4)


def test_runner_shards_jobs_and_records_started_run_before_training(tmp_path: Path):
    calls = []
    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return {"checkpoint": "generation", "export": "policy.npz"}

    result = run_batch(batch=1, output=tmp_path, device="cuda", shard_index=0,
                       shard_count=2, train_callable=fake_train)
    assert result["planned_jobs"] == 6
    assert len(result["jobs"]) == len(calls) == 3
    assert result["formal_eligible"] is False
    assert result["locked_outcomes_accessed"] is False
    for args, kwargs in calls:
        assert args[0] == "a2c"
        assert kwargs["episodes"] == 512
        assert kwargs["exploratory"] is True
        assert kwargs["perception_mode"] == "path-aware-v2"
        assert kwargs["resource_profile"] == "resource-rich-v1"
        assert len(kwargs["paired_seed_set"]) == 3


def test_runner_refuses_ddp_and_dual_shard_cpu(tmp_path: Path):
    with pytest.raises(ValueError, match="CUDA"):
        run_batch(batch=1, output=tmp_path, device="cpu", shard_count=2)


def test_candidate_manifest_requires_complete_nonformal_exports(tmp_path: Path):
    for spec in batch_specs(1):
        folder = tmp_path / f"variant-{spec['variant']}" / f"seed-{spec['seed']}" / "a2c"
        folder.mkdir(parents=True)
        policy = folder / f"torch-a2c-seed-{spec['seed']}.npz"
        policy.write_bytes(b"synthetic-test-only")
        policy.with_suffix(".json").write_text(json.dumps({
            "seed": spec["seed"], "episodes": 512,
            "metadata": {"config": {"exploratory_only": True,
                "formal_eligible": False, "locked_outcomes_accessed": False,
                "perception_mode": "path-aware-v2", "resource_profile": "resource-rich-v1",
                "exploratory_variant": spec["variant"]}},
        }))
    manifest = create_candidate_manifest(batch=1, output=tmp_path)
    doc = json.loads(manifest.read_text())
    assert len(doc["candidate_checkpoints"]) == 6
    assert doc["locked_test_access"] is False
