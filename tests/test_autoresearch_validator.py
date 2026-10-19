from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.validate_autoresearch_candidate import validate


def _group(*, reward: float, forage: int, water: int, hunger: float, thirst: float,
           nav_failures: int = 0, episodes: int = 36):
    return {
        "episodes": episodes,
        "mean_original_reward_per_episode": reward,
        "reward_component_counts": {"forage_success": forage, "water_success": water},
        "mean_final_drives": {"hunger": hunger, "thirst": thirst},
        "invalid_actions": 0,
        "navigation_failure_count": nav_failures,
    }


def _comparison(*, candidate=None, seed=9484999):
    return {
        "mode": "autoresearch_comparison",
        "formal_eligible": False,
        "benchmark_eligible": False,
        "locked_test_access": False,
        "locked_outcomes_accessed": False,
        "reward_mutations": False,
        "reserved_data_read": False,
        "evaluation_root_seed": seed,
        "episodes_per_family": 3,
        "ticks_per_episode": 72,
        "evaluation_reward_source": "original simulator reward",
        "scenario_families": ["starter", "resource_scarce_relocated", "wide", "unseen_corridor"],
        "training_seed_sets": {"candidate": [9485101, 9485102, 9485103]},
        "seed_disjointness": {"eval_root_distinct_from_training_roots": True,
                              "world_seeds_unique": True,
                              "common_worlds_disjoint_from_training_worlds": True},
        "conditions": {
            "candidate": {"common_worlds": [{"scenario": fam, "seed": seed * 100 + i}
                                               for i, fam in enumerate(["starter", "resource_scarce_relocated", "wide", "unseen_corridor"])] if candidate is None else [{"scenario": fam, "seed": seed * 100 + i} for i, fam in enumerate(["starter", "resource_scarce_relocated", "wide", "unseen_corridor"])],
                          "policies": {"trained": candidate or _group(reward=5, forage=100, water=40, hunger=.5, thirst=.4)}},
            "reference-A": {"common_worlds": [{"scenario": fam, "seed": seed * 100 + i} for i, fam in enumerate(["starter", "resource_scarce_relocated", "wide", "unseen_corridor"])],
                            "policies": {"trained": _group(reward=1, forage=20, water=10, hunger=.9, thirst=.9),
                                         "untrained_same_seed": _group(reward=2, forage=30, water=15, hunger=.8, thirst=.8),
                                         "heuristic_baseline": _group(reward=3, forage=35, water=18, hunger=.7, thirst=.7)}},
        },
    }


def test_accepts_candidate_only_when_all_resource_return_and_state_gates_pass(tmp_path):
    doc = _comparison()
    result = validate(doc, doc)
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["gate_metrics"]["forage_successes_per_episode"]["candidate"] > result["gate_metrics"]["forage_successes_per_episode"]["best_reference"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert json.loads(path.read_text())["passed"] is True


@pytest.mark.parametrize("metric", ["reward", "forage", "water"])
def test_any_failed_primary_gate_rejects_candidate(metric):
    candidate = _group(reward=5, forage=100, water=40, hunger=.5, thirst=.4)
    if metric == "reward": candidate["mean_original_reward_per_episode"] = 3
    if metric == "forage": candidate["reward_component_counts"]["forage_success"] = 35
    if metric == "water": candidate["reward_component_counts"]["water_success"] = 18
    doc = _comparison(candidate=candidate)
    assert validate(doc, doc)["passed"] is False


def test_mismatched_evaluation_worlds_fail_closed():
    candidate = _comparison()
    refs = _comparison()
    refs["conditions"]["reference-A"]["common_worlds"][0]["seed"] += 1
    result = validate(candidate, refs)
    assert result["passed"] is False
    assert any("world" in failure.lower() for failure in result["failures"])


def test_formal_or_locked_inputs_are_rejected():
    candidate = _comparison()
    candidate["locked_test_access"] = True
    result = validate(candidate, _comparison())
    assert result["passed"] is False
    assert any("locked" in failure.lower() for failure in result["failures"])


@pytest.mark.parametrize("field", ["locked_outcomes_accessed", "reward_mutations", "reserved_data_read"])
def test_reward_or_reserved_data_mutation_flags_are_rejected(field):
    candidate = _comparison()
    candidate[field] = True
    result = validate(candidate, _comparison())
    assert result["passed"] is False
    assert any(field in failure for failure in result["failures"])


def test_selects_one_candidate_cohort_from_multi_candidate_report():
    doc = _comparison()
    group = doc["conditions"].pop("candidate")
    doc["conditions"]["candidate-G"] = group
    doc["training_seed_sets"]["candidate-G"] = doc["training_seed_sets"].pop("candidate")
    result = validate(doc, doc, candidate_condition="candidate-G")
    assert result["passed"] is True
    assert result["training_seeds"] == [9485101, 9485102, 9485103]
