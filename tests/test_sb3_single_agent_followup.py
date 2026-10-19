import pytest
import sys
from types import ModuleType

from validation.sb3_single_agent_followup import (
    DEFAULT_SEED,
    DEFAULT_TRAINING_STEPS,
    EVALUATION_SEEDS,
    learning_gate,
    run_followup,
)


def _result(trained, heuristic):
    return {
        "trained": trained,
        "baselines": {"need_controller": heuristic},
    }


def test_learning_gate_uses_behavior_metrics_and_requires_all_heuristic_checks():
    passed = _result(
        {"mean_resource_successes": 54, "mean_controlled_hunger": 0.08, "mean_controlled_thirst": 0.5},
        {"mean_resource_successes": 53, "mean_controlled_hunger": 0.1, "mean_controlled_thirst": 0.6},
    )
    failed = _result(
        {"mean_resource_successes": 54, "mean_controlled_hunger": 0.08, "mean_controlled_thirst": 0.7},
        {"mean_resource_successes": 53, "mean_controlled_hunger": 0.1, "mean_controlled_thirst": 0.6},
    )
    assert learning_gate(passed)["status"] == "PASS"
    assert learning_gate(failed) == {
        "status": "FAIL",
        "checks": {
            "resource_successes_at_least_heuristic": True,
            "controlled_hunger_no_worse_than_heuristic": True,
            "controlled_thirst_no_worse_than_heuristic": False,
        },
    }


def test_run_followup_reuses_single_agent_budget_and_fixed_seed_evaluation(monkeypatch):
    calls = []
    result = {
        "algorithm": "SB3 PPO",
        "agents": 1,
        "training_steps": DEFAULT_TRAINING_STEPS,
        "seed": DEFAULT_SEED,
        "trained": {"mean_resource_successes": 50, "mean_controlled_hunger": 1, "mean_controlled_thirst": 1},
        "baselines": {
            "need_controller": {"mean_resource_successes": 53, "mean_controlled_hunger": 0.1, "mean_controlled_thirst": 1},
            "random": {"mean_resource_successes": 38, "mean_controlled_hunger": 1, "mean_controlled_thirst": 1},
        },
        "evaluation_seeds": list(EVALUATION_SEEDS),
    }

    def fake_train_and_evaluate(**kwargs):
        calls.append(kwargs)
        return result

    pilot = ModuleType("validation.sb3_pilot")
    pilot.train_and_evaluate = fake_train_and_evaluate
    monkeypatch.setitem(sys.modules, "validation.sb3_pilot", pilot)
    actual = run_followup()
    assert calls == [{"agents": 1, "total_timesteps": DEFAULT_TRAINING_STEPS, "seed": DEFAULT_SEED}]
    assert actual["evaluation_seeds"] == [101, 102, 103]
    assert actual["learning_gate"]["status"] == "FAIL"
    assert actual["training_budget_cap"] == DEFAULT_TRAINING_STEPS


@pytest.mark.parametrize("steps", [0, 7_201])
def test_run_followup_rejects_out_of_scope_training_budget(steps):
    with pytest.raises(ValueError, match="training_steps"):
        run_followup(training_steps=steps)
