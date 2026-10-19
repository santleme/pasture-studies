from __future__ import annotations

import numpy as np
import importlib.util
import pytest

from farm_sim.simulation import make_world

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="optional PyTorch dependency is not installed")


SEEDS = (11, 29, 47, 83, 101)


def test_resource_expansion_is_copy_only_and_does_not_rewrite_starter_world():
    from validation.renewable_rl import add_resource_sites
    starter = make_world(17)
    expanded = add_resource_sites(starter, forage_sites=8, water_sites=6)
    assert len(starter.forage) == 3
    assert len(starter.water) == 1
    assert len(expanded.forage) == 8
    assert len(expanded.water) == 6
    assert not ({p.position for p in expanded.forage} & {p.position for p in expanded.water})
    assert starter == make_world(17)


def test_teacher_proves_one_agent_solvable_on_independent_behavior_metrics():
    from validation.renewable_rl import teacher_ceiling
    metrics = teacher_ceiling(agents=1, seeds=SEEDS, forage_sites=8, water_sites=6)["controlled_mean"]
    assert metrics["forage_successes"] >= 9
    assert metrics["water_successes"] >= 8
    assert metrics["hunger"] <= 0.25
    assert metrics["thirst"] <= 0.25


def test_teacher_proves_five_agent_solvable_per_agent_not_only_in_aggregate():
    from validation.renewable_rl import teacher_ceiling
    result = teacher_ceiling(agents=5, seeds=SEEDS, forage_sites=8, water_sites=6)
    for animal_id, metrics in result["per_agent"].items():
        assert metrics["forage_successes"] >= 8, animal_id
        assert metrics["water_successes"] >= 8, animal_id
        assert metrics["hunger"] <= 0.25, animal_id
        assert metrics["thirst"] <= 0.25, animal_id


def test_global_observation_and_action_spaces_match_renewable_world():
    from validation.renewable_rl import RenewablePastureEnv
    env = RenewablePastureEnv(
        agents=1, seed=11, forage_sites=8, water_sites=6,
        global_observations=True, context="rest",
    )
    observation, info = env.reset(seed=11)
    assert observation.shape == (30,)
    assert env.observation_space.contains(observation)
    assert info["forage_sites"] == 8
    assert info["water_sites"] == 6
    np.testing.assert_array_equal(observation[-10:-8], np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(observation[-6:], np.eye(6, dtype=np.float32)[3])
    env.close()
