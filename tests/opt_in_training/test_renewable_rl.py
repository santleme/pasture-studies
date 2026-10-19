from __future__ import annotations

from validation.opt_in_training_tests import require_opt_in_training_tests

require_opt_in_training_tests()

import numpy as np

from farm_sim.simulation import make_world
from validation.renewable_rl import (
    BehaviorCloner,
    collect_teacher_data,
    evaluate_policy,
    teacher_ceiling,
    train_ppo,
    RenewablePastureEnv,
    add_resource_sites,
)


SEEDS = (11, 29, 47, 83, 101)


def test_resource_expansion_is_copy_only_and_does_not_rewrite_starter_world():
    starter = make_world(17)
    expanded = add_resource_sites(starter, forage_sites=8, water_sites=6)
    assert len(starter.forage) == 3
    assert len(starter.water) == 1
    assert len(expanded.forage) == 8
    assert len(expanded.water) == 6
    assert not ({p.position for p in expanded.forage} & {p.position for p in expanded.water})
    assert starter == make_world(17)


def test_teacher_proves_one_agent_solvable_on_independent_behavior_metrics():
    result = teacher_ceiling(agents=1, seeds=SEEDS, forage_sites=8, water_sites=6)
    metrics = result["controlled_mean"]
    assert metrics["forage_successes"] >= 9
    assert metrics["water_successes"] >= 8
    assert metrics["hunger"] <= 0.25
    assert metrics["thirst"] <= 0.25


def test_teacher_proves_five_agent_solvable_per_agent_not_only_in_aggregate():
    result = teacher_ceiling(agents=5, seeds=SEEDS, forage_sites=8, water_sites=6)
    for animal_id, metrics in result["per_agent"].items():
        assert metrics["forage_successes"] >= 8, animal_id
        assert metrics["water_successes"] >= 8, animal_id
        assert metrics["hunger"] <= 0.25, animal_id
        assert metrics["thirst"] <= 0.25, animal_id


def test_global_observation_and_action_spaces_match_renewable_world():
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
    assert np.all((observation[-8:-6] >= 0.0) & (observation[-8:-6] <= 1.0))
    np.testing.assert_array_equal(observation[-6:], np.eye(6, dtype=np.float32)[3])
    env.close()


def test_behavior_cloning_and_ppo_adapters_execute_for_both_scales():
    for agents in (1, 5):
        observations, labels = collect_teacher_data(
            agents=agents, episodes=2, seed=41,
            forage_sites=8, water_sites=6,
        )
        assert observations.shape[0] == labels.shape[0]
        assert observations.shape[1] == 30
        cloner = BehaviorCloner(30, 6, seed=41, agents=agents).fit(
            observations, labels, epochs=2,
        )
        env = RenewablePastureEnv(
            agents=agents, seed=41, forage_sites=8, water_sites=6,
        )
        obs, _ = env.reset(seed=43)
        actions, _ = cloner.predict(obs)
        assert env.action_space.contains(actions)
        env.close()
        ppo = train_ppo(
            agents=agents, seed=41, total_timesteps=512,
            forage_sites=8, water_sites=6,
            global_observations=True, reward_mode="state_cost",
        )
        evaluation = evaluate_policy(
            ppo, agents=agents, seeds=(43,), forage_sites=8, water_sites=6,
        )
        assert len(evaluation["per_agent"]) == 5
        assert np.isfinite(evaluation["controlled_mean"]["resource_successes"])
