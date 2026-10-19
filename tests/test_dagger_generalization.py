from __future__ import annotations

from dataclasses import replace
import importlib.util
import numpy as np
import pytest
from unittest.mock import patch

from farm_sim.model import Action

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="optional PyTorch dependency is not installed")


def test_train_environment_profiles_do_not_include_unseen_test_values():
    from validation.dagger_generalization import TEST_DOMAINS, TRAIN_PROFILE, domain_for
    trained = set(TRAIN_PROFILE)
    for name in ("layout_reference", "layout_shift", "supply_shift",
                 "regrowth_shift", "combined_unseen"):
        assert TEST_DOMAINS[name] not in trained
    assert TEST_DOMAINS["regrowth_shift"][2] not in {profile[2] for profile in trained}
    assert all(TEST_DOMAINS[name] not in trained for name in
               ("layout_reference", "layout_shift", "supply_shift",
                "regrowth_shift", "combined_unseen"))
    assert domain_for(100, "randomized_train") in trained


def test_layout_shift_is_seeded_and_changes_resource_positions_only():
    from validation.dagger_generalization import GeneralizationEnv
    first = GeneralizationEnv(agents=1, seed=7, profile="layout_shift", observation="domain")
    second = GeneralizationEnv(agents=1, seed=7, profile="layout_shift", observation="domain")
    other = GeneralizationEnv(agents=1, seed=7, profile="layout_shift", observation="domain")
    _, _ = first.reset(seed=301)
    _, _ = second.reset(seed=301)
    _, _ = other.reset(seed=307)
    positions1 = (first.sim.world.forage, first.sim.world.water)
    positions2 = (second.sim.world.forage, second.sim.world.water)
    positions3 = (other.sim.world.forage, other.sim.world.water)
    assert positions1 == positions2
    assert positions1 != positions3
    assert len(first.sim.world.animals) == len(other.sim.world.animals) == 5
    assert not ({p.position for p in first.sim.world.forage}
                & {p.position for p in first.sim.world.water})
    for env in (first, second, other):
        env.close()


def test_domain_and_history_observations_fit_declared_space_for_both_scales():
    from validation.dagger_generalization import GeneralizationEnv
    for agents in (1, 5):
        env = GeneralizationEnv(agents=agents, seed=11, profile="combined_unseen",
                                observation="history_domain")
        obs, info = env.reset(seed=411)
        assert info["forage_sites"] == 12
        assert info["water_sites"] == 10
        assert info["regrowth_ticks"] == 1
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        next_obs, *_ = env.step(action)
        assert env.observation_space.contains(next_obs)
        assert np.all((next_obs >= 0) & (next_obs <= 1))
        env.close()


def test_due_regrowth_is_observable_before_the_next_action_without_transition_change():
    from validation.dagger_generalization import GeneralizationEnv
    env = GeneralizationEnv(agents=1, seed=11, profile="nominal", observation="domain")
    obs, _ = env.reset(seed=411)
    patch = env.sim.world.forage[0]
    env.sim.world = replace(env.sim.world,
                             forage=(replace(patch, available=False),) + env.sim.world.forage[1:])
    env.regrow_at[("forage", 0)] = env.sim.world.tick
    updated = env._observation()
    assert env.sim.world.forage[0].available
    assert updated[20] > 0
    env.close()


def test_untrained_joint_policy_cloner_returns_five_coordinated_actions():
    from validation.dagger_generalization import GeneralizationCloner
    model = GeneralizationCloner(35, seed=19, agents=5, architecture="joint")
    action, _ = model.predict(np.zeros(175, dtype=np.float32))
    assert action.shape == (5,)
    assert np.all((action >= 0) & (action < 6))


def test_ordered_navigation_preview_matches_simulator_resource_commits():
    from validation.dagger_generalization import ACTION_LIST, GeneralizationEnv, _ordered_navigation_outcomes
    env = GeneralizationEnv(agents=5, seed=29, profile="combined_unseen",
                            observation="history_domain")
    env.reset(seed=522)
    actions = np.asarray([ACTION_LIST.index(Action.GRAZE)] * 5)
    before = {name: row["forage_success"] for name, row in env.agent_behavior.items()}
    expected = _ordered_navigation_outcomes(env.sim.world, actions)
    env.step(actions)
    actual_successes = sum(
        env.agent_behavior[name]["forage_success"] - before[name]
        for name in before)
    assert actual_successes == sum(outcome == "success" for outcome in expected.values())
    assert set(expected) == {f"cow-{i}" for i in range(1, 6)}
    env.close()


def test_direct_dagger_and_ppo_training_helpers_are_kaggle_only():
    from validation.dagger_generalization import train_generalized_dagger, train_ppo_finetune
    with patch("farm_sim.execution_policy.is_kaggle_runtime", return_value=False), \
            patch("validation.dagger_generalization.collect_domain_data") as collect:
        try:
            train_generalized_dagger(agents=1, seed=7)
        except RuntimeError as exc:
            assert "Kaggle-only" in str(exc)
        else:
            raise AssertionError("direct DAgger training must be denied outside Kaggle")
        collect.assert_not_called()

        try:
            train_ppo_finetune(agents=1, seed=7, total_timesteps=8)
        except RuntimeError as exc:
            assert "Kaggle-only" in str(exc)
        else:
            raise AssertionError("direct PPO fine-tuning must be denied outside Kaggle")
