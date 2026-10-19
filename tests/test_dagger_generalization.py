from __future__ import annotations

from dataclasses import replace
import numpy as np

from validation.dagger_generalization import (
    GeneralizationCloner, GeneralizationEnv, TEST_DOMAINS, TRAIN_PROFILE,
    domain_for,
)


def test_train_environment_profiles_do_not_include_unseen_test_values():
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


def test_joint_policy_cloner_returns_five_coordinated_actions():
    model = GeneralizationCloner(35, seed=19, agents=5, architecture="joint")
    x = np.zeros((8, 175), dtype=np.float32)
    y = np.zeros((8, 5), dtype=np.int64)
    model.fit(x, y, epochs=1)
    action, _ = model.predict(np.zeros(175, dtype=np.float32))
    assert action.shape == (5,)
    assert np.all((action >= 0) & (action < 6))
