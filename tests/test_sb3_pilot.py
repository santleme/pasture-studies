import numpy as np
import importlib.util
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None,
                                reason="optional Gymnasium dependency is not installed")


def test_single_agent_wrapper_is_gymnasium_compatible_and_episode_is_fixed_horizon():
    from validation.sb3_pilot import EPISODE_TICKS, PastureSB3Env
    env = PastureSB3Env(agents=1, seed=13)
    obs, info = env.reset(seed=13)
    assert env.observation_space.contains(obs)
    assert info["seed"] == 13
    terminated = False
    ticks = 0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0)
        ticks += 1
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not truncated
    assert ticks == EPISODE_TICKS
    assert "resource_successes" in info
    assert set(info["per_agent"]) == {f"cow-{i}" for i in range(1, 6)}
    assert info["per_agent"]["cow-1"]["forage_successes"] + info["per_agent"]["cow-1"]["water_successes"] <= info["resource_successes"]
    assert info["per_agent"]["cow-1"]["hunger"] == info["controlled_hunger"]
    assert info["per_agent"]["cow-1"]["thirst"] == info["controlled_thirst"]


def test_five_agent_joint_wrapper_emits_one_action_per_agent_from_snapshot():
    from validation.sb3_pilot import PastureSB3Env
    env = PastureSB3Env(agents=5, seed=19)
    obs, _ = env.reset(seed=19)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (5,)
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.int64))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated and not truncated
    assert sum(info["action_counts"].values()) == 5 * 4
