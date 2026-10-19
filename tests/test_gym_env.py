from __future__ import annotations

import importlib.util
import unittest

import numpy as np

HAS_GYMNASIUM = importlib.util.find_spec("gymnasium") is not None


@unittest.skipUnless(HAS_GYMNASIUM, "optional Gymnasium RL dependency is not installed")
class GymnasiumEnvironmentTests(unittest.TestCase):
    def test_single_animal_env_uses_simulator_and_truncates_at_horizon(self):
        from farm_sim.gym_env import PastureEnv
        env = PastureEnv(horizon=2)
        first, _ = env.reset(seed=7)
        replay, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, replay)
        self.assertEqual(first.shape, (20,))
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(observation.shape, (20,))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["tick"], 1)
        self.assertEqual(info["actions"]["cow-1"], "graze")
        self.assertTrue(all(info["actions"][animal] == "walk" for animal in ("cow-2", "cow-3", "cow-4", "cow-5")))
        _, _, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["tick"], 2)
        env.close()

    def test_joint_env_controls_five_animals_with_mean_reward(self):
        from farm_sim.gym_env import PastureEnv
        from farm_sim.model import Action
        env = PastureEnv(joint=True, horizon=1)
        observation, _ = env.reset(seed=11)
        self.assertEqual(observation.shape, (100,))
        actions = np.arange(5, dtype=np.int64)
        _, reward, terminated, truncated, info = env.step(actions)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["tick"], 1)
        self.assertEqual(set(info["individual_rewards"]), {f"cow-{i}" for i in range(1, 6)})
        self.assertAlmostEqual(reward, np.mean(list(info["individual_rewards"].values())))
        self.assertEqual(list(info["actions"].values()), [action.value for action in tuple(Action)[:5]])
        env.close()

    def test_invalid_actions_fail_before_mutating_world(self):
        from farm_sim.gym_env import PastureEnv
        env = PastureEnv(joint=True)
        env.reset(seed=2)
        world = env.simulation.world
        for bad_action in ([0, 0, 0, 0], [0, 0, 0, 0, 6], [0.0] * 5):
            with self.subTest(action=bad_action), self.assertRaises(ValueError):
                env.step(np.asarray(bad_action))
            self.assertIs(env.simulation.world, world)

    def test_gymnasium_api_checker(self):
        from gymnasium.utils.env_checker import check_env
        from farm_sim.gym_env import PastureEnv
        check_env(PastureEnv(joint=False, horizon=3))
        check_env(PastureEnv(joint=True, horizon=3))

