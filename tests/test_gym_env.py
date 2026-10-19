import unittest

import numpy as np

try:
    import gymnasium
    from farm_sim.gym_env import PastureEnv
except ImportError:
    gymnasium = None
    PastureEnv = None


@unittest.skipIf(gymnasium is None, "optional Gymnasium dependency is not installed")
class PastureGymEnvTests(unittest.TestCase):
    def test_one_animal_reset_step_and_horizon_reuse_simulator_contract(self):
        env = PastureEnv(controlled_animals=1, horizon=2)
        observation, info = env.reset(seed=31)
        self.assertEqual(observation.shape, (20,))
        self.assertEqual(info["controlled_animal_ids"], ["cow-1"])
        self.assertTrue(env.observation_space.contains(observation))

        next_observation, reward, terminated, truncated, step_info = env.step(0)
        self.assertEqual(next_observation.shape, (20,))
        self.assertEqual(reward, step_info["individual_rewards"][0])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["tick"], 1)
        self.assertEqual(env._simulation.world.tick, 1)
        self.assertEqual([a.action.value for a in env._simulation.world.animals[1:]], ["rest"] * 4)

        _, _, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        env.close()

    def test_five_animal_joint_step_matches_simulator_rewards(self):
        env = PastureEnv(controlled_animals=5, horizon=5)
        observation, _ = env.reset(seed=42)
        self.assertEqual(observation.shape, (5, 20))
        actions = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        self.assertTrue(env.action_space.contains(actions))
        next_observation, reward, terminated, truncated, info = env.step(actions)
        expected = [a.last_reward for a in env._simulation.world.animals]
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertEqual(info["individual_rewards"], expected)
        self.assertEqual(reward, float(np.mean(expected)))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env._simulation.world.tick, 1)
        self.assertEqual(info["actions"], ["graze", "drink", "walk", "rest", "ruminate"])

    def test_seeded_reset_and_rollout_are_reproducible(self):
        first, second = PastureEnv(horizon=3), PastureEnv(horizon=3)
        obs_a, _ = first.reset(seed=9)
        obs_b, _ = second.reset(seed=9)
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in (0, 2, 3):
            out_a, reward_a, term_a, trunc_a, info_a = first.step(action)
            out_b, reward_b, term_b, trunc_b, info_b = second.step(action)
            np.testing.assert_array_equal(out_a, out_b)
            self.assertEqual((reward_a, term_a, trunc_a, info_a), (reward_b, term_b, trunc_b, info_b))
        first.close()
        second.close()

    def test_invalid_action_does_not_advance_simulation(self):
        env = PastureEnv()
        env.reset(seed=1)
        before = env._simulation.world
        with self.assertRaises(ValueError):
            env.step(99)
        self.assertEqual(env._simulation.world, before)
        env.close()

    def test_sb3_ppo_smoke_for_single_and_five_controlled_animals(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        for count in (1, 5):
            env = PastureEnv(controlled_animals=count, horizon=4)
            check_env(env, warn=True)
            model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, n_epochs=1, seed=12, device="cpu", verbose=0)
            model.learn(total_timesteps=8)
            observation, _ = env.reset(seed=23)
            action, _ = model.predict(observation, deterministic=True)
            self.assertTrue(env.action_space.contains(action))
            _, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["tick"], 1)
            env.close()


if __name__ == "__main__":
    unittest.main()
