import unittest

import numpy as np

from farm_sim.model import Action, FEATURE_NAMES
from farm_sim.exploratory_learning import (
    entropy_regularized_actor_loss,
    normalize_advantages,
    state_aware_rewards,
)


class ExploratoryLearningTests(unittest.TestCase):
    def test_state_reward_is_opt_in_copied_and_availability_aware(self):
        observations = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
        observations[:, FEATURE_NAMES.index("fatigue")] = [0.0, 1.0, 0.2]
        observations[:, FEATURE_NAMES.index("forage_available")] = [0.0, 0.0, 0.0]
        observations[:, FEATURE_NAMES.index("hunger")] = [1.0, 1.0, 0.8]
        # Enum order is the actor output/index contract.
        actions = np.array([list(Action).index(Action.REST), list(Action).index(Action.REST),
                            list(Action).index(Action.GRAZE)])
        rewards = np.zeros(3, dtype=np.float32)
        original_obs, original_actions, original_rewards = observations.copy(), actions.copy(), rewards.copy()
        shaped = state_aware_rewards(observations, actions, rewards)
        self.assertLess(shaped[0], 0)
        self.assertGreater(shaped[1], 0)
        self.assertEqual(shaped[2], 0)  # no bonus for unavailable forage
        np.testing.assert_array_equal(observations, original_obs)
        np.testing.assert_array_equal(actions, original_actions)
        np.testing.assert_array_equal(rewards, original_rewards)
        np.testing.assert_array_equal(state_aware_rewards(observations, actions, rewards, scale=0), rewards)

    def test_advantage_and_entropy_helpers_keep_control_defaults(self):
        advantages = np.array([1.0, 2.0, 3.0])
        self.assertIs(normalize_advantages(advantages), advantages)
        normalized = normalize_advantages(advantages, enabled=True)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=7)
        self.assertEqual(entropy_regularized_actor_loss(3.0, 2.0), 3.0)
        self.assertEqual(entropy_regularized_actor_loss(3.0, 2.0, coefficient=0.25), 2.5)

    def test_shapes_and_coefficients_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "leading dimensions"):
            state_aware_rewards(np.zeros((2, 20)), np.zeros(1), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            entropy_regularized_actor_loss(1.0, 1.0, coefficient=-1)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            normalize_advantages(np.ones(2), enabled=True, eps=0)


if __name__ == "__main__":
    unittest.main()
