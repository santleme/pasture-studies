import unittest
import importlib.util

import numpy as np

from farm_sim.model import Action, FEATURE_NAMES
from farm_sim.exploratory_learning import (
    calculate_exploratory_loss,
    entropy_regularized_actor_loss,
    normalize_advantages,
    outcome_aware_rewards,
    state_aware_rewards,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


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

    def test_outcome_aware_reward_only_shapes_observed_successful_drive_changes(self):
        observations = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
        next_observations = observations.copy()
        hunger = FEATURE_NAMES.index("hunger")
        thirst = FEATURE_NAMES.index("thirst")
        fatigue = FEATURE_NAMES.index("fatigue")
        observations[:, hunger] = [0.9, 0.9, 0.9, 0.2]
        next_observations[:, hunger] = [0.7, 0.95, 0.9, 0.2]
        observations[:, thirst] = [0.8, 0.8, 0.8, 0.8]
        next_observations[:, thirst] = [0.8, 0.8, 0.5, 0.8]
        observations[:, fatigue] = [0.9, 0.9, 0.2, 0.2]
        next_observations[:, fatigue] = [0.9, 0.9, 0.02, 0.02]
        actions = np.array([list(Action).index(Action.GRAZE), list(Action).index(Action.GRAZE),
                            list(Action).index(Action.DRINK), list(Action).index(Action.REST)])
        rewards = np.zeros(4, dtype=np.float32)
        shaped = outcome_aware_rewards(observations, next_observations, actions, rewards, scale=0.5)
        self.assertAlmostEqual(float(shaped[0]), 0.1)
        self.assertEqual(float(shaped[1]), 0.0)
        self.assertAlmostEqual(float(shaped[2]), 0.15)
        self.assertEqual(float(shaped[3]), 0.0)
        np.testing.assert_array_equal(rewards, np.zeros(4, dtype=np.float32))

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

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_exploratory_loss_default_matches_original_control_exactly(self):
        import torch
        from farm_sim.torch_training import ActorCritic, Trajectory, calculate_loss
        model = ActorCritic(seed=30, value_head=True)
        observations = np.zeros((2, 1, len(FEATURE_NAMES)), np.float32)
        actions = np.zeros((2, 1), np.int64)
        rewards = np.array([[0.25], [-0.1]], np.float32)
        trajectory = Trajectory(observations, actions, rewards, np.zeros_like(rewards), 7)
        original_loss, original_metrics = calculate_loss(model, [trajectory], "a2c")
        exploratory_loss, exploratory_metrics = calculate_exploratory_loss(model, [trajectory], "a2c")
        torch.testing.assert_close(original_loss, exploratory_loss, rtol=0, atol=0)
        self.assertEqual(original_metrics, exploratory_metrics)
        model.zero_grad(set_to_none=True)
        original_loss, _ = calculate_loss(model, [trajectory], "a2c")
        original_loss.backward()
        control_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)
        exploratory_loss, _ = calculate_exploratory_loss(model, [trajectory], "a2c")
        exploratory_loss.backward()
        for expected, parameter in zip(control_gradients, model.parameters()):
            torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_exploratory_loss_applies_opt_in_entropy_and_advantage_options(self):
        import torch
        from farm_sim.torch_training import ActorCritic, Trajectory
        model = ActorCritic(seed=31, value_head=True)
        observations = np.zeros((2, 1, len(FEATURE_NAMES)), np.float32)
        observations[:, :, FEATURE_NAMES.index("fatigue")] = np.array([0.0, 1.0])[:, None]
        actions = np.full((2, 1), list(Action).index(Action.REST), np.int64)
        rewards = np.zeros((2, 1), np.float32)
        trajectory = Trajectory(observations, actions, rewards, np.zeros_like(rewards), 8)
        control, base = calculate_exploratory_loss(model, [trajectory], "a2c")
        shaped, shaped_stats = calculate_exploratory_loss(
            model, [trajectory], "a2c", reward_variant="state_aware")
        self.assertGreater(shaped_stats["mean_return_to_go"], base["mean_return_to_go"])
        regularized, _ = calculate_exploratory_loss(
            model, [trajectory], "a2c", entropy_coefficient=0.2)
        torch.testing.assert_close(regularized, control - 0.2 * base["entropy"])
        normalized, normalized_stats = calculate_exploratory_loss(
            model, [trajectory], "a2c", normalize_advantage=True)
        self.assertTrue(torch.isfinite(normalized))
        self.assertAlmostEqual(normalized_stats["mean_advantage"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
