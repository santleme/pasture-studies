"""Small deterministic viability tests; deliberately do not train a policy."""
from __future__ import annotations

import unittest

from farm_sim.model import Action, ToyPhysiology
from farm_sim.physiology import step_physiology
from validation.viability_v2 import marginal_contact_reward, run_behavior


class LearningSolvabilityV2Tests(unittest.TestCase):
    def test_saturated_social_contact_has_no_marginal_reward_in_v2(self):
        state = ToyPhysiology(hunger=0.9, thirst=0.9, fatigue=0.9, social_need=0.0)
        _, old_reward, _ = step_physiology(state, Action.HERD_INTERACT, herd_nearby=True)
        updated, _, _ = step_physiology(state, Action.HERD_INTERACT, herd_nearby=True)
        v2_reward = marginal_contact_reward(old_reward, state.social_need,
                                            updated.social_need)
        self.assertAlmostEqual(old_reward, 0.0302)
        self.assertAlmostEqual(v2_reward, -0.0698)

        unmet = ToyPhysiology(hunger=0.5, thirst=0.5, fatigue=0.5, social_need=0.5)
        next_state, reward, _ = step_physiology(unmet, Action.HERD_INTERACT,
                                                herd_nearby=True)
        self.assertLess(next_state.social_need, unmet.social_need)
        self.assertAlmostEqual(
            marginal_contact_reward(reward, unmet.social_need,
                                    next_state.social_need), reward
        )

    def test_fixed_resource_world_has_four_success_budget(self):
        # Three forage patches + one water patch; each can be consumed once.
        for seed in (11, 29, 47, 83, 101):
            events, _ = run_behavior(seed, regrowth_ticks=None)
            self.assertEqual(events["forage_success"] + events["water_success"], 4)

    def test_renewable_scenario_is_solvable_on_behavior_metrics_without_training(self):
        fixed_successes = []
        renewable_successes = []
        fixed_drive = []
        renewable_drive = []
        for seed in (11, 29, 47, 83, 101):
            fixed, fixed_sum = run_behavior(seed, regrowth_ticks=None)
            renewed, renewed_sum = run_behavior(seed, regrowth_ticks=3)
            fixed_successes.append(fixed["forage_success"] + fixed["water_success"])
            renewable_successes.append(renewed["forage_success"] + renewed["water_success"])
            fixed_drive.append(fixed_sum)
            renewable_drive.append(renewed_sum)
        self.assertEqual(fixed_successes, [4] * 5)
        self.assertGreater(min(renewable_successes), 30)
        self.assertGreater(sum(renewable_successes), 10 * sum(fixed_successes))
        # Hunger + thirst are an external behavior metric, not the optimized reward.
        self.assertLess(sum(renewable_drive) / 5, sum(fixed_drive) / 5)


if __name__ == "__main__":
    unittest.main()
