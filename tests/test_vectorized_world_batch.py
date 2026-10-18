import unittest
from dataclasses import replace

import numpy as np

from farm_sim.model import Action, PolicyDecision
from farm_sim.simulation import Simulation, make_world


class FeaturePolicy:
    def decide(self, features):
        # A fixed, feature-sensitive mix deliberately produces collisions.
        idx = int((features[0] * 13 + features[1] * 7 + features[5] * 3) * 10) % len(Action)
        probabilities = [0.0] * len(Action)
        probabilities[idx] = 1.0
        return PolicyDecision(tuple(Action)[idx], tuple(probabilities), "parity-test")


class VectorizedWorldBatchTests(unittest.TestCase):
    def test_decision_batch_commit_matches_scalar_step_for_100_seeds_72_ticks(self):
        for seed in range(100):
            scalar = Simulation(make_world(seed), FeaturePolicy())
            precomputed = Simulation(make_world(seed), FeaturePolicy())
            for _ in range(72):
                feature_rows = precomputed.decision_features()
                decisions = {}
                for animal_id, features in feature_rows:
                    decisions[animal_id] = (precomputed.policy.decide(features), features)
                scalar_result = scalar.step()
                batch_result = precomputed.step_with_decisions(decisions)
                self.assertEqual(scalar_result.world, batch_result.world, (seed, scalar.world.tick))
                self.assertEqual(scalar_result.events, batch_result.events, (seed, scalar.world.tick))
            self.assertEqual(scalar.world.tick, 72)

    def test_missing_or_extra_decision_rejected_without_committing(self):
        sim = Simulation(make_world(4), FeaturePolicy())
        rows = sim.decision_features()
        decisions = {animal_id: (sim.policy.decide(features), features) for animal_id, features in rows[:-1]}
        with self.assertRaises(ValueError):
            sim.step_with_decisions(decisions)
        self.assertEqual(sim.world.tick, 0)

    def test_stale_snapshot_features_rejected_before_world_commit(self):
        from farm_sim.model import Position

        sim = Simulation(make_world(14), FeaturePolicy())
        rows = sim.decision_features()
        decisions = {animal_id: (sim.policy.decide(features), features) for animal_id, features in rows}
        first = sim.world.animals[0]
        moved = replace(first, position=Position(first.position.x + 1, first.position.y))
        sim.world = replace(sim.world, animals=(moved, *sim.world.animals[1:]))
        before = sim.world
        with self.assertRaisesRegex(ValueError, "current world snapshot"):
            sim.step_with_decisions(decisions)
        self.assertEqual(sim.world, before)


try:
    import torch
except ImportError:  # optional training dependency
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch training dependency is not installed")
class BatchedCollectorTests(unittest.TestCase):
    def test_actor_critic_scalar_vs_batched_policy_parity_for_100_seeds_72_ticks(self):
        from farm_sim.torch_training import ActorCritic, N_ACTIONS, derive_seed

        model = ActorCritic(seed=29).eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for seed in range(100):
                scalar = Simulation(make_world(seed), FeaturePolicy())
                batched = Simulation(make_world(seed), FeaturePolicy())
                for tick in range(72):
                    scalar_rows = scalar.decision_features()
                    batched_rows = batched.decision_features()
                    self.assertEqual(scalar_rows, batched_rows)
                    batch_x = torch.as_tensor(np.asarray([features for _, features in batched_rows], dtype=np.float32), device=device)
                    batch_probabilities = model.probabilities(batch_x).cpu().numpy()
                    scalar_probabilities = np.stack([
                        model.probabilities(torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)).cpu().numpy()
                        for _, features in scalar_rows
                    ])
                    np.testing.assert_allclose(batch_probabilities, scalar_probabilities, rtol=1e-6, atol=1e-7,
                                               err_msg=f"seed={seed}, tick={tick}")

                    scalar_decisions = {}
                    batch_decisions = {}
                    for animal_index, ((animal_id, scalar_features), (_, batch_features)) in enumerate(zip(scalar_rows, batched_rows)):
                        scalar_p = scalar_probabilities[animal_index]
                        batch_p = batch_probabilities[animal_index]
                        uniform_rng = np.random.default_rng(derive_seed(99173, "action-uniform", seed, seed, tick, animal_index))
                        uniform = float(uniform_rng.random())
                        scalar_action = min(N_ACTIONS - 1, int(np.searchsorted(np.cumsum(scalar_p), uniform, side="right")))
                        batch_action = min(N_ACTIONS - 1, int(np.searchsorted(np.cumsum(batch_p), uniform, side="right")))
                        self.assertEqual(scalar_action, batch_action, f"seed={seed}, tick={tick}, animal={animal_id}")
                        scalar_decisions[animal_id] = (PolicyDecision(tuple(Action)[scalar_action], tuple(float(p) for p in scalar_p), "torch-sampled-batched"), scalar_features)
                        batch_decisions[animal_id] = (PolicyDecision(tuple(Action)[batch_action], tuple(float(p) for p in batch_p), "torch-sampled-batched"), batch_features)

                    scalar_result = scalar.step_with_decisions(scalar_decisions)
                    batch_result = batched.step_with_decisions(batch_decisions)
                    # Probability telemetry is compared with the declared numeric
                    # tolerance above; strip only that telemetry before requiring
                    # exact physical/world state and event equality.
                    scalar_world = replace(scalar_result.world, animals=tuple(
                        replace(animal, last_probabilities=()) for animal in scalar_result.world.animals))
                    batch_world = replace(batch_result.world, animals=tuple(
                        replace(animal, last_probabilities=()) for animal in batch_result.world.animals))
                    self.assertEqual(scalar_world, batch_world, (seed, tick))
                    self.assertEqual(scalar_result.events, batch_result.events, (seed, tick))

    def test_batch_order_and_environment_count_do_not_change_trajectories(self):
        from farm_sim.torch_training import ActorCritic, collect_episodes_batch

        model = ActorCritic(seed=14)
        specs = [(101, 0), (202, 1), (303, 2)]
        full = collect_episodes_batch(model, specs, root_seed=77)
        reordered = collect_episodes_batch(model, list(reversed(specs)), root_seed=77)
        subset = collect_episodes_batch(model, [specs[1]], root_seed=77)[0]
        indexed = {trajectory.seed: trajectory for trajectory in full}
        reordered_indexed = {trajectory.seed: trajectory for trajectory in reordered}
        for seed in (101, 202, 303):
            for field in ("observations", "actions", "rewards", "uniforms"):
                np.testing.assert_array_equal(getattr(indexed[seed], field), getattr(reordered_indexed[seed], field))
            np.testing.assert_allclose(indexed[seed].probabilities, reordered_indexed[seed].probabilities,
                                       rtol=1e-6, atol=1e-7)
        for field in ("observations", "actions", "rewards", "uniforms", "probabilities"):
            if field == "probabilities":
                np.testing.assert_allclose(getattr(indexed[202], field), getattr(subset, field), rtol=1e-6, atol=1e-7)
            else:
                np.testing.assert_array_equal(getattr(indexed[202], field), getattr(subset, field))

    def test_batch_uses_one_policy_forward_per_tick_not_per_animal(self):
        from farm_sim.torch_training import ActorCritic, collect_episodes_batch

        model = ActorCritic(seed=3)
        calls = []
        original = model.probabilities

        def counted(observations):
            calls.append(tuple(observations.shape))
            return original(observations)

        model.probabilities = counted
        trajectories = collect_episodes_batch(model, [(1, 0), (2, 1)], root_seed=8)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(len(calls), 72)
        self.assertTrue(all(shape == (10, 20) for shape in calls))


if __name__ == "__main__":
    unittest.main()
