"""Focused deterministic tests for the independent simulation-core contracts."""
from dataclasses import replace
import unittest

from farm_sim.memory import remember, retrieve
from farm_sim.model import (
    Action, AnimalState, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FarmWorld,
    MemoryEvent, PolicyDecision, Position, ResourcePatch, SimEvent, ToyPhysiology,
)
from farm_sim.navigation import navigate
from farm_sim.perception import encode_features, perceive
from farm_sim.physiology import step_physiology
from farm_sim.simulation import EVENT_HISTORY_LIMIT, Simulation, make_world, serialize_world


class FeaturePolicy:
    """A deterministic test double; not a production decision policy."""
    def decide(self, features):
        # Vary with the encoded logical-time phase while remaining side-effect-free.
        import math
        phase = math.atan2(features[17], features[18])
        index = int(round((phase % (2 * math.pi)) / (2 * math.pi) * len(Action))) % len(Action)
        probs = tuple(1.0 if i == index else 0.0 for i, _ in enumerate(Action))
        return PolicyDecision(tuple(Action)[index], probs, "test-double")


class SimulationCoreTests(unittest.TestCase):
    def test_seeded_world_and_replay_are_deterministic(self):
        self.assertEqual(make_world(41), make_world(41))
        first, second = Simulation(make_world(41), FeaturePolicy()), Simulation(make_world(41), FeaturePolicy())
        snapshots_a, snapshots_b = [], []
        for _ in range(5):
            result_a, result_b = first.step(), second.step()
            snapshots_a.append((serialize_world(result_a.world), result_a.events))
            snapshots_b.append((serialize_world(result_b.world), result_b.events))
        self.assertEqual(snapshots_a, snapshots_b)
        self.assertEqual(first.world.tick, 5)
        self.assertEqual(len(first.world.animals), 5)

    def test_five_animals_are_required(self):
        world = make_world(5)
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            Simulation(replace(world, animals=world.animals[:4]), FeaturePolicy())
        with self.assertRaisesRegex(ValueError, "must be injected"):
            Simulation(world, None)

    def test_versioned_features_include_memory_and_all_named_inputs(self):
        world = make_world(3)
        animal = world.animals[0]
        obs = perceive(world, animal)
        empty = retrieve(animal.memory, world.tick)
        initial = encode_features(obs, empty, animal, world.width, world.height)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "cattle-observation-v1")
        self.assertEqual(len(initial), len(FEATURE_NAMES))
        event = MemoryEvent(world.tick, "water_access", True, animal.position)
        changed_animal = replace(animal, memory=remember(animal.memory, event))
        changed = encode_features(obs, retrieve(changed_animal.memory, world.tick), changed_animal,
                                  world.width, world.height)
        self.assertNotEqual(initial, changed)
        self.assertEqual(changed[FEATURE_NAMES.index("mem_water_success")], 1.0)
        self.assertEqual(changed[FEATURE_NAMES.index("mem_water_blocked")], 0.0)
        # Exactly one designated marker changes for the paired intervention.
        diffs = [name for name, a, b in zip(FEATURE_NAMES, initial, changed) if a != b]
        self.assertEqual(diffs, ["mem_water_success", "mem_event_recency"])

    def test_policy_and_serialization_do_not_mutate_world(self):
        world = make_world(13)
        frozen_snapshot = serialize_world(world)
        Simulation(world, FeaturePolicy()).step()
        self.assertEqual(serialize_world(world), frozen_snapshot)
        # Rendering/API-facing serialization returns detached mutable containers.
        exposed = serialize_world(world)
        exposed["animals"][0]["physiology"]["hunger"] = -99
        self.assertEqual(serialize_world(world), frozen_snapshot)

    def test_physiology_and_navigation_are_independent(self):
        world = make_world(21)
        animal = world.animals[0]
        next_state, _, _ = step_physiology(animal.physiology, Action.REST)
        self.assertNotEqual(next_state, animal.physiology)
        self.assertEqual(animal.position, world.animals[0].position)
        moved = navigate(world, animal, Action.WALK)
        self.assertTrue(0 <= moved.position.x < world.width)
        self.assertTrue(0 <= moved.position.y < world.height)
        self.assertEqual(animal.physiology, world.animals[0].physiology)
        # Navigation only returns intent resolution; it cannot change its input.
        self.assertEqual(world, make_world(21))

    def test_step_advances_one_logical_tick_and_exposes_decisions(self):
        sim = Simulation(make_world(10), FeaturePolicy())
        result = sim.step()
        self.assertEqual(result.world.tick, 1)
        self.assertEqual(sum(e.kind == "decision" for e in result.events), 5)
        self.assertTrue(all(len(a.last_features) == len(FEATURE_NAMES) for a in result.world.animals))
        self.assertTrue(all(len(a.last_probabilities) == len(Action) for a in result.world.animals))
        self.assertEqual(len(sim.events), len(result.events))

    def test_event_history_is_bounded_and_ids_remain_monotonic_after_trim(self):
        sim = Simulation(make_world(12), FeaturePolicy())
        sim.append_events(tuple(SimEvent(i, "cow-1", "test", str(i))
                                for i in range(EVENT_HISTORY_LIMIT + 3)))
        self.assertEqual(len(sim.events), EVENT_HISTORY_LIMIT)
        self.assertEqual(list(sim.event_ids), list(range(3, EVENT_HISTORY_LIMIT + 3)))
        retention = sim.event_retention(after=0)
        self.assertEqual(retention["oldest_id"], 3)
        self.assertEqual(retention["latest_id"], EVENT_HISTORY_LIMIT + 2)
        self.assertTrue(retention["has_older_events"])
        self.assertTrue(retention["has_gap_after_cursor"])
        sim.reset(12)
        self.assertEqual(len(sim.events), 0)
        self.assertEqual(sim.event_retention()["oldest_id"], None)
        # Replay/reset begins a fresh ID sequence.
        sim.append_events((SimEvent(1, "cow-1", "test", "after reset"),))
        self.assertEqual(list(sim.event_ids), [0])


if __name__ == "__main__":
    unittest.main()
