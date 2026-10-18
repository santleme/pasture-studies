"""Synthetic clock contract and reproducibility tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import farm_sim.simulation as simulation_module
from farm_sim.model import Action, PolicyDecision
from farm_sim.policy import load_checkpoint
from farm_sim.server import DEFAULT_CHECKPOINT
from farm_sim.simulation import TICKS_PER_VIRTUAL_DAY, Simulation, make_world, serialize_world, synthetic_time


# This replay contract is bound to the exact local checkpoint shipped with MVP1.
DEFAULT_CHECKPOINT_SHA256 = "129be7419eb690bd99f55abc7c4434e30a80e818af2ebf304fcf6560d7d78cdd"
PAIRED_REPLAY_SEEDS = tuple(range(72_001, 72_101))
PAIRED_REPLAY_TICKS = 3 * TICKS_PER_VIRTUAL_DAY


class ClockPolicy:
    def decide(self, features):
        probabilities = tuple(1.0 if action == Action.REST else 0.0 for action in Action)
        return PolicyDecision(Action.REST, probabilities, "clock-test")


class SyntheticTimeTests(TestCase):
    def test_day_boundaries_are_half_open_and_reset_scoped(self):
        self.assertEqual(TICKS_PER_VIRTUAL_DAY, 24)
        sim = Simulation(make_world(19), ClockPolicy())
        self.assertEqual(synthetic_time(sim.world.tick)["synthetic_day_index"], 0)
        for expected in range(1, 25):
            world = sim.step().world
            self.assertEqual(world.tick, expected)
            clock = synthetic_time(world.tick)
            if expected == 23:
                self.assertEqual((clock["synthetic_day_index"], clock["tick_within_day"]), (0, 23))
            if expected == 24:
                self.assertEqual((clock["synthetic_day_index"], clock["tick_within_day"]), (1, 0))
        for expected in range(25, 49):
            sim.step()
            if expected == 47:
                self.assertEqual((synthetic_time(expected)["synthetic_day_index"],
                                  synthetic_time(expected)["tick_within_day"]), (1, 23))
            if expected == 48:
                self.assertEqual((synthetic_time(expected)["synthetic_day_index"],
                                  synthetic_time(expected)["tick_within_day"]), (2, 0))
        reset = sim.reset(19)
        self.assertEqual(reset.tick, 0)
        self.assertEqual(synthetic_time(reset.tick)["synthetic_day_index"], 0)

    def test_each_tick_updates_physiology_once_per_animal(self):
        sim = Simulation(make_world(31), ClockPolicy())
        with patch("farm_sim.simulation.step_physiology",
                   wraps=simulation_module.step_physiology) as update:
            sim.step()
        self.assertEqual(update.call_count, 5)
        self.assertEqual(sim.world.tick, 1)

    def test_one_hundred_seeded_day_replays_are_identical(self):
        for seed in range(100):
            first, second = Simulation(make_world(seed), ClockPolicy()), Simulation(make_world(seed), ClockPolicy())
            for _ in range(TICKS_PER_VIRTUAL_DAY):
                first.step()
                second.step()
            self.assertEqual(serialize_world(first.world), serialize_world(second.world), f"seed {seed}")
            self.assertEqual(tuple(first.events), tuple(second.events), f"seed {seed}")

    def test_shipped_policy_has_identical_paired_replays_for_100_seeds_and_three_days(self):
        checkpoint = Path(DEFAULT_CHECKPOINT)
        actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(actual_sha256, DEFAULT_CHECKPOINT_SHA256, "shipped policy checkpoint changed")
        policy = load_checkpoint(checkpoint)

        for seed in PAIRED_REPLAY_SEEDS:
            first = Simulation(make_world(seed), policy)
            second = Simulation(make_world(seed), policy)
            for _ in range(PAIRED_REPLAY_TICKS):
                first.step()
                second.step()
            with self.subTest(seed=seed):
                self.assertEqual(first.world.tick, PAIRED_REPLAY_TICKS)
                self.assertEqual(serialize_world(first.world), serialize_world(second.world))
                self.assertEqual(tuple(first.events), tuple(second.events))
