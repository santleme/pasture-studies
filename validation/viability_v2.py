"""Readily reproducible solvability prototype; does not change the legacy simulator."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

from farm_sim.model import Action, PolicyDecision
from farm_sim.simulation import Simulation, make_world


class NeedController:
    """Transparent hand-coded oracle; used to test feasibility, never training."""

    def decide(self, features):
        hunger, thirst, fatigue = features[:3]
        forage_available, water_available = features[8:10]
        choices = []
        if forage_available:
            choices.append((hunger, Action.GRAZE))
        if water_available:
            choices.append((thirst, Action.DRINK))
        if not choices:
            action = Action.WALK  # legacy navigation provides deterministic pathfinding
        else:
            unmet, candidate = max(choices, key=lambda item: item[0])
            action = candidate if unmet > fatigue else Action.REST
        probabilities = tuple(1.0 if item == action else 0.0 for item in Action)
        return PolicyDecision(action, probabilities, "solvability-oracle")


def marginal_contact_reward(reward: float, social_need_before: float,
                            social_need_after: float) -> float:
    """Prototype correction: do not award the +0.1 contact bonus without progress."""
    return reward - 0.1 if social_need_after >= social_need_before else reward


def run_behavior(seed: int, *, regrowth_ticks: int | None) -> tuple[Counter, float]:
    """Run a fixed 72-tick behavior probe with optional external resource regrowth."""
    sim = Simulation(make_world(seed), NeedController())
    events = Counter()
    regrow_at: dict[tuple[str, int], int] = {}
    for _ in range(72):
        tick = sim.world.tick
        if regrowth_ticks is not None:
            def refresh(patches, kind):
                result = []
                for index, patch in enumerate(patches):
                    if not patch.available and regrow_at.get((kind, index), tick + 1) <= tick:
                        result.append(replace(patch, available=True))
                    else:
                        result.append(patch)
                return tuple(result)
            sim.world = replace(sim.world,
                                forage=refresh(sim.world.forage, "forage"),
                                water=refresh(sim.world.water, "water"))
        before = sim.world
        result = sim.step()
        for kind in ("forage", "water"):
            old_patches = getattr(before, kind)
            new_patches = getattr(sim.world, kind)
            for index, (old, new) in enumerate(zip(old_patches, new_patches)):
                if old.available and not new.available and regrowth_ticks is not None:
                    regrow_at[(kind, index)] = tick + 1 + regrowth_ticks
        events.update(event.kind for event in result.events
                      if event.kind in {"forage_success", "water_success"})
    mean_final_hunger_thirst = sum(
        animal.physiology.hunger + animal.physiology.thirst
        for animal in sim.world.animals
    ) / len(sim.world.animals)
    return events, mean_final_hunger_thirst
