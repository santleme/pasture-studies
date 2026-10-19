"""Seeded discrete simulation orchestrator with explicit component boundaries."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import replace
from typing import Mapping, Sequence

from .memory import remember, retrieve
from .model import (
    Action, ActionPolicy, AnimalMemory, AnimalState, FarmWorld, MemoryEvent,
    PolicyDecision, Position, ResourcePatch, SimEvent, TickResult, ToyPhysiology,
)
from .navigation import navigate
from .perception import encode_features, perceive, manhattan
from .perception import PERCEPTION_MODES
from .physiology import step_physiology

ANIMAL_COUNT = 5
EVENT_HISTORY_LIMIT = 5_000
# A deliberately abstract unit convention: 24 committed simulation ticks form a
# synthetic virtual day. This is not a mapping to real hours or animal time.
TICKS_PER_VIRTUAL_DAY = 24


def synthetic_time(tick: int) -> dict[str, int]:
    """Describe the next abstract interval using a reset-scoped absolute tick.

    Tick ``n`` denotes interval ``[n, n + 1)``; after it commits, the world
    clock is ``n + 1``. Thus a committed tick count of 24 is day 1, tick 0.
    """
    if tick < 0:
        raise ValueError("tick must be non-negative")
    return {
        "absolute_tick": tick,
        "synthetic_day_index": tick // TICKS_PER_VIRTUAL_DAY,
        "tick_within_day": tick % TICKS_PER_VIRTUAL_DAY,
        "ticks_per_virtual_day": TICKS_PER_VIRTUAL_DAY,
    }


def make_world(seed: int = 7, *, width: int = 14, height: int = 10) -> FarmWorld:
    """Create the reproducible five-cow starter scenario."""
    rng = random.Random(seed)
    spots = [Position(x, y) for y in range(1, height - 1) for x in range(1, width - 1)]
    rng.shuffle(spots)
    obstacles = frozenset({Position(width // 2, y) for y in (2, 3, 4)} | {Position(width // 2, y) for y in (6, 7)})
    spots = [p for p in spots if p not in obstacles]
    animals = tuple(AnimalState(
        animal_id=f"cow-{i + 1}", position=spots[i],
        physiology=ToyPhysiology(hunger=0.25 + rng.random() * 0.5,
                                thirst=0.2 + rng.random() * 0.6,
                                fatigue=0.1 + rng.random() * 0.5,
                                social_need=0.2 + rng.random() * 0.5),
        memory=AnimalMemory(),
    ) for i in range(ANIMAL_COUNT))
    forage = (ResourcePatch(Position(2, 2)), ResourcePatch(Position(3, height - 3)),
              ResourcePatch(Position(width - 3, 2)))
    water = (ResourcePatch(Position(width - 3, height - 3)),)
    return FarmWorld(width, height, animals, forage, water, obstacles, seed=seed)


def make_resource_rich_world(seed: int = 7, *, forage_count: int = 8, water_count: int = 5) -> FarmWorld:
    """Build a nonformal training-distribution variant with finite extra patches.

    The default starter environment and all transition/reward equations remain
    untouched. This opt-in variant only adds finite forage/water locations so
    learning rollouts are not dominated by one-time shared-source scarcity.
    It is not calibrated to cattle or used as the common evaluation task.
    """
    if forage_count < 3 or water_count < 1:
        raise ValueError("resource-rich world must retain at least three forage and one water patch")
    world = make_world(seed)
    occupied = (set(world.obstacles) | {a.position for a in world.animals}
                | {p.position for p in world.forage} | {p.position for p in world.water})
    cells = [Position(x, y) for y in range(1, world.height - 1)
             for x in range(1, world.width - 1) if Position(x, y) not in occupied]
    random.Random(seed ^ 0x5A17C0DE).shuffle(cells)
    extra_foraging = max(0, forage_count - len(world.forage))
    extra_watering = max(0, water_count - len(world.water))
    if extra_foraging + extra_watering > len(cells):
        raise ValueError("requested resource counts exceed available free cells")
    added = cells[:extra_foraging + extra_watering]
    forage = world.forage + tuple(ResourcePatch(position, True) for position in added[:extra_foraging])
    water = world.water + tuple(ResourcePatch(position, True)
                                for position in added[extra_foraging:])
    return replace(world, forage=forage, water=water)


class Simulation:
    """Owns logical time and component orchestration; policy is injected, never fallback."""
    def __init__(self, world: FarmWorld | None, policy: ActionPolicy, *, perception_mode: str | None = None):
        if policy is None:
            raise ValueError("A local policy must be injected; no rule fallback is provided")
        self.world = world if world is not None else make_world()
        if len(self.world.animals) != ANIMAL_COUNT:
            raise ValueError(f"simulation requires exactly {ANIMAL_COUNT} animals")
        self.policy = policy
        self.perception_mode = perception_mode or getattr(policy, "perception_mode", "legacy-v1")
        if self.perception_mode not in PERCEPTION_MODES:
            raise ValueError(f"unknown perception mode {self.perception_mode!r}")
        # IDs are assigned before retention trimming; they remain stable and strictly
        # increasing for one reset/replay even after old records are evicted.
        self.events: deque[SimEvent] = deque(maxlen=EVENT_HISTORY_LIMIT)
        self.event_ids: deque[int] = deque(maxlen=EVENT_HISTORY_LIMIT)
        self._next_event_id = 0
        self._evicted_event_count = 0

    def reset(self, seed: int = 7) -> FarmWorld:
        self.world = make_world(seed)
        self.events.clear()
        self.event_ids.clear()
        self._next_event_id = 0
        self._evicted_event_count = 0
        return self.world

    def append_events(self, events: tuple[SimEvent, ...]) -> None:
        """Retain the newest bounded event window and assign reset-scoped cursors."""
        for event in events:
            if len(self.events) == EVENT_HISTORY_LIMIT:
                self._evicted_event_count += 1
            self.events.append(event)
            self.event_ids.append(self._next_event_id)
            self._next_event_id += 1

    def event_retention(self, after: int | None = None) -> dict[str, int | bool | None]:
        oldest_id = self.event_ids[0] if self.event_ids else None
        latest_id = self.event_ids[-1] if self.event_ids else None
        return {
            "limit": EVENT_HISTORY_LIMIT,
            "retained_count": len(self.events),
            "oldest_id": oldest_id,
            "latest_id": latest_id,
            "evicted_count": self._evicted_event_count,
            "has_older_events": self._evicted_event_count > 0,
            "has_gap_after_cursor": bool(after is not None and oldest_id is not None and after < oldest_id - 1),
        }

    def step(self) -> TickResult:
        """Advance exactly one logical tick; use stable ID ordering for arbitration."""
        return self._step_with_provider(None)

    def decision_features(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        """Gather all decisions' features from this world's pre-transition snapshot."""
        starting = self.world
        rows = []
        for animal in sorted(starting.animals, key=lambda a: a.animal_id):
            observation = perceive(starting, animal, mode=self.perception_mode)
            memories = retrieve(animal.memory, starting.tick)
            features = encode_features(observation, memories, animal, starting.width, starting.height)
            rows.append((animal.animal_id, tuple(features)))
        return tuple(rows)

    def step_with_decisions(self, decisions: Mapping[str, tuple[PolicyDecision, Sequence[float]]]) -> TickResult:
        """Commit precomputed decisions through normal sequential arbitration.

        This is the vectorized-inference boundary. Features must be those returned
        by ``decision_features`` for the current world tick; all physical updates,
        resource conflicts, event order, and commit semantics remain scalar.
        """
        expected = {animal.animal_id for animal in self.world.animals}
        if set(decisions) != expected:
            raise ValueError("decisions must contain exactly one entry per animal")
        current_features = dict(self.decision_features())
        for animal_id, (_, features) in decisions.items():
            # Validate the caller's snapshot before any sequential arbitration
            # mutates resources or animals. This rejects stale-tick batches as
            # well as accidentally cross-world feature rows.
            if tuple(features) != current_features[animal_id]:
                raise ValueError("decision features do not match the current world snapshot")
        return self._step_with_provider(decisions)

    def _step_with_provider(self, decisions: Mapping[str, tuple[PolicyDecision, Sequence[float]]] | None) -> TickResult:
        starting = self.world
        working_animals = {a.animal_id: a for a in starting.animals}
        events: list[SimEvent] = []
        forage = list(starting.forage)
        water = list(starting.water)
        # One decision and at most one movement update per individual per tick.
        for original in sorted(starting.animals, key=lambda a: a.animal_id):
            current = working_animals[original.animal_id]
            # Each agent perceives the same pre-transition snapshot; only navigation
            # arbitration is sequential, so another cow's new position cannot leak
            # into this tick's observation.
            if decisions is None:
                decision_observation = perceive(starting, current, mode=self.perception_mode)
                memories = retrieve(current.memory, starting.tick)
                features = encode_features(decision_observation, memories, current, starting.width, starting.height)
                # Policy contract is pure: only an action and probability vector come back.
                decision: PolicyDecision = self.policy.decide(features)
            else:
                decision, features = decisions[current.animal_id]
            if len(decision.probabilities) != len(Action):
                raise ValueError("policy probability vector must align with Action enum")
            if decision.action not in Action:
                raise ValueError("policy selected an unsupported action")
            occupied = {a.position for a in working_animals.values() if a.animal_id != current.animal_id}
            nav = navigate(replace(starting, forage=tuple(forage), water=tuple(water),
                                   animals=tuple(working_animals.values())), current, decision.action, occupied)
            # Consumable availability is evaluated at the post-navigation location.
            forage_at_position = next((i for i, p in enumerate(forage) if p.position == nav.position), None)
            water_at_position = next((i for i, p in enumerate(water) if p.position == nav.position), None)
            forage_available = (forage[forage_at_position].available if forage_at_position is not None else None)
            water_available = (water[water_at_position].available if water_at_position is not None else None)
            nearby = any(manhattan(nav.position, p) <= 2 for p in occupied)
            updated_phys, reward, labels = step_physiology(
                current.physiology, decision.action,
                forage_available=forage_available, water_available=water_available,
                herd_nearby=nearby,
            )
            if decision.action == Action.GRAZE and forage_at_position is not None and forage_available:
                forage[forage_at_position] = replace(forage[forage_at_position], available=False)
            if decision.action == Action.DRINK and water_at_position is not None and water_available:
                water[water_at_position] = replace(water[water_at_position], available=False)
            memory = current.memory
            for label in labels:
                if label in ("water_success", "water_blocked"):
                    memory = remember(memory, MemoryEvent(starting.tick, "water_access", label == "water_success", nav.position))
                elif label in ("forage_success", "forage_depleted"):
                    memory = remember(memory, MemoryEvent(starting.tick, "forage_patch", label == "forage_success", nav.position))
            updated = replace(current, position=nav.position, physiology=updated_phys, memory=memory,
                              action=decision.action, last_reward=reward,
                              last_probabilities=tuple(float(p) for p in decision.probabilities),
                              last_features=tuple(features))
            working_animals[current.animal_id] = updated
            events.append(SimEvent(starting.tick + 1, current.animal_id, "decision",
                                   f"{decision.action.value} via {decision.policy_id}", reward))
            if nav.moved:
                events.append(SimEvent(starting.tick + 1, current.animal_id, "movement",
                                       f"{current.position.x},{current.position.y}->{nav.position.x},{nav.position.y}"))
            for label in labels:
                events.append(SimEvent(starting.tick + 1, current.animal_id, label, label, reward))
        self.world = replace(starting, tick=starting.tick + 1,
                             animals=tuple(working_animals[a.animal_id] for a in starting.animals),
                             forage=tuple(forage), water=tuple(water))
        self.append_events(tuple(events))
        return TickResult(self.world, tuple(events))


def serialize_world(world: FarmWorld) -> dict:
    """Return a detached JSON-friendly view; this does not mutate simulation state."""
    return {
        "tick": world.tick, **synthetic_time(world.tick),
        "seed": world.seed, "width": world.width, "height": world.height,
        "animals": [{
            "id": a.animal_id, "x": a.position.x, "y": a.position.y,
            "action": a.action.value,
            "physiology": {"hunger": a.physiology.hunger, "thirst": a.physiology.thirst,
                           "fatigue": a.physiology.fatigue, "social_need": a.physiology.social_need},
            "reward": a.last_reward, "probabilities": list(a.last_probabilities),
            "features": list(a.last_features),
            "memory": [{"tick": e.tick, "kind": e.kind, "success": e.success,
                        "x": e.position.x, "y": e.position.y} for e in a.memory.events],
        } for a in world.animals],
        "forage": [{"x": p.position.x, "y": p.position.y, "available": p.available} for p in world.forage],
        "water": [{"x": p.position.x, "y": p.position.y, "available": p.available} for p in world.water],
        "obstacles": [{"x": p.x, "y": p.y} for p in sorted(world.obstacles)],
    }
