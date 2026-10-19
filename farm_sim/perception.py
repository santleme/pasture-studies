"""Bounded local perception and fixed observation feature encoding."""
from __future__ import annotations

import math
from typing import Sequence

from .model import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, AnimalState, FarmWorld, MemoryFeatures,
    Observation, Position,
)
from .navigation import static_path_distance

VISION_RADIUS = 4
HERD_RADIUS = 3
PERCEPTION_MODES = {"legacy-v1", "path-aware-v2"}


def manhattan(a: Position, b: Position) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def perceive(world: FarmWorld, animal: AnimalState, *, mode: str = "legacy-v1") -> Observation:
    if mode not in PERCEPTION_MODES:
        raise ValueError(f"unknown perception mode {mode!r}")
    def visible(patch):
        return manhattan(animal.position, patch.position) <= VISION_RADIUS
    forage = tuple(p for p in world.forage if visible(p))
    water = tuple(p for p in world.water if visible(p))
    others = tuple(a.position for a in world.animals if a.animal_id != animal.animal_id
                   and manhattan(animal.position, a.position) <= HERD_RADIUS)
    nearby_cells = [Position(x, y) for x in range(max(0, animal.position.x - 1), min(world.width, animal.position.x + 2))
                    for y in range(max(0, animal.position.y - 1), min(world.height, animal.position.y + 2))]
    blocked = sum(p in world.obstacles for p in nearby_cells)
    denom = max(1, len(nearby_cells))
    if mode == "legacy-v1":
        forage_distance = min((manhattan(animal.position, p.position) for p in forage), default=VISION_RADIUS + 1)
        water_distance = min((manhattan(animal.position, p.position) for p in water), default=VISION_RADIUS + 1)
    else:
        # V2 corrects stale target distances and reports obstacle-aware travel
        # costs using the same static grid topology as the navigation layer.
        def target_distance(patches):
            distances = (static_path_distance(world, animal.position, p.position)
                         for p in patches if p.available)
            reachable = [distance for distance in distances if distance is not None]
            return min(reachable, default=VISION_RADIUS + 1)
        forage_distance = target_distance(forage)
        water_distance = target_distance(water)
    return Observation(
        tick=world.tick, animal_id=animal.animal_id, position=animal.position,
        visible_forage=forage, visible_water=water,
        nearest_forage_distance=float(forage_distance),
        nearest_water_distance=float(water_distance),
        local_herd_count=len(others), local_herd_positions=others,
        local_blocked_fraction=blocked / denom,
    )


def encode_features(observation: Observation, memory: MemoryFeatures, animal: AnimalState,
                    width: int, height: int) -> tuple[float, ...]:
    """Encode the versioned 20-float policy input in FEATURE_NAMES order."""
    phase = (observation.tick % 24) / 24.0 * 2.0 * math.pi
    visible_forage = bool(observation.visible_forage)
    visible_water = bool(observation.visible_water)
    forage_available = any(p.available for p in observation.visible_forage)
    water_available = any(p.available for p in observation.visible_water)
    radius = max(width, height, 1)
    feats = (
        animal.physiology.hunger, animal.physiology.thirst,
        animal.physiology.fatigue, animal.physiology.social_need,
        float(visible_forage), float(visible_water),
        min(1.0, observation.nearest_forage_distance / radius),
        min(1.0, observation.nearest_water_distance / radius),
        float(forage_available), float(water_available),
        observation.local_blocked_fraction, min(1.0, observation.local_herd_count / 4.0),
        memory.recent_water_success, memory.recent_forage_success,
        memory.water_access_blocked, memory.forage_depleted, memory.event_recency,
        math.sin(phase), math.cos(phase), float(observation.local_herd_count > 0),
    )
    if len(feats) != len(FEATURE_NAMES):
        raise AssertionError("feature schema implementation mismatch")
    return tuple(float(x) for x in feats)
