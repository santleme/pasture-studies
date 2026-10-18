"""Typed, versioned state contracts for the cattle-farm toy simulation.

All quantities and rewards here are synthetic software-task variables, not validated
cattle physiology. State records are immutable; transitions return replacements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence


class Action(str, Enum):
    GRAZE = "graze"
    DRINK = "drink"
    WALK = "walk"
    REST = "rest"
    RUMINATE = "ruminate"
    HERD_INTERACT = "herd_interact"


@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class ResourcePatch:
    position: Position
    available: bool = True


@dataclass(frozen=True)
class ToyPhysiology:
    """Normalized toy drives in [0, 1]; not biological units or clinical metrics."""
    hunger: float = 0.5
    thirst: float = 0.5
    fatigue: float = 0.2
    social_need: float = 0.3


@dataclass(frozen=True)
class MemoryEvent:
    tick: int
    kind: str
    success: bool
    position: Position


@dataclass(frozen=True)
class AnimalMemory:
    events: tuple[MemoryEvent, ...] = ()


@dataclass(frozen=True)
class AnimalState:
    animal_id: str
    position: Position
    physiology: ToyPhysiology = field(default_factory=ToyPhysiology)
    memory: AnimalMemory = field(default_factory=AnimalMemory)
    action: Action = Action.REST
    last_reward: float = 0.0
    last_probabilities: tuple[float, ...] = ()
    last_features: tuple[float, ...] = ()


@dataclass(frozen=True)
class FarmWorld:
    width: int
    height: int
    animals: tuple[AnimalState, ...]
    forage: tuple[ResourcePatch, ...]
    water: tuple[ResourcePatch, ...]
    obstacles: frozenset[Position] = frozenset()
    tick: int = 0
    seed: int = 0


@dataclass(frozen=True)
class Observation:
    tick: int
    animal_id: str
    position: Position
    visible_forage: tuple[ResourcePatch, ...]
    visible_water: tuple[ResourcePatch, ...]
    nearest_forage_distance: float
    nearest_water_distance: float
    local_herd_count: int
    local_herd_positions: tuple[Position, ...]
    local_blocked_fraction: float


@dataclass(frozen=True)
class MemoryFeatures:
    recent_water_success: float = 0.0
    recent_forage_success: float = 0.0
    water_access_blocked: float = 0.0
    forage_depleted: float = 0.0
    event_recency: float = 0.0
    retrieved_events: tuple[MemoryEvent, ...] = ()


FEATURE_SCHEMA_VERSION = "cattle-observation-v1"
FEATURE_NAMES: tuple[str, ...] = (
    "hunger", "thirst", "fatigue", "social_need",
    "forage_visible", "water_visible", "forage_distance_norm", "water_distance_norm",
    "forage_available", "water_available", "local_blocked_fraction", "herd_count_norm",
    "mem_water_success", "mem_forage_success", "mem_water_blocked", "mem_forage_depleted",
    "mem_event_recency", "tick_phase_sin", "tick_phase_cos", "self_in_herd",
)


@dataclass(frozen=True)
class PolicyDecision:
    """A side-effect-free policy result aligned to Action enum order."""
    action: Action
    probabilities: tuple[float, ...]
    policy_id: str = "injected-policy"


class ActionPolicy(Protocol):
    def decide(self, features: Sequence[float]) -> PolicyDecision:
        """Return an action distribution without mutating simulation state."""


@dataclass(frozen=True)
class SimEvent:
    tick: int
    animal_id: str
    kind: str
    detail: str
    reward: float = 0.0


@dataclass(frozen=True)
class TickResult:
    world: FarmWorld
    events: tuple[SimEvent, ...]


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
