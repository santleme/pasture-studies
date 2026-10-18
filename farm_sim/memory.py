"""Bounded per-animal event memory and transparent toy retrieval features."""
from __future__ import annotations

from .model import AnimalMemory, MemoryEvent, MemoryFeatures

MEMORY_CAPACITY = 32
RETRIEVAL_WINDOW = 12


def remember(memory: AnimalMemory, event: MemoryEvent) -> AnimalMemory:
    """Append an event and retain only the most recent bounded history."""
    return AnimalMemory((memory.events + (event,))[-MEMORY_CAPACITY:])


def retrieve(memory: AnimalMemory, tick: int) -> MemoryFeatures:
    """Retrieve recent task-relevant markers; recency is explicit and deterministic."""
    events = tuple(e for e in memory.events if 0 <= tick - e.tick <= RETRIEVAL_WINDOW)
    water = next((e for e in reversed(events) if e.kind == "water_access"), None)
    forage = next((e for e in reversed(events) if e.kind == "forage_patch"), None)
    latest_age = min((tick - e.tick for e in events), default=RETRIEVAL_WINDOW + 1)
    return MemoryFeatures(
        recent_water_success=float(water is not None and water.success),
        recent_forage_success=float(forage is not None and forage.success),
        water_access_blocked=float(water is not None and not water.success),
        forage_depleted=float(forage is not None and not forage.success),
        event_recency=max(0.0, 1.0 - latest_age / RETRIEVAL_WINDOW),
        retrieved_events=events[-4:],
    )
