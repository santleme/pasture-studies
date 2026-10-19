"""Deterministic one-cell grid navigation, independent of animal physiology.

``NavigationResult.subreason`` is the exclusive, reason-coded outcome.  The
older ``reason`` field is retained as an aggregate for existing clients:
``no_available_target``, ``all_target_cells_occupied``,
``topologically_unreachable``, and ``dynamically_blocked`` map to the legacy
``no_reachable_target`` bucket; all other subreasons map to themselves.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .model import Action, AnimalState, FarmWorld, Position


@dataclass(frozen=True)
class CandidateDiagnostic:
    """Immutable audit record for one eligible target candidate."""

    target: Position
    target_class: str
    occupied: bool
    geometrically_reachable: bool
    dynamically_reachable: bool


@dataclass(frozen=True)
class NavigationResult:
    position: Position
    moved: bool
    target: Position | None
    reason: str
    subreason: str = "no_available_target"
    candidates: tuple[CandidateDiagnostic, ...] = ()


_DIRECTIONS = ((0, -1), (-1, 0), (1, 0), (0, 1))
_LEGACY_NO_TARGET_BUCKET = frozenset({
    "no_available_target", "all_target_cells_occupied",
    "topologically_unreachable", "dynamically_blocked",
})


def _reachable(world: FarmWorld, start: Position, target: Position,
               occupied: set[Position]) -> bool:
    """Independent connectivity query with deterministic 4-neighbour BFS."""
    if start == target:
        return True
    if not _in_bounds(world, target) or target in world.obstacles:
        return False
    queue = deque([start])
    seen = {start}
    while queue:
        here = queue.popleft()
        for dx, dy in _DIRECTIONS:
            nxt = Position(here.x + dx, here.y + dy)
            if not _in_bounds(world, nxt) or nxt in world.obstacles or nxt in occupied or nxt in seen:
                continue
            if nxt == target:
                return True
            seen.add(nxt)
            queue.append(nxt)
    return False


def _in_bounds(world: FarmWorld, p: Position) -> bool:
    return 0 <= p.x < world.width and 0 <= p.y < world.height


def static_path_distance(world: FarmWorld, start: Position, target: Position) -> int | None:
    """Shortest 4-neighbour path length ignoring temporary animal occupancy.

    This is a perception helper only. Actual movement still uses ``navigate``
    with current occupancy and its reason-coded dynamic diagnostics.
    """
    if not _in_bounds(world, start) or not _in_bounds(world, target):
        return None
    if start in world.obstacles or target in world.obstacles:
        return None
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        here, distance = queue.popleft()
        if here == target:
            return distance
        for dx, dy in _DIRECTIONS:
            nxt = Position(here.x + dx, here.y + dy)
            if not _in_bounds(world, nxt) or nxt in world.obstacles or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def _path_step(world: FarmWorld, start: Position, target: Position,
               occupied: set[Position]) -> Position | None:
    """Return first legal step on a shortest path, with stable tie-breaking."""
    if start == target:
        return None
    queue = deque([start])
    first: dict[Position, Position | None] = {start: None}
    while queue:
        here = queue.popleft()
        if here == target:
            break
        for dx, dy in _DIRECTIONS:
            nxt = Position(here.x + dx, here.y + dy)
            if not _in_bounds(world, nxt):
                continue
            if nxt in world.obstacles or (nxt in occupied and nxt != target) or nxt in first:
                continue
            first[nxt] = here
            queue.append(nxt)
    if target not in first:
        return None
    cursor = target
    while first[cursor] != start:
        parent = first[cursor]
        if parent is None:
            return None
        cursor = parent
    return cursor


def _ordered_unique(targets: Iterable[tuple[Position, str]], start: Position) -> list[tuple[Position, str]]:
    # A cell may host both water and forage in a fixture. Keep the first class
    # (forage precedes water for WALK) and retain deterministic spatial order.
    by_position: dict[Position, str] = {}
    for target, kind in targets:
        by_position.setdefault(target, kind)
    return sorted(by_position.items(), key=lambda item: (
        abs(item[0].x - start.x) + abs(item[0].y - start.y), item[0].x, item[0].y,
    ))


def _herd_targets(world: FarmWorld, animal: AnimalState,
                  occupied: set[Position]) -> tuple[list[tuple[Position, str]], bool]:
    """Legal cells from which an animal is in one-cell Manhattan interaction range.

    An animal already at distance <= 1 from a herd mate is considered in range;
    the call returns ``at_target`` with the mate's cell as the descriptive target
    and does not try to move onto an animal-occupied cell.
    """
    others = [a for a in world.animals if a.animal_id != animal.animal_id]
    if any(abs(a.position.x - animal.position.x) + abs(a.position.y - animal.position.y) <= 1
           for a in others):
        return [], True
    cells: set[Position] = set()
    for other in others:
        for dx, dy in _DIRECTIONS:
            cell = Position(other.position.x + dx, other.position.y + dy)
            # Keep occupied cells in diagnostics so an otherwise exhausted
            # interaction target set can be classified as all-occupied.
            if _in_bounds(world, cell):
                cells.add(cell)
    return _ordered_unique(((cell, "herd_interaction") for cell in cells), animal.position), False


def _result(animal: AnimalState, target: Position | None, subreason: str,
            candidates: tuple[CandidateDiagnostic, ...] = (),
            position: Position | None = None) -> NavigationResult:
    next_position = position or animal.position
    reason = "no_reachable_target" if subreason in _LEGACY_NO_TARGET_BUCKET else subreason
    return NavigationResult(next_position, next_position != animal.position, target, reason, subreason, candidates)


def navigate(world: FarmWorld, animal: AnimalState, action: Action,
             occupied: set[Position] | None = None) -> NavigationResult:
    """Resolve action intent to at most one legal grid step; never changes state.

    WALK prioritizes forage but falls back to water if no forage candidate has
    a legal route. HERD_INTERACT approaches an unoccupied adjacent cell.
    """
    occupied = set(occupied or ())
    occupied.discard(animal.position)
    if action not in {Action.WALK, Action.GRAZE, Action.DRINK, Action.HERD_INTERACT}:
        return _result(animal, None, "stationary_action")

    in_herd_range = False
    if action == Action.WALK:
        classes = [
            [(p.position, "forage") for p in world.forage if p.available],
            [(p.position, "water") for p in world.water if p.available],
        ]
    elif action == Action.GRAZE:
        classes = [[(p.position, "forage") for p in world.forage if p.available]]
    elif action == Action.DRINK:
        classes = [[(p.position, "water") for p in world.water if p.available]]
    else:
        targets, in_herd_range = _herd_targets(world, animal, occupied)
        if in_herd_range:
            other = min((a for a in world.animals if a.animal_id != animal.animal_id),
                        key=lambda a: (abs(a.position.x-animal.position.x) + abs(a.position.y-animal.position.y),
                                       a.position.x, a.position.y, a.animal_id))
            return _result(animal, other.position, "at_target")
        classes = [targets]

    all_targets = _ordered_unique((item for group in classes for item in group), animal.position)
    if not all_targets:
        return _result(animal, None, "no_available_target")

    diagnostics: list[CandidateDiagnostic] = []
    # Approved precedence: first priority candidate already at the animal wins.
    if all_targets[0][0] == animal.position:
        return _result(animal, animal.position, "at_target", (
            CandidateDiagnostic(animal.position, all_targets[0][1], False, True, True),
        ))

    for group in classes:
        ordered_group = _ordered_unique(group, animal.position)
        for target, target_class in ordered_group:
            is_occupied = target in occupied
            geometric = _reachable(world, animal.position, target, set())
            dynamic = False if is_occupied else _reachable(world, animal.position, target, occupied)
            diagnostics.append(CandidateDiagnostic(target, target_class, is_occupied, geometric, dynamic))
            if is_occupied:
                continue
            step = _path_step(world, animal.position, target, occupied)
            if step is not None:
                return _result(animal, target, "path_step", tuple(diagnostics), step)
            if target == animal.position:
                return _result(animal, target, "at_target", tuple(diagnostics))

    # Classify the full intent candidate set using the accepted exclusive order.
    if all(target in occupied for target, _ in all_targets):
        subreason = "all_target_cells_occupied"
    elif not any(_reachable(world, animal.position, target, set()) for target, _ in all_targets):
        subreason = "topologically_unreachable"
    else:
        subreason = "dynamically_blocked"
    return _result(animal, None, subreason, tuple(diagnostics))
