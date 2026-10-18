"""Independent fixtures for navigation outcomes and legality contracts."""
from collections import deque
import unittest

from farm_sim.model import Action, AnimalState, FarmWorld, Position, ResourcePatch
from farm_sim.navigation import navigate


def animal(name="cow-1", pos=Position(0, 0)):
    return AnimalState(name, pos)


def world(width=5, height=5, *, cow=None, cows=(), forage=(), water=(), obstacles=()):
    herd = (cow or animal(),) + tuple(cows)
    return FarmWorld(width, height, herd, tuple(ResourcePatch(p, bool(rest[0]) if rest else True)
                                                   for p, *rest in forage),
                     tuple(ResourcePatch(p, bool(rest[0]) if rest else True)
                                                   for p, *rest in water),
                     frozenset(obstacles))


def occupied_by_others(w, cow):
    return {a.position for a in w.animals if a.animal_id != cow.animal_id}


def oracle_reachable(width, height, obstacles, start, target):
    """Test-only BFS, deliberately independent of navigation implementation."""
    if start == target:
        return True
    pending, seen = deque([start]), {start}
    while pending:
        p = pending.popleft()
        for q in (Position(p.x, p.y - 1), Position(p.x - 1, p.y),
                  Position(p.x + 1, p.y), Position(p.x, p.y + 1)):
            if not (0 <= q.x < width and 0 <= q.y < height) or q in obstacles or q in seen:
                continue
            if q == target:
                return True
            seen.add(q)
            pending.append(q)
    return False


class NavigationTests(unittest.TestCase):
    def test_open_grid_moves_one_legal_step_and_uses_stable_tie_break(self):
        w = world(forage=[(Position(4, 2), True)])
        cow = w.animals[0]
        result = navigate(w, cow, Action.WALK)
        self.assertEqual(result.position, Position(1, 0))  # left, right, down, up order after out-of-bounds
        self.assertEqual(result.target, Position(4, 2))
        self.assertEqual(result.reason, "path_step")
        self.assertEqual(result.subreason, "path_step")
        self.assertFalse(result.position in w.obstacles)

    def test_wall_without_gap_is_geometric_unreachable_but_gap_is_reachable(self):
        wall = {Position(2, y) for y in range(5)}
        cow = animal(pos=Position(0, 2))
        target = Position(4, 2)
        blocked = world(cow=cow, forage=[(target, True)], obstacles=wall)
        result = navigate(blocked, cow, Action.GRAZE)
        self.assertEqual(result.subreason, "topologically_unreachable")
        self.assertFalse(oracle_reachable(5, 5, wall, cow.position, target))
        gap = wall - {Position(2, 4)}
        opened = world(cow=cow, forage=[(target, True)], obstacles=gap)
        moved = navigate(opened, cow, Action.GRAZE)
        self.assertEqual(moved.subreason, "path_step")
        self.assertTrue(oracle_reachable(5, 5, gap, cow.position, target))

    def test_walk_falls_back_to_reachable_water_when_forage_is_disconnected(self):
        wall = {Position(2, y) for y in range(5)}
        cow = animal(pos=Position(0, 2))
        w = world(cow=cow, forage=[(Position(4, 2), True)], water=[(Position(1, 2), True)], obstacles=wall)
        result = navigate(w, cow, Action.WALK)
        self.assertEqual(result.target, Position(1, 2))
        self.assertEqual(result.subreason, "path_step")
        self.assertEqual(result.candidates[0].target_class, "forage")
        self.assertEqual(result.candidates[-1].target_class, "water")

    def test_no_available_targets_and_depleted_targets_are_not_unreachable(self):
        cow = animal()
        empty = world(cow=cow)
        depleted = world(cow=cow, forage=[(Position(4, 4), False)])
        for w in (empty, depleted):
            result = navigate(w, cow, Action.GRAZE)
            self.assertEqual(result.subreason, "no_available_target")
            self.assertEqual(result.reason, "no_reachable_target")
            self.assertEqual(result.candidates, ())

    def test_current_cell_target_is_at_target(self):
        cow = animal(pos=Position(2, 2))
        w = world(cow=cow, forage=[(cow.position, True)])
        result = navigate(w, cow, Action.GRAZE, occupied_by_others(w, cow))
        self.assertEqual((result.position, result.target, result.subreason),
                         (cow.position, cow.position, "at_target"))

    def test_all_candidate_cells_occupied_is_distinct(self):
        cow = animal(pos=Position(0, 0))
        target = Position(2, 0)
        other = animal("cow-2", target)
        w = world(cow=cow, cows=(other,), forage=[(target, True)])
        result = navigate(w, cow, Action.GRAZE, occupied_by_others(w, cow))
        self.assertEqual(result.subreason, "all_target_cells_occupied")
        self.assertTrue(result.candidates[0].occupied)
        self.assertTrue(result.candidates[0].geometrically_reachable)

    def test_dynamic_cow_blocking_is_distinct_from_static_topology(self):
        cow = animal(pos=Position(0, 1))
        target = Position(2, 1)
        blocker = animal("cow-2", Position(1, 1))
        w = world(cow=cow, cows=(blocker,), forage=[(target, True)],
                  obstacles=(Position(0, 0), Position(0, 2), Position(1, 0), Position(1, 2)))
        result = navigate(w, cow, Action.GRAZE, occupied_by_others(w, cow))
        self.assertEqual(result.subreason, "dynamically_blocked")
        self.assertTrue(result.candidates[0].geometrically_reachable)
        self.assertFalse(result.candidates[0].dynamically_reachable)

    def test_herd_interaction_approaches_legal_unoccupied_adjacent_cell(self):
        cow = animal(pos=Position(0, 2))
        other = animal("cow-2", Position(4, 2))
        w = world(cow=cow, cows=(other,))
        result = navigate(w, cow, Action.HERD_INTERACT, occupied_by_others(w, cow))
        self.assertEqual(result.subreason, "path_step")
        self.assertNotEqual(result.position, other.position)
        self.assertEqual(abs(result.position.x - cow.position.x) + abs(result.position.y - cow.position.y), 1)
        self.assertEqual(result.target, Position(3, 2))

    def test_herd_interaction_current_range_is_no_move_at_target(self):
        cow = animal(pos=Position(2, 2))
        other = animal("cow-2", Position(2, 3))
        w = world(cow=cow, cows=(other,))
        result = navigate(w, cow, Action.HERD_INTERACT, occupied_by_others(w, cow))
        self.assertEqual(result.subreason, "at_target")
        self.assertFalse(result.moved)
        self.assertEqual(result.target, other.position)

    def test_herd_interaction_targets_separated_by_true_cut_are_unreachable(self):
        cow = animal(pos=Position(0, 2))
        other = animal("cow-2", Position(4, 2))
        # Block every in-bounds approach cell adjacent to the distant herd mate.
        walls = {Position(3, 2), Position(4, 1), Position(4, 3)}
        w = world(cow=cow, cows=(other,), obstacles=walls)
        result = navigate(w, cow, Action.HERD_INTERACT, occupied_by_others(w, cow))
        self.assertEqual(result.subreason, "topologically_unreachable")
        self.assertTrue(result.candidates)
        self.assertTrue(all(not c.geometrically_reachable for c in result.candidates))

    def test_multiple_cows_resource_arbitration_never_enters_occupied_cell(self):
        first = animal("cow-1", Position(0, 0))
        second = animal("cow-2", Position(1, 0))
        resource = Position(2, 0)
        w = world(cow=first, cows=(second,), forage=[(resource, True)])
        result = navigate(w, first, Action.GRAZE, occupied={first.position, second.position})
        self.assertNotEqual(result.position, second.position)
        self.assertEqual(result.subreason, "path_step")

    def test_boundaries_stationary_and_obstacles(self):
        cow = animal(pos=Position(0, 0))
        w = world(cow=cow, forage=[(Position(1, 1), True)], obstacles=(Position(1, 0),))
        result = navigate(w, cow, Action.GRAZE)
        self.assertEqual(result.position, Position(0, 1))
        self.assertFalse(result.position in w.obstacles)
        self.assertEqual(navigate(w, cow, Action.REST).subreason, "stationary_action")
        boundary = world(cow=animal(pos=Position(4, 4)), water=[(Position(4, 3), True)])
        self.assertEqual(navigate(boundary, boundary.animals[0], Action.DRINK).position, Position(4, 3))

    def test_reference_oracle_matches_candidate_geometric_diagnostics(self):
        obstacles = {Position(2, y) for y in range(4)}
        cow, target = animal(pos=Position(0, 1)), Position(4, 1)
        w = world(cow=cow, forage=[(target, True)], obstacles=obstacles)
        result = navigate(w, cow, Action.GRAZE)
        for candidate in result.candidates:
            self.assertEqual(candidate.geometrically_reachable,
                             oracle_reachable(w.width, w.height, set(w.obstacles), cow.position, candidate.target))


if __name__ == "__main__":
    unittest.main()
