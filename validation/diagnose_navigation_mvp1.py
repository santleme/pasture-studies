#!/usr/bin/env python3
"""Reconstruct and reason-code MVP1 navigation calls from frozen evaluation inputs.

This deliberately does not alter product navigation. Per-call rows are a new,
deterministic replay of the saved evaluation configuration, not historic traces.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import farm_sim.simulation as simulation_module
from farm_sim.model import Action, Position
from farm_sim.navigation import navigate
from farm_sim.simulation import Simulation
from validation import evaluate_policies as evaluator

OUT = ROOT / ".omx/research/mvp1-remediation-baseline"
EVAL_JSON = ROOT / ".omx/research/mvp1-policy-evaluation.json"
PRECEDENCE = ["stationary_action", "no_available_target", "at_target",
              "all_target_cells_occupied", "topologically_unreachable",
              "dynamically_blocked", "path_step"]
NAV_ACTIONS = {Action.WALK, Action.GRAZE, Action.DRINK, Action.HERD_INTERACT}


def pos(p):
    return [p.x, p.y]


def manhattan(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


def oracle_path(world, start, target, occupied=frozenset()):
    """Independent BFS: first-step and full path; None means no legal route."""
    if start == target:
        return [start]
    queue = deque([start])
    parent = {start: None}
    while queue:
        here = queue.popleft()
        if here == target:
            break
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            nxt = Position(here.x + dx, here.y + dy)
            if not (0 <= nxt.x < world.width and 0 <= nxt.y < world.height):
                continue
            if nxt in world.obstacles or (nxt in occupied and nxt != target) or nxt in parent:
                continue
            parent[nxt] = here
            queue.append(nxt)
    if target not in parent:
        return None
    route = []
    cursor = target
    while cursor is not None:
        route.append(cursor)
        cursor = parent[cursor]
    return list(reversed(route))


def targets_for(world, animal, action):
    if action == Action.WALK:
        targets = [p.position for p in world.forage if p.available]
        if not targets:
            targets = [p.position for p in world.water if p.available]
        return "forage_then_water", targets
    if action == Action.GRAZE:
        return "forage", [p.position for p in world.forage if p.available]
    if action == Action.DRINK:
        return "water", [p.position for p in world.water if p.available]
    if action == Action.HERD_INTERACT:
        return "other_animals", [a.position for a in world.animals if a.animal_id != animal.animal_id]
    return "none", []


def classify(world, animal, action, occupied, original_result):
    """Apply the frozen plan's exclusive precedence; retain overlapping diagnostics."""
    target_class, raw = targets_for(world, animal, action)
    candidates = sorted(raw, key=lambda p: (manhattan(p, animal.position), p.x, p.y))
    candidate_rows = []
    for target in candidates:
        static = oracle_path(world, animal.position, target)
        dynamic = oracle_path(world, animal.position, target, occupied)
        candidate_rows.append({"position": pos(target), "occupied_by_other": target in occupied,
                               "obstacle_only_reachable": static is not None,
                               "occupancy_aware_reachable": dynamic is not None,
                               "obstacle_only_path": [pos(p) for p in static] if static else None,
                               "occupancy_aware_path": [pos(p) for p in dynamic] if dynamic else None})
    if action not in NAV_ACTIONS:
        reason = "stationary_action"
    elif not candidates:
        reason = "no_available_target"
    elif candidates[0] == animal.position:
        reason = "at_target"
    elif all(p in occupied for p in candidates):
        reason = "all_target_cells_occupied"
    elif not any(row["obstacle_only_reachable"] for row in candidate_rows):
        reason = "topologically_unreachable"
    elif not any(row["occupancy_aware_reachable"] for row in candidate_rows):
        reason = "dynamically_blocked"
    elif original_result.moved:
        reason = "path_step"
    else:
        # A route exists under the oracle but production returned no step. Keep this
        # visible as an implementation discrepancy, not an invented accepted class.
        reason = "oracle_navigation_disagreement"
    selected_path = None
    if original_result.target is not None:
        path = oracle_path(world, animal.position, original_result.target, occupied)
        selected_path = [pos(p) for p in path] if path else None
    return {"primary_reason": reason, "target_class": target_class,
            "candidate_positions_priority_order": [pos(p) for p in candidates],
            "candidate_diagnostics": candidate_rows,
            "occupied_positions": [pos(p) for p in sorted(occupied)],
            "original_result": {"reason": original_result.reason,
                                "moved": original_result.moved,
                                "target": pos(original_result.target) if original_result.target else None,
                                "next_position": pos(original_result.position)},
            "selected_target_oracle_path": selected_path}


def fixture_checks():
    class World:
        width = 4
        height = 4
        obstacles = frozenset({Position(1, 0), Position(1, 1), Position(1, 2)})
    w = World()
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    check("reachable_path_has_legal_adjacent_steps", oracle_path(w, Position(0, 0), Position(0, 3)) ==
          [Position(0, 0), Position(0, 1), Position(0, 2), Position(0, 3)])
    check("obstacle_wall_with_gap_connects", oracle_path(w, Position(0, 0), Position(2, 0)) is not None)
    w.obstacles = frozenset({Position(1, y) for y in range(4)})
    check("full_wall_is_topologically_unreachable", oracle_path(w, Position(0, 1), Position(2, 1)) is None)
    w.obstacles = frozenset()
    check("occupied_detour_respects_blocked_cell", oracle_path(w, Position(0, 1), Position(2, 1),
          {Position(1, 1)}) == [Position(0, 1), Position(0, 0), Position(1, 0), Position(2, 0), Position(2, 1)])
    check("same_cell_returns_zero_length_route", oracle_path(w, Position(1, 1), Position(1, 1)) == [Position(1, 1)])
    return checks


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def replay(policy_name, seed, scenario):
    sim = Simulation(evaluator.scenario_world(scenario, seed), evaluator.POLICIES[policy_name]())
    calls = []
    original = simulation_module.navigate
    def capture(world, animal, action, occupied=None):
        occupied = set(occupied or ())
        occupied.discard(animal.position)
        result = original(world, animal, action, occupied)
        details = classify(world, animal, action, occupied, result)
        calls.append({"tick": world.tick, "animal_id": animal.animal_id,
                      "action": action.value, "position_before": pos(animal.position), **details})
        return result
    simulation_module.navigate = capture
    try:
        for _ in range(evaluator.STEPS):
            sim.step()
    finally:
        simulation_module.navigate = original
    return calls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [EVAL_JSON, ROOT / "validation/evaluate_policies.py"]
    source_paths += sorted((ROOT / "farm_sim").glob("*.py"))
    source_paths += sorted((ROOT / "artifacts/policies").glob("*.npz"))
    source_paths += sorted((ROOT / "artifacts/policies").glob("*.json"))
    frozen = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
              for p in source_paths if p.is_file()]
    old = json.loads(EVAL_JSON.read_text())
    runs, summary = [], []
    for row in old["runs"]:
        calls = replay(row["policy"], row["world_seed"], row["scenario"])
        counts = Counter(c["primary_reason"] for c in calls)
        original_counts = row.get("navigation_reason_counts", {})
        mismatch = {k: {"reconstructed": counts.get(k, 0), "original": original_counts.get(k, 0)}
                    for k in sorted(set(counts) | set(original_counts))
                    if counts.get(k, 0) != original_counts.get(k, 0)}
        runs.append({"policy": row["policy"], "scenario": row["scenario"],
                     "world_seed": row["world_seed"], "ticks": row["ticks"],
                     "navigation_call_count": len(calls), "reconstructed_primary_reason_counts": dict(sorted(counts.items())),
                     "original_aggregate_reason_counts": original_counts,
                     "reason_count_mismatches": mismatch,
                     "reconstructed_calls": calls})
        summary.append({"policy": row["policy"], "scenario": row["scenario"],
                        "world_seed": row["world_seed"], "calls": len(calls),
                        "reconstructed_primary_reason_counts": dict(sorted(counts.items())),
                        "original_aggregate_reason_counts": original_counts,
                        "mismatch": mismatch})
    doc = {"title": "MVP1 navigation reconstructed replay diagnostics",
           "evidence_status": "RECONSTRUCTED deterministic replay; not historic per-call records",
           "source_artifact": str(EVAL_JSON.relative_to(ROOT)),
           "reproducible_command": ".venv/bin/python validation/diagnose_navigation_mvp1.py",
           "scope": {"episodes": len(old["runs"]), "ticks_per_episode": evaluator.STEPS,
                     "ticks_per_day_convention": evaluator.TICKS_PER_DAY,
                     "total_navigation_calls": sum(r["navigation_call_count"] for r in runs),
                     "primary_reason_precedence": PRECEDENCE,
                     "classification_note": "exclusive reconstructed classification; WALK uses original forage-else-water target semantics; oracle ignores cattle for static topology"},
           "frozen_inputs": frozen,
           "independent_bfs_oracle_fixture_checks": fixture_checks(),
           "run_comparisons": summary,
           "aggregate_reconstructed_reason_counts": dict(sorted(sum_counters(runs).items())),
           "aggregate_original_reason_counts": dict(sorted(sum_counters(runs, "original_aggregate_reason_counts").items())),
           "aggregate_mismatch_counts_by_reason": dict(sorted(sum_counters(runs, "reason_count_mismatches").items())),
           "runs": runs,
           "limitations": ["The saved evaluator JSON contains per-run aggregates, not historic call records or full runtime provenance.",
                           "Call rows below are generated by replaying the saved seeds, policies, scenarios, and 72 ticks against the currently hashed source/checkpoints.",
                           "The historical 5,009 outcomes cannot be asserted to have this exact per-call decomposition; aggregate mismatches are disclosed.",
                           "The classifier reports original navigation output alongside independent static and occupancy-aware BFS oracle paths."]}
    (OUT / "navigation-reconstructed-replay.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str((OUT / "navigation-reconstructed-replay.json").relative_to(ROOT)),
                      "calls": doc["scope"]["total_navigation_calls"],
                      "reconstructed": doc["aggregate_reconstructed_reason_counts"],
                      "original": doc["aggregate_original_reason_counts"],
                      "mismatch": doc["aggregate_mismatch_counts_by_reason"],
                      "oracle_fixture_checks": len(doc["independent_bfs_oracle_fixture_checks"])}, indent=2))


def sum_counters(runs, field="reconstructed_primary_reason_counts"):
    total = Counter()
    for run in runs:
        data = run.get(field, {})
        if field == "reason_count_mismatches":
            for reason, vals in data.items():
                total[reason] += vals["reconstructed"] - vals["original"]
        else:
            total.update(data)
    return total


if __name__ == "__main__":
    main()
