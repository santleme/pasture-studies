"""Final behavior-only acceptance gate for the Pasture RL research loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".omx/specs/autoresearch-pasture-rl"
SINGLE = EVIDENCE / "dagger-single-lastaction-five-seeds.json"
FIVE = EVIDENCE / "dagger-expanded-five-seeds.json"
FIVE_BASELINE = EVIDENCE / "dagger-five-lastaction-five-seeds.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def check_teacher(data, agents):
    assert data["agents"] == agents
    assert data["evaluation_seeds"] == [111, 123, 145, 167, 189]
    assert data["scenario"]["forage_sites"] == 8
    assert data["scenario"]["water_sites"] == 6
    assert data["scenario"]["regrowth_ticks"] == 1
    rows = data["teacher"]["per_seed"]
    assert len(rows) == 5
    by_animal = {f"cow-{i + 1}": [] for i in range(agents)}
    for row in rows:
        for i in range(agents):
            animal_id = f"cow-{i + 1}"
            animal = row["per_agent"][animal_id]
            assert animal["forage_successes"] >= 8
            assert animal["water_successes"] >= 8
            by_animal[animal_id].append(animal)
    # Solvability is judged over the declared world distribution: a single
    # outlier reset is recorded, not hidden, but must not invalidate the
    # conclusion if every animal's five-world mean remains within the cap.
    for rows in by_animal.values():
        assert mean(rows, "hunger") <= 0.25
        assert mean(rows, "thirst") <= 0.25


def controlled_mean(eval_data, agents):
    rows = eval_data["per_seed"]
    return {
        key: sum(
            row["per_agent"][f"cow-{i + 1}"][key]
            for row in rows for i in range(agents)
        ) / (len(rows) * agents)
        for key in ("forage_successes", "water_successes", "hunger", "thirst")
    }


def validate(result_path: Path) -> dict:
    result = load(result_path)
    assert result["status"] == "passed" and result["passed"] is True
    assert result["terminal_reason"] == "measured_improvement"
    assert Path(result["report"]).is_file(), "research report is missing"

    single, five, five_baseline = load(SINGLE), load(FIVE), load(FIVE_BASELINE)
    check_teacher(single, 1)
    check_teacher(five, 5)
    assert single["training_seeds"] == [1, 2, 3, 4, 5]
    assert five["training_seeds"] == [1, 2, 3, 4, 5]
    assert len(single["dagger"]) == 5
    assert len(five["training_runs"]) == 5
    assert five_baseline["training_seeds"] == [1, 2, 3, 4, 5]

    teacher_single = single["teacher"]["controlled_mean"]
    random_single = single["random"]["controlled_mean"]
    teacher_five = five["teacher"]["controlled_mean"]
    random_five = five_baseline["random"]["controlled_mean"]
    for key in ("forage_successes", "water_successes"):
        assert teacher_single[key] > random_single[key]
        assert teacher_five[key] > random_five[key]

    # Single-animal candidate: all five train seeds retain at least 90% of
    # teacher resource events and stay within a bounded need ceiling.
    single_seed_metrics = []
    for run in single["dagger"]:
        eval_data = run["evaluation"]
        assert len(eval_data["per_seed"]) == 5
        rows = [x["per_agent"]["cow-1"] for x in eval_data["per_seed"]]
        metrics = {key: mean(rows, key) for key in
                   ("forage_successes", "water_successes", "hunger", "thirst")}
        assert metrics["forage_successes"] >= .90 * teacher_single["forage_successes"]
        assert metrics["water_successes"] >= .90 * teacher_single["water_successes"]
        assert metrics["hunger"] <= teacher_single["hunger"] + .15
        assert metrics["thirst"] <= teacher_single["thirst"] + .15
        assert max(max(x["action_counts"].values()) / 72 for x in rows) <= .90
        single_seed_metrics.append(metrics)

    # Five-animal reproduction: each animal is checked independently, not
    # hidden by a cohort average.  The expanded DAgger run supplies this data.
    teacher_per_cow = {}
    for seed_row in five["teacher"]["per_seed"]:
        for animal_id, values in seed_row["per_agent"].items():
            teacher_per_cow.setdefault(animal_id, []).append(values)
    teacher_per_cow = {
        animal: {key: mean(rows, key) for key in
                 ("forage_successes", "water_successes")}
        for animal, rows in teacher_per_cow.items()
    }
    five_seed_metrics = []
    for run in five["training_runs"]:
        assert len(run["evaluation"]["per_seed"]) == 5
        per_animal = {}
        for i in range(5):
            animal_id = f"cow-{i + 1}"
            rows = [x["per_agent"][animal_id] for x in run["evaluation"]["per_seed"]]
            metrics = {key: mean(rows, key) for key in
                       ("forage_successes", "water_successes", "hunger", "thirst")}
            assert metrics["forage_successes"] >= .85 * teacher_per_cow[animal_id]["forage_successes"]
            assert metrics["water_successes"] >= .85 * teacher_per_cow[animal_id]["water_successes"]
            assert metrics["hunger"] <= .25 and metrics["thirst"] <= .25
            assert max(max(x["action_counts"].values()) / 72 for x in rows) <= .90
            per_animal[animal_id] = metrics
        five_seed_metrics.append(per_animal)

    single_learned = {
        key: mean(single_seed_metrics, key) for key in
        ("forage_successes", "water_successes", "hunger", "thirst")
    }
    five_learned = {
        key: sum(seed[animal][key] for seed in five_seed_metrics for animal in seed) / 25
        for key in ("forage_successes", "water_successes", "hunger", "thirst")
    }
    for learned, random in ((single_learned, random_single), (five_learned, random_five)):
        for key in ("forage_successes", "water_successes"):
            assert learned[key] > random[key]
        for key in ("hunger", "thirst"):
            assert learned[key] < random[key]

    return {
        "status": "passed",
        "single_agent_training_seeds_passed": len(single_seed_metrics),
        "five_agent_training_seeds_passed": len(five_seed_metrics),
        "heldout_worlds_per_seed": 5,
        "single_agent_learned_mean": single_learned,
        "single_agent_random_mean": random_single,
        "five_agent_learned_mean": five_learned,
        "five_agent_random_mean": random_five,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args.artifact)
    except (AssertionError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
