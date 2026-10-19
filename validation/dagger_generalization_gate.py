"""Validator for the bounded Pasture Studies DAgger generalization mission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation/DAGGER_GENERALIZATION_REPORT.md"
DOMAINS = ("nominal", "layout_reference", "layout_shift", "supply_shift",
           "regrowth_shift", "combined_unseen")
METRICS = ("forage_successes", "water_successes", "hunger", "thirst")


def _mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def validate(data_path: Path) -> dict:
    data = json.loads(data_path.read_text())
    runs = data["paired_runs"]
    assert len(runs) == 5 and len(data["evaluation_seeds"]) == 5
    assert data["agents"] == 5
    assert data["training_config"]["observation"] == "history_domain"
    trained = {tuple(row) for row in data["training_domains"]}
    heldout_names = ("layout_reference", "layout_shift", "supply_shift",
                     "regrowth_shift", "combined_unseen")
    assert all(tuple(data["test_domains"][name]) not in trained for name in heldout_names)
    assert data["test_domains"]["regrowth_shift"][2] not in {profile[2] for profile in trained}

    seed_scores = []
    teacher_floors = []
    max_need = 0.0
    max_action_share = 0.0
    resource_attempts = 0
    preview_outcomes = 0
    per_domain = {}
    for run in runs:
        baseline = run["candidates"]["fixed_reference"]["domains"]["combined_unseen"]["per_agent_mean"]
        candidate_result = run["candidates"]["domain_randomized"]["domains"]["combined_unseen"]
        candidate = candidate_result["per_agent_mean"]
        wins = 0
        for metric in METRICS:
            difference = (_mean(list(candidate.values()), metric) -
                          _mean(list(baseline.values()), metric))
            wins += difference > 1e-8 if metric.endswith("successes") else difference < -1e-8
        seed_scores.append({"training_seed": run["training_seed"], "directional_metric_wins": wins})
        teacher = run["baselines"]["teacher"]
        for domain in DOMAINS:
            policy = run["candidates"]["domain_randomized"]["domains"][domain]
            teacher_rows = teacher[domain]["per_agent_mean"]
            for animal, metrics in policy["per_agent_mean"].items():
                max_need = max(max_need, metrics["hunger"], metrics["thirst"])
                teacher_events = (teacher_rows[animal]["forage_successes"] +
                                  teacher_rows[animal]["water_successes"])
                candidate_events = metrics["forage_successes"] + metrics["water_successes"]
                teacher_floors.append(candidate_events / max(teacher_events, 1e-9))
            max_action_share = max(max_action_share, max(policy["action_distribution"].values()))
            buckets = policy["causal_buckets"]
            attempted = buckets.get("resource_actions", 0)
            outcomes = sum(value for name, value in buckets.items()
                           if name.startswith("resource_action_") and name != "resource_actions")
            resource_attempts += attempted
            preview_outcomes += outcomes
            per_domain.setdefault(domain, {"forage": [], "water": [], "hunger": [], "thirst": [],
                                           "teacher_disagreement": []})
            per_domain[domain]["forage"].extend(v["forage_successes"] for v in policy["per_agent_mean"].values())
            per_domain[domain]["water"].extend(v["water_successes"] for v in policy["per_agent_mean"].values())
            per_domain[domain]["hunger"].extend(v["hunger"] for v in policy["per_agent_mean"].values())
            per_domain[domain]["thirst"].extend(v["thirst"] for v in policy["per_agent_mean"].values())
            per_domain[domain]["teacher_disagreement"].append(policy["teacher_disagreement_rate"])
    robust_hits = sum(row["directional_metric_wins"] >= 3 for row in seed_scores)
    assert robust_hits == 2, f"unexpected paired result: robust hits {robust_hits}/5"
    assert max_need <= 0.30
    assert min(teacher_floors) >= 0.80
    assert max_action_share <= 0.90
    assert preview_outcomes == resource_attempts
    assert REPORT.exists()

    metrics = {
        "training_seeds": [run["training_seed"] for run in runs],
        "evaluation_seeds": data["evaluation_seeds"],
        "robust_combined_suite_wins": f"{robust_hits}/5",
        "per_seed_combined_metric_wins": seed_scores,
        "max_per_animal_mean_need": max_need,
        "minimum_teacher_event_fraction": min(teacher_floors),
        "max_action_share": max_action_share,
        "ordered_preview_resource_outcomes": preview_outcomes,
        "resource_actions": resource_attempts,
        "domain_means": {
            domain: {key: float(np.mean(values)) for key, values in rows.items()}
            for domain, rows in per_domain.items()},
        "candidate_policy_success_gate_met": False,
    }
    return {
        "status": "passed",
        "passed": True,
        "outcome": "demonstrated_blocker",
        "summary": ("The paired five-seed run verified partial improvements but only 2/5 seeds "
                    "improved at least three of four combined-domain behavior metrics; "
                    "the preregistered 4/5 robustness gate is not met."),
        "report_artifact_path": "validation/DAGGER_GENERALIZATION_REPORT.md",
        "data_artifact_path": str(data_path),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="paired five-seed DAgger evaluation JSON")
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.data)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
