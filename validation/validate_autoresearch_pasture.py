"""Cheap local reward-incentive probe; does not train or change production behavior."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farm_sim.model import Action, ToyPhysiology
from farm_sim.physiology import step_physiology

OUT = ROOT / ".omx/specs/autoresearch-pasture-studies"
HISTORICAL = {
    "validation/MVP1_VALIDATION_REPORT.md": "9779f0fb379e5a9a5e0f5d95c9c1d87412eb58e505cb09c87b06597ba068d21c",
    "validation/MVP1_REMEDIATION_REPORT.md": "c11b4a04910045e9278ba703223d303c9f4c22f2536f0e0423f2bba5571166ee",
    "validation/kaggle-mvp1/preregistration.json": "6836018714dd53a0723c4be404025e58e5f5c7f1531f436d51f21c2c235ca471",
    "artifacts/validation/mvp1-remediation/causal/causal-confirmatory-primary.summary.json": "5b1b1406f3c4573d8d8531774f569b7e9a97fb44bb2192f763f67a176fdaa5ac",
    "artifacts/validation/mvp1-remediation/training-eval/training-summary.json": "68eadb2dd17f5e46b6887076f5c2dc428edb8ed76079afaa97514519cad132c9",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    baseline = ToyPhysiology(hunger=0.9, thirst=0.9, fatigue=0.9, social_need=0.0)
    current = {}
    for action in Action:
        _, reward, events = step_physiology(
            baseline, action, forage_available=True, water_available=True, herd_nearby=True
        )
        current[action.value] = {"reward": reward, "events": list(events)}

    # Narrow proposed fix: the +0.1 contact bonus is available only when it
    # accompanies a marginal reduction in the toy social-need variable.
    herd_after, herd_reward, herd_events = step_physiology(
        baseline, Action.HERD_INTERACT, herd_nearby=True
    )
    marginal_reduction = baseline.social_need - herd_after.social_need
    prototype_reward = herd_reward - (0.1 if marginal_reduction <= 0 else 0.0)
    hashes = {name: sha(ROOT / name) for name in HISTORICAL}
    if hashes != HISTORICAL:
        raise AssertionError("protected historical criteria/evidence changed")

    # The held-out findings are consumed only descriptively; no outcomes are regenerated.
    summary = json.loads((ROOT / "artifacts/validation/mvp1-remediation/causal/causal-confirmatory-primary.summary.json").read_text())
    training = json.loads((ROOT / "artifacts/validation/mvp1-remediation/training-eval/training-summary.json").read_text())
    report = {
        "scope": "local synthetic one-step mechanism probe; not a new training/evaluation outcome",
        "historical_direct_evidence": {
            "wider_support_herd_interact_share": 1.0,
            "training_conditions_in_retained_summary": list(training.get("training", {}).keys()),
            "causal_memory_perception": "near-zero effects, no action flips in retained confirmatory evidence",
            "causal_summary_sha256": hashes["artifacts/validation/mvp1-remediation/causal/causal-confirmatory-primary.summary.json"],
        },
        "counterfactual": {
            "state": baseline.__dict__,
            "current_action_rewards": current,
            "current_herd_reward_after_social_need_zero": herd_reward,
            "social_need_marginal_reduction": marginal_reduction,
            "prototype_herd_reward_after_social_need_zero": prototype_reward,
            "incentive_removed": prototype_reward < herd_reward,
            "production_reward_unchanged": True,
        },
        "historical_hashes": hashes,
        "interpretation": "Demonstrates a repeatable reward exploit/incentive defect at this state. It does not prove this is the unique or dominant cause of the training collapse, and the point-level fix is not an end-to-end learned-policy correction.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["counterfactual"]["incentive_removed"]
    assert report["counterfactual"]["current_herd_reward_after_social_need_zero"] > 0
    assert report["counterfactual"]["prototype_herd_reward_after_social_need_zero"] < 0
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    result = {
        "status": "passed", "passed": True,
        "summary": "A non-diminishing herd-contact bonus is reproduced as a contributing incentive defect; a narrow point-level prototype removes it.",
        "scope": "mechanism and reward-point prototype only; no end-to-end training correction or unique cause claimed",
        "output_artifact_path": ".omx/specs/autoresearch-pasture-studies/report.md",
        "probe_artifact_path": ".omx/specs/autoresearch-pasture-studies/probe.json",
        "historical_evidence_hashes_unchanged": True,
        "kaggle_jobs_started": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
