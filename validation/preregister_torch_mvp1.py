#!/usr/bin/env python3
"""Create a hash-bound, outcome-blind split/protocol manifest for Torch MVP1.

This command does not read model outcomes. It binds the approved plan and the
historical MVP1 report hash, freezes disjoint seed groups and copies acceptance
thresholds verbatim into a pre-outcome manifest. Pilot-derived delta0 remains
unset until a precision-only pilot is independently run and reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farm_sim.torch_training import derive_seed

APPROVED_HASHES_PATH = ROOT / "validation/kaggle-mvp1/approved-plan-hashes.json"
LOCAL_APPROVED_ARTIFACTS = {
    "plan": ROOT / ".omx/plans/pasture-kaggle-cognition-plan.md",
    "architect_review": ROOT / ".omx/plans/ralplan-review-pasture-kaggle-architect.md",
    "critic_review": ROOT / ".omx/plans/ralplan-review-pasture-kaggle-critic.md",
}
MVP1_REPORT_PATH = ROOT / "validation/MVP1_REMEDIATION_REPORT.md"
OUTPUT_PATH = ROOT / "validation/kaggle-mvp1/preregistration.json"
HISTORICAL_MANIFEST_SHA256 = "b489f0d2bb0bd8796f5e9e0013c8c4a5c8558bd5d926a1b82a72a9379f400d1b"

FAMILIES = ("starter", "resource_scarce_relocated", "wide", "unseen_corridor")
TRAIN_SEEDS = tuple(range(71001, 71011))
SPLIT_SEEDS = {
    "validation": {family: tuple(range(72001 + i * 100, 72031 + i * 100)) for i, family in enumerate(FAMILIES)},
    "precision_pilot": {family: tuple(range(73001 + i * 100, 73061 + i * 100)) for i, family in enumerate(FAMILIES)},
    "locked_test": {family: tuple(range(74001 + i * 100, 74031 + i * 100)) for i, family in enumerate(FAMILIES)},
    "precision_reserve": {family: tuple(range(78001 + i * 100, 78031 + i * 100)) for i, family in enumerate(FAMILIES)},
}

SOURCE_PATHS = (
    "farm_sim/torch_training.py", "farm_sim/torch_checkpoint.py", "farm_sim/simulation.py",
    "farm_sim/model.py", "farm_sim/navigation.py", "farm_sim/perception.py",
    "farm_sim/memory.py", "farm_sim/physiology.py", "farm_sim/policy.py",
    "validation/train_torch_mvp1.py", "validation/evaluate_torch_mvp1.py",
    "validation/benchmark_torch_fixed_corpus.py", "validation/preregister_torch_mvp1.py",
    "validation/profile_torch_training.py", "validation/profile_training_pipeline.py",
    "tests/test_torch_training.py", "tests/test_torch_checkpoint.py",
    "tests/test_torch_evaluation.py", "tests/test_torch_fixed_corpus.py",
    "tests/test_torch_preregistration.py", "tests/test_profile_training_pipeline.py",
    "tests/test_vectorized_world_batch.py",
    "validation/kaggle-mvp1/approved-plan-hashes.json",
    "requirements-training.txt", "notebooks/pasture_studies_kaggle.ipynb", "docs/kaggle-training.md",
)
TRAINING_SOURCE_PATHS = (
    "validation/train_torch_mvp1.py", "farm_sim/torch_training.py", "farm_sim/torch_checkpoint.py",
    "requirements-training.txt", "farm_sim/simulation.py", "farm_sim/model.py",
    "farm_sim/navigation.py", "farm_sim/perception.py", "farm_sim/memory.py",
    "farm_sim/physiology.py", "farm_sim/policy.py",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def approved_artifact_hashes(*, verify_local_files: bool = True) -> dict[str, str]:
    """Load tracked pins; optionally cross-check ignored approval files if present.

    Public checkouts do not contain `.omx`, so its plan/review files must never
    be required at runtime. The tracked attestation stores only the exact file
    hashes. In the authoring workspace, verify those pins against all three
    local files when the complete set is available.
    """
    doc = json.loads(APPROVED_HASHES_PATH.read_text())
    expected_keys = {"plan", "architect_review", "critic_review"}
    if (not isinstance(doc, dict) or set(doc) != {"schema_version", "approved_artifacts"}
            or doc.get("schema_version") != 1 or not isinstance(doc.get("approved_artifacts"), dict)
            or set(doc["approved_artifacts"]) != expected_keys):
        raise ValueError("approved plan-hash attestation has an unsupported schema")
    hashes = doc["approved_artifacts"]
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("approved plan-hash attestation contains an invalid SHA-256 pin")
    if verify_local_files and all(path.is_file() for path in LOCAL_APPROVED_ARTIFACTS.values()):
        actual = {name: sha(path) for name, path in LOCAL_APPROVED_ARTIFACTS.items()}
        if actual != hashes:
            raise ValueError("tracked approved plan-hash attestation does not match local approved artifacts")
    return dict(hashes)


def _walk_ints(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_ints(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_ints(item)


def historical_seed_ids() -> set[int]:
    """Historical IDs transcribed from frozen MVP1 report/protocol.

    The underlying `.omx` manifest is intentionally not shipped in public
    checkouts, so keep its recorded SHA-256 and explicit seed namespaces here.
    """
    seeds = set(range(11001, 11011)) | set(range(12001, 12031))
    seeds |= set(range(21001, 21031)) | set(range(31001, 31061))
    for first in (41001, 51001, 61001, 41101, 51101, 61101):
        seeds |= set(range(first, first + 30))
    # Historical MVP1 software smoke/stress evaluation seed IDs.
    seeds.update((11, 17, 23, 29, 47, 71, 83, 101, 211, 307))
    return seeds


def _make_seed_sets(episodes_per_train_seed: int = 278) -> dict[str, Any]:
    train_worlds = {str(seed): [derive_seed(seed, "train-world", i) for i in range(episodes_per_train_seed)]
                    for seed in TRAIN_SEEDS}
    return {"training_root_seeds": list(TRAIN_SEEDS),
            "training_world_seeds_by_root": train_worlds,
            **{key: {family: list(values) for family, values in families.items()}
               for key, families in SPLIT_SEEDS.items()},
            "derivation": {"training": "sha256(root, train-world, episode-index), matching validation/train_torch_mvp1.py",
                           "validation/pilot/locked/reserve": "explicit integer world-seed IDs frozen in this manifest"}}


def acceptance_spec() -> dict[str, Any]:
    """Exact numeric/falsifiable criteria copied from the approved plan."""
    return {
        "hardware_fixed_corpus": {"gpu_compute_speedup_ci95_lower_vs_best_cpu": 2.0,
                                  "fixed_work_elapsed_reduction_min_fraction": 0.30,
                                  "t4x2_speedup_ci95_lower_vs_t4x1": 1.20,
                                  "minimum_paired_full_repeats": 5},
        "end_to_end_pipeline": {"speedup_ci95_lower_vs_cpu": 2.0,
                                "elapsed_reduction_min_fraction": 0.30,
                                "t4x2_speedup_ci95_lower_vs_t4x1": 1.20,
                                "maximum_action_disagreement_fraction": 0.005,
                                "maximum_absolute_simulator_work_counter_difference_fraction": 0.01,
                                "minimum_paired_full_repeats": 5},
        "training_validity": {"training_seed_count_per_arm": 10,
                              "locked_worlds_per_family_minimum": 30,
                              "precision_reserve_worlds_per_family_maximum": 60,
                              "near_zero_untrained_mean_abs_threshold": 0.1,
                              "utility_margin_when_non_near_zero": "0.05 * abs(mean(untrained return))",
                              "non_harm_margin_when_non_near_zero": "0.10 * abs(mean(untrained return))",
                              "near_zero_margin_delta0_by_family": None,
                              "delta0_formula": "0.05 * SD(untrained returns) from separate precision-only pilot",
                              "holm_familywise_alpha": 0.05,
                              "delta0_status": "INCONCLUSIVE_PENDING_PRECISION_ONLY_PILOT"},
        "cognition": {"memory_effect": "P(DRINK | water_access success) - P(DRINK | blocked)",
                      "perception_effect": "P(DRINK | water_available=1) - P(DRINK | water_available=0)",
                      "minimum_each_primary_effect_absolute_probability": 0.05,
                      "minimum_ablation_attenuation_fraction": 0.50,
                      "minimum_attenuation_ci_lower": 0.0,
                      "holm_family_size": 4, "holm_alpha": 0.05,
                      "effect_ci_lower_strictly_greater_than": 0.05,
                      "requires_bit_identical_sham": True},
        "claim_gates": {"gpu_acceleration": "INCONCLUSIVE until required device runs/repeats/parity evidence are available",
                        "utility": "INCONCLUSIVE until fixed pilot-derived delta0 is frozen and locked outcomes are analyzed",
                        "cognition": "INCONCLUSIVE until locked paired cells, clustered intervals, Holm correction, fidelity and sham evidence are analyzed",
                        "biological_validity": "NOT_VALIDATED; explicitly out of scope"},
    }


def build_manifest(*, episodes_per_train_seed: int = 278) -> dict[str, Any]:
    if episodes_per_train_seed <= 0:
        raise ValueError("episodes_per_train_seed must be positive")
    seeds = _make_seed_sets(episodes_per_train_seed)
    all_new = set(seeds["training_root_seeds"])
    for seed in _walk_ints({k: v for k, v in seeds.items() if k != "training_world_seeds_by_root"}):
        all_new.add(seed)
    for row in seeds["training_world_seeds_by_root"].values():
        all_new.update(row)
    historical = historical_seed_ids()
    collision = all_new & historical
    if collision:
        raise ValueError(f"new seed split collides with historical MVP1 seed IDs: {sorted(collision)[:12]}")
    source_hashes = {relative: sha(ROOT / relative) for relative in SOURCE_PATHS}
    report_hash = sha(MVP1_REPORT_PATH)
    approved = approved_artifact_hashes()
    body = {
        "schema_version": 1,
        "protocol_id": "pasture-torch-mvp1-preregistered-v1",
        "status": "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0",
        "generated_from_sources_without_reading_model_outcomes": True,
        "scope": "synthetic simulation only; no biological/production husbandry claim; no MVP2",
        "approved_artifact_sha256": {**approved, "historical_mvp1_report": report_hash},
        "frozen_baseline": {"mvp1_report_path": "validation/MVP1_REMEDIATION_REPORT.md",
                            "mvp1_report_sha256": report_hash,
                            "historical_manifest_sha256": HISTORICAL_MANIFEST_SHA256,
                            "evidence_policy": "read-only; failed/inconclusive MVP1 outcomes and seed IDs are never changed or reused"},
        "seed_splits": seeds,
        "seed_overlap_audit": {"historical_seed_ids_count": len(historical),
                               "new_seed_ids_count": len(all_new), "overlap_count": 0,
                               "disjoint": True, "historical_values_sha256": canonical_sha(sorted(historical)),
                               "new_values_sha256": canonical_sha(sorted(all_new))},
        "source_sha256": source_hashes,
        "training_config": {"algorithm_arms": ["reinforce", "a2c"], "episodes_per_train_seed": episodes_per_train_seed,
                            "rollout_ticks": 72, "ticks_per_day": 24, "batch_episodes": 16,
                            "optimizer": "Adam", "learning_rate": 0.001, "grad_clip_norm": 2.0,
                            "dtype": "float32", "amp": False, "tf32": False,
                            "checkpoint_artifact_sha256": None,
                            "checkpoint_binding_status": "MUST_BIND_EACH_CHECKPOINT_BEFORE_ANY_LOCKED_OUTCOME_ACCESS"},
        "acceptance_thresholds": acceptance_spec(),
        "analysis_protocol": {"training_selection": "validation only; locked test is never used for tuning or checkpoint selection",
                              "heldout_pairing": "same world seed for trained, same-seed untrained, and heuristic policy",
                              "independent_unit": "checkpoint × world-seed episode cell; ticks/animals are repeated measures",
                              "locked_test_access": "forbidden until selected candidate, checkpoint hashes, source hashes, δ0, margins, sample size, analysis code and artifact hashes are frozen",
                              "outcome_artifacts_read_at_preregistration": [],
                              "precision_pilot_outputs": "variance/precision only; mask means, signs, differences and CI direction",
                              "inference": "deterministic argmax", "rewards": "existing synthetic rewards unchanged"},
    }
    body["manifest_sha256"] = canonical_sha(body)
    return body


def verify_manifest(manifest: dict[str, Any], *, verify_current_sources: bool = True) -> None:
    claimed = manifest.get("manifest_sha256")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if claimed != canonical_sha(body):
        raise ValueError("preregistration manifest checksum mismatch")
    approved = approved_artifact_hashes()
    if any(manifest["approved_artifact_sha256"].get(name) != value for name, value in approved.items()):
        raise ValueError("approved plan/review hash pins changed since preregistration")
    if manifest["frozen_baseline"]["mvp1_report_sha256"] != sha(MVP1_REPORT_PATH):
        raise ValueError("historical MVP1 report hash changed since preregistration")
    if verify_current_sources:
        actual = {relative: sha(ROOT / relative) for relative in SOURCE_PATHS}
        if actual != manifest["source_sha256"]:
            changed = sorted(k for k in actual if actual[k] != manifest["source_sha256"].get(k))
            raise ValueError(f"pre-outcome source hashes changed; regenerate/freeze protocol before outcomes: {changed}")
    split = manifest["seed_splits"]
    if split.get("training_root_seeds") != list(TRAIN_SEEDS):
        raise ValueError("protocol requires the exact approved ten paired training seeds")
    episode_count = manifest.get("training_config", {}).get("episodes_per_train_seed")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError("training episode count must be a positive integer")
    train_worlds = split.get("training_world_seeds_by_root")
    expected_root_keys = {str(seed) for seed in TRAIN_SEEDS}
    if not isinstance(train_worlds, dict) or set(train_worlds) != expected_root_keys:
        raise ValueError("training world seed map must contain exactly the approved training-root keys")
    expected_derivation = "sha256(root, train-world, episode-index), matching validation/train_torch_mvp1.py"
    if split.get("derivation", {}).get("training") != expected_derivation:
        raise ValueError("training world seed derivation metadata changed")
    all_train_worlds: set[int] = set()
    for root_seed in TRAIN_SEEDS:
        actual = train_worlds[str(root_seed)]
        if (not isinstance(actual, list) or len(actual) != episode_count
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in actual)):
            raise ValueError(f"training world seed list has invalid cardinality or values for root {root_seed}")
        expected = [derive_seed(root_seed, "train-world", i) for i in range(episode_count)]
        if actual != expected:
            raise ValueError(f"training world seeds do not match the frozen derivation for root {root_seed}")
        if len(actual) != len(set(actual)):
            raise ValueError(f"training world seeds contain duplicates for root {root_seed}")
        all_train_worlds.update(actual)
    if len(all_train_worlds) != episode_count * len(TRAIN_SEEDS):
        raise ValueError("training world seeds must be disjoint across all training roots")

    eval_sets: list[int] = []
    for name in ("validation", "precision_pilot", "locked_test", "precision_reserve"):
        family_map = split.get(name)
        if not isinstance(family_map, dict) or set(family_map) != set(FAMILIES):
            raise ValueError(f"{name} split must contain exactly the approved scenario families")
        for family in FAMILIES:
            actual = family_map[family]
            expected = list(SPLIT_SEEDS[name][family])
            if actual != expected:
                raise ValueError(f"{name} seeds changed from the approved frozen split for {family}")
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in actual):
                raise ValueError(f"{name} seed values must be integers")
            eval_sets.extend(actual)
    historical = historical_seed_ids()
    new_declared = set(TRAIN_SEEDS) | all_train_worlds | set(eval_sets)
    if len(eval_sets) != len(set(eval_sets)) or new_declared & historical:
        raise ValueError("training-world and evaluation seed IDs must be pairwise disjoint from each other and historical MVP1")
    if all_train_worlds.intersection(eval_sets):
        raise ValueError("training-world seeds must be disjoint from all evaluation split seeds")
    delta0 = manifest["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"]
    status = manifest.get("status")
    if status == "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0":
        if delta0 is not None or manifest["acceptance_thresholds"]["training_validity"]["delta0_status"] != "INCONCLUSIVE_PENDING_PRECISION_ONLY_PILOT":
            raise ValueError("pending preregistration must not contain pilot-derived delta0")
        if "pre_outcome_freeze" in manifest:
            raise ValueError("pending preregistration cannot contain a pre-outcome freeze")
    elif status == "LOCKED_PRE_OUTCOME":
        freeze = manifest.get("pre_outcome_freeze")
        if not isinstance(freeze, dict):
            raise ValueError("locked manifest requires a pre_outcome_freeze attestation")
        if not isinstance(delta0, dict) or set(delta0) != set(FAMILIES):
            raise ValueError("locked manifest requires pilot-derived delta0 for every family")
        if manifest["acceptance_thresholds"]["training_validity"]["delta0_status"] != "FROZEN_FROM_PRECISION_ONLY_PILOT":
            raise ValueError("locked manifest delta0 status is not frozen from pilot")
        for family, value in delta0.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"locked manifest has invalid delta0 for {family}")
        if freeze.get("source_sha256") != manifest.get("source_sha256"):
            raise ValueError("locked freeze source hashes do not match the preregistered source hashes")
        if not isinstance(freeze.get("pilot_artifact_sha256"), str) or len(freeze["pilot_artifact_sha256"]) != 64:
            raise ValueError("locked manifest must bind a masked precision-pilot artifact")
        pilot_sds = freeze.get("pilot_sd_untrained_returns_by_family")
        if not isinstance(pilot_sds, dict) or set(pilot_sds) != set(FAMILIES):
            raise ValueError("locked manifest must retain the masked pilot SD values used for delta0")
        for family in FAMILIES:
            sd = pilot_sds[family]
            if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not math.isfinite(sd) or sd <= 0:
                raise ValueError(f"locked manifest has invalid pilot SD for {family}")
            if not math.isclose(float(delta0[family]), .05 * float(sd), rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"locked manifest delta0 does not match the preregistered pilot formula for {family}")
        expected_checkpoints = {f"{algorithm}:{seed}" for algorithm in ("reinforce", "a2c") for seed in TRAIN_SEEDS}
        if set(freeze.get("checkpoint_sha256", {})) != expected_checkpoints:
            raise ValueError("locked manifest must bind both algorithms' ten selected checkpoints")
        if set(freeze.get("checkpoint_manifest_sha256", {})) != expected_checkpoints:
            raise ValueError("locked manifest must bind each selected checkpoint generation manifest")
        if any(not isinstance(value, str) or len(value) != 64 for value in freeze["checkpoint_sha256"].values()):
            raise ValueError("locked manifest contains an invalid checkpoint hash")
        if any(not isinstance(value, str) or len(value) != 64 for value in freeze["checkpoint_manifest_sha256"].values()):
            raise ValueError("locked manifest contains an invalid checkpoint generation-manifest hash")
        analysis_hashes = freeze.get("analysis_sha256", {})
        required_analysis = {"validation/evaluate_torch_mvp1.py", "validation/benchmark_torch_fixed_corpus.py"}
        if set(analysis_hashes) != required_analysis:
            raise ValueError("locked manifest must bind exactly evaluator and fixed-corpus analysis code")
        if any(not isinstance(value, str) or len(value) != 64 for value in analysis_hashes.values()):
            raise ValueError("locked manifest contains an invalid analysis-code hash")
        actual_analysis = {relative: sha(ROOT / relative) for relative in required_analysis}
        if analysis_hashes != actual_analysis:
            raise ValueError("locked analysis-code hash does not match the actual evaluator/fixed-corpus files")
    else:
        raise ValueError(f"unsupported preregistration status {status!r}")


def _named_artifact_hashes(items: list[str], *, label: str, required: set[str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} artifact must use NAME=PATH: {item!r}")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path or name in result:
            raise ValueError(f"invalid or duplicate {label} artifact name: {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"{label} artifact is not a file: {path}")
        result[name] = sha(path)
    if required and not required.issubset(result):
        raise ValueError(f"{label} artifacts missing required names: {sorted(required - set(result))}")
    return result


def _checkpoint_artifact_hashes(items: list[str], *, pending_source_hashes: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Hash each selected generation state and verify its algorithm/seed identity.

    The lock binds the immutable ``training.pt`` bytes.  Its adjacent generation
    manifest is checked before trusting the declared algorithm, seed, and state
    digest; no checkpoint is unpickled while preregistering.
    """
    state_hashes = {}
    manifest_hashes = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"checkpoint artifact must use NAME=PATH: {item!r}")
        name, raw_path = item.split("=", 1)
        if name in state_hashes or not raw_path:
            raise ValueError(f"invalid or duplicate checkpoint artifact name: {name!r}")
        try:
            algorithm, raw_seed = name.split(":", 1)
            seed = int(raw_seed)
        except (ValueError, TypeError):
            raise ValueError(f"checkpoint name must be algorithm:training-seed: {name!r}")
        if algorithm not in {"reinforce", "a2c"} or seed not in TRAIN_SEEDS:
            raise ValueError(f"checkpoint name is not a selected training arm/seed: {name!r}")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file() or path.name != "training.pt":
            raise ValueError(f"checkpoint artifact must be a generation training.pt file: {path}")
        if not re.fullmatch(r"generation-[0-9]{8}", path.parent.name):
            raise ValueError(f"checkpoint artifact is not inside a committed generation directory: {path}")
        manifest_path = path.parent / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"checkpoint generation manifest is missing: {manifest_path}")
        try:
            generation = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoint generation manifest is invalid: {manifest_path}") from exc
        config = generation.get("config")
        if (generation.get("format_version") != 1 or generation.get("generation") != int(path.parent.name[-8:])
                or generation.get("algorithm") != algorithm or not isinstance(config, dict)
                or isinstance(config.get("seed"), bool) or config.get("seed") != seed):
            raise ValueError(f"checkpoint artifact metadata does not match selected {name}")
        checkpoint_sources = config.get("source_hashes")
        expected_source_hashes = {relative: pending_source_hashes[relative]
                                  for relative in TRAINING_SOURCE_PATHS}
        if not isinstance(checkpoint_sources, dict) or checkpoint_sources != expected_source_hashes:
            raise ValueError(f"checkpoint training source hashes do not match the frozen preregistration for {name}")
        digest = sha(path)
        if generation.get("state_file") != path.name or generation.get("state_sha256") != digest:
            raise ValueError(f"checkpoint artifact state hash does not match its generation manifest: {path}")
        state_hashes[name] = digest
        manifest_hashes[name] = sha(manifest_path)
    expected = {f"{algorithm}:{seed}" for algorithm in ("reinforce", "a2c") for seed in TRAIN_SEEDS}
    if set(state_hashes) != expected:
        raise ValueError("freeze must include every selected REINFORCE and A2C checkpoint (ten seeds each)")
    return state_hashes, manifest_hashes


def finalize_manifest(pending: dict[str, Any], pilot_artifact: dict[str, Any], *,
                      checkpoint_artifacts: list[str], analysis_artifacts: list[str]) -> dict[str, Any]:
    """Create a locked copy only from a masked precision pilot and frozen artifacts.

    Pilot input is deliberately schema-limited to SDs and provenance. Means,
    signs, effects, intervals, rewards, and action metrics are rejected. This
    workflow never reads locked outcomes and never modifies the pending input.
    """
    verify_manifest(pending, verify_current_sources=True)
    if pending.get("status") != "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0":
        raise ValueError("only a pending preregistration can be finalized")
    allowed = {"schema_version", "status", "manifest_sha256", "pilot_seed_ids_sha256", "sd_untrained_returns_by_family"}
    if set(pilot_artifact) != allowed:
        raise ValueError("pilot artifact has unsupported fields; only masked precision/provenance fields are accepted")
    if pilot_artifact.get("schema_version") != 1 or pilot_artifact.get("status") != "MASKED_PRECISION_ONLY":
        raise ValueError("pilot artifact must attest MASKED_PRECISION_ONLY schema version 1")
    if pilot_artifact.get("manifest_sha256") != pending.get("manifest_sha256"):
        raise ValueError("pilot artifact was not generated from this pending manifest")
    pilot_seeds = pending["seed_splits"]["precision_pilot"]
    expected_pilot_seed_hash = canonical_sha({family: pilot_seeds[family] for family in FAMILIES})
    if pilot_artifact.get("pilot_seed_ids_sha256") != expected_pilot_seed_hash:
        raise ValueError("pilot artifact seed provenance does not match the preregistered pilot split")
    sds = pilot_artifact.get("sd_untrained_returns_by_family")
    if not isinstance(sds, dict) or set(sds) != set(FAMILIES):
        raise ValueError("pilot artifact must provide one masked SD per scenario family")
    delta0 = {}
    for family in FAMILIES:
        sd = sds[family]
        if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"precision pilot SD must be finite and positive for {family}")
        delta0[family] = 0.05 * float(sd)

    pending_source_hashes = pending["source_sha256"]
    checkpoints, checkpoint_manifests = _checkpoint_artifact_hashes(
        checkpoint_artifacts, pending_source_hashes=pending_source_hashes)
    required_analysis = {"validation/evaluate_torch_mvp1.py", "validation/benchmark_torch_fixed_corpus.py"}
    analysis = _named_artifact_hashes(analysis_artifacts, label="analysis", required=required_analysis)
    if set(analysis) != required_analysis:
        raise ValueError("freeze must include exactly evaluator and fixed-corpus analysis code")
    actual_analysis = {name: sha(ROOT / name) for name in required_analysis}
    if analysis != actual_analysis:
        raise ValueError("analysis artifacts must be the actual current evaluator and fixed-corpus files")
    locked = json.loads(json.dumps(pending))
    locked["status"] = "LOCKED_PRE_OUTCOME"
    training = locked["acceptance_thresholds"]["training_validity"]
    training["near_zero_margin_delta0_by_family"] = delta0
    training["delta0_status"] = "FROZEN_FROM_PRECISION_ONLY_PILOT"
    locked["pre_outcome_freeze"] = {
        "pilot_artifact_sha256": canonical_sha(pilot_artifact),
        "pilot_seed_ids_sha256": expected_pilot_seed_hash,
        "pilot_sd_untrained_returns_by_family": {family: float(sds[family]) for family in FAMILIES},
        "checkpoint_sha256": checkpoints,
        "checkpoint_manifest_sha256": checkpoint_manifests,
        "analysis_sha256": analysis,
        "source_sha256": dict(locked["source_sha256"]),
        "locked_outcomes_read": False,
        "freeze_semantics": "hashes and pilot-derived margins frozen before any locked-test outcome access",
    }
    locked["manifest_sha256"] = canonical_sha({k: v for k, v in locked.items() if k != "manifest_sha256"})
    verify_manifest(locked, verify_current_sources=True)
    return locked


def write_manifest(path: str | Path = OUTPUT_PATH, *, episodes_per_train_seed: int = 278) -> Path:
    manifest = build_manifest(episodes_per_train_seed=episodes_per_train_seed)
    verify_manifest(manifest, verify_current_sources=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--episodes-per-train-seed", type=int, default=278)
    parser.add_argument("--finalize-from", type=Path, help="pending manifest to finalize using masked pilot and frozen artifact files")
    parser.add_argument("--pilot-artifact", type=Path, help="masked precision-only pilot JSON (no means/effects/outcomes)")
    parser.add_argument("--checkpoint-artifact", action="append", default=[], metavar="NAME=PATH",
                        help="selected checkpoint file; repeat for reinforce:SEED and a2c:SEED, all ten seeds each")
    parser.add_argument("--analysis-artifact", action="append", default=[], metavar="NAME=PATH",
                        help="frozen analysis file; include evaluator and fixed-corpus scripts")
    args = parser.parse_args(argv)
    if args.finalize_from:
        if not args.pilot_artifact:
            parser.error("--finalize-from requires --pilot-artifact")
        pending = json.loads(args.finalize_from.read_text())
        pilot = json.loads(args.pilot_artifact.read_text())
        finalized = finalize_manifest(pending, pilot, checkpoint_artifacts=args.checkpoint_artifact,
                                      analysis_artifacts=args.analysis_artifact)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(finalized, indent=2, sort_keys=True) + "\n")
        target = args.output
    else:
        if args.pilot_artifact or args.checkpoint_artifact or args.analysis_artifact:
            parser.error("pilot/checkpoint/analysis artifacts require --finalize-from")
        target = write_manifest(args.output, episodes_per_train_seed=args.episodes_per_train_seed)
    manifest = json.loads(target.read_text())
    print(json.dumps({"output": str(target), "manifest_sha256": manifest["manifest_sha256"],
                      "seed_overlap_audit": manifest["seed_overlap_audit"],
                      "locked_seed_counts": {k: len(v) for k, v in manifest["seed_splits"]["locked_test"].items()},
                      "delta0_status": manifest["acceptance_thresholds"]["training_validity"]["delta0_status"],
                      "locked_outcomes_read": False}, indent=2))


if __name__ == "__main__":
    main()
