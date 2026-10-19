#!/usr/bin/env python3
"""Analyze preregistered paired Torch cohort outputs behind a fail-closed loader.

``analyze_locked_cohort`` is a pure hashable analysis function. The guarded
``analyze_locked_files`` loader authenticates and appends an access audit before
reading evaluator documents; it rejects historical freezes that do not bind
the new analysis code and remains blocked by the empty trusted-root allowlist.
The module does not create worlds or train policies.

The implementation follows the approved nine-test utility family and separate
four-test cognition family. Results are synthetic-simulator evidence only.
"""
from __future__ import annotations

import hashlib
import math
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from validation.torch_locked_statistics import (
    centered_bootstrap_p_value,
    holm_adjust,
    holm_inverted_lower_bound,
)

FAMILIES = ("starter", "resource_scarce_relocated", "wide", "unseen_corridor")
ARMS = ("reinforce", "a2c")
COGNITION_TESTS = ("memory_effect", "perception_effect", "memory_attenuation", "perception_attenuation")
LOCKED_ANALYSIS_FILES = ("validation/evaluate_torch_mvp1.py", "validation/benchmark_torch_fixed_corpus.py",
                         "validation/analyze_torch_cohort.py", "validation/torch_locked_statistics.py")
ANALYSIS_CONFIG_KEYS = {"bootstrap_method", "bootstrap_resamples", "bootstrap_seed"}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _run_index(documents: Sequence[Mapping[str, Any]], *, expected_manifest_sha256: str,
               seeds: Sequence[int], expected_worlds: Mapping[str, Sequence[int]],
               expected_provenance: Mapping[str, Any]) -> dict[tuple[str, int, str, int, str], dict[str, Any]]:
    """Validate identity/split completeness and index per-policy episode rows."""
    expected_seeds = set(seeds)
    expected = {(arm, seed) for arm in ARMS for seed in expected_seeds}
    seen_docs: set[tuple[str, int]] = set()
    indexed: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
    for doc in documents:
        if not isinstance(doc, Mapping):
            raise ValueError("each checkpoint evaluation must be an object")
        if doc.get("locked_pre_outcome_binding") is None:
            raise ValueError("cohort document is not bound to a locked pre-outcome freeze")
        binding = doc["locked_pre_outcome_binding"]
        if binding.get("preregistration_manifest_sha256") != expected_manifest_sha256:
            raise ValueError("cohort document is bound to a different preregistration manifest")
        arm, seed = doc.get("algorithm"), doc.get("training_seed")
        if arm not in ARMS or type(seed) is not int or seed not in expected_seeds:
            raise ValueError("cohort document has an unexpected arm or training seed")
        key = (arm, int(seed))
        expected_checkpoint = expected_provenance["checkpoints"].get(f"{arm}:{seed}")
        checkpoint_binding = binding.get("checkpoint_binding", {})
        if not isinstance(expected_checkpoint, Mapping) or any(
                checkpoint_binding.get(field) != expected_checkpoint.get(field)
                for field in ("checkpoint_sha256", "checkpoint_manifest_sha256", "checkpoint_key")):
            raise ValueError(f"checkpoint provenance does not match frozen selection for {arm}:{seed}")
        if checkpoint_binding.get("source_sha256") != expected_provenance.get("source_sha256"):
            raise ValueError("checkpoint source hashes do not match the frozen release source set")
        if checkpoint_binding.get("analysis_sha256") != expected_provenance.get("analysis_sha256"):
            raise ValueError("checkpoint analysis hashes do not match the frozen analysis source set")
        if (binding.get("authorization_id") != expected_provenance.get("authorization_id")
                or binding.get("authorization_sha256") != expected_provenance.get("authorization_sha256")
                or binding.get("trusted_root_sha256") != expected_provenance.get("trusted_root_sha256")
                or binding.get("access_audit_record_sha256") != expected_provenance.get("audit_records", {}).get(f"{arm}:{seed}")):
            raise ValueError("cohort document is missing the frozen authorization/audit binding")
        if key in seen_docs:
            raise ValueError(f"duplicate checkpoint evaluation document: {key}")
        seen_docs.add(key)
        split = doc.get("seed_split", {})
        if split.get("evaluation_split") != "locked_test":
            raise ValueError("cohort analysis accepts only frozen locked-test evaluation documents")
        rows = doc.get("runs")
        if not isinstance(rows, list):
            raise ValueError("checkpoint evaluation is missing run rows")
        row_keys: set[tuple[str, int, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("episode run row must be an object")
            family, world_seed, policy = row.get("scenario"), row.get("world_seed"), row.get("policy")
            if family not in FAMILIES or type(world_seed) is not int:
                raise ValueError("episode row has an invalid family or world seed")
            if world_seed not in set(expected_worlds[family]):
                raise ValueError("episode row includes a world outside its registered family schedule")
            if policy not in {"trained", "untrained_same_seed", "heuristic_baseline"}:
                raise ValueError("episode row has an invalid policy label")
            unique = (str(family), world_seed, str(policy))
            if unique in row_keys:
                raise ValueError(f"duplicate policy/world episode row: {unique}")
            row_keys.add(unique)
            row_id = (arm, int(seed), str(family), world_seed, str(policy))
            indexed[row_id] = dict(row)
        expected_rows = {(family, world, policy)
                         for family in FAMILIES for world in expected_worlds[family]
                         for policy in ("trained", "untrained_same_seed", "heuristic_baseline")}
        if row_keys != expected_rows:
            raise ValueError(f"incomplete evaluator rows for {arm}:{seed}; missing={len(expected_rows-row_keys)}, extra={len(row_keys-expected_rows)}")
    if seen_docs != expected:
        raise ValueError(f"cohort must contain exactly every arm × registered checkpoint; missing={sorted(expected-seen_docs)}")
    return indexed


def _matrix(index: Mapping[tuple[str, int, str, int, str], dict[str, Any]], *, arm: str,
            family: str, seeds: Sequence[int], worlds: Sequence[int], policy: str,
            baseline_policy: str | None = None) -> list[list[float]]:
    values = []
    for seed in seeds:
        row = []
        for world in worlds:
            result = _finite(index[(arm, seed, family, world, policy)].get("total_reward"), "episode total_reward")
            if baseline_policy:
                baseline = _finite(index[(arm, seed, family, world, baseline_policy)].get("total_reward"), "baseline total_reward")
                result -= baseline
            row.append(result)
        values.append(row)
    return values


def _test_record(name: str, cells: Sequence[Sequence[float]], margin: float, *, seed: int,
                 resamples: int) -> dict[str, Any]:
    p = centered_bootstrap_p_value(cells, margin, resamples=resamples, seed=seed)
    return {"name": name, "estimate": float(sum(map(sum, cells)) / (len(cells) * len(cells[0]))),
            "null_margin": float(margin), "raw_one_sided_p": float(p)}


def analyze_locked_cohort(documents: Sequence[Mapping[str, Any]], *,
                          expected_manifest_sha256: str,
                          expected_provenance: Mapping[str, Any],
                          training_seeds: Sequence[int],
                          worlds_by_family: Mapping[str, Sequence[int]],
                          delta0_by_family: Mapping[str, float],
                          selected_cognition_arm: str,
                          validation_selection_sha256: str,
                          bootstrap_resamples: int = 10_000,
                          seed: int = 20261018) -> dict[str, Any]:
    """Compute the preregistered utility/algorithm and selected-arm cognition families.

    This function must only be called after external authorization, freeze
    verification and append-only access audit. It has no file access itself,
    and synthetic unit tests exercise it exclusively with fabricated records.
    """
    if selected_cognition_arm not in ARMS:
        raise ValueError("selected cognition arm must be frozen as reinforce or a2c before locked access")
    if len(validation_selection_sha256) != 64 or any(c not in "0123456789abcdef" for c in validation_selection_sha256):
        raise ValueError("validation-only cognition-arm selection hash is required")
    if set(worlds_by_family) != set(FAMILIES) or set(delta0_by_family) != set(FAMILIES):
        raise ValueError("world schedule and frozen delta0 must cover exactly the four preregistered families")
    for family in FAMILIES:
        worlds = worlds_by_family[family]
        if len(worlds) < 30 or len(worlds) != len(set(worlds)):
            raise ValueError(f"{family} must have at least 30 unique locked worlds")
        d0 = _finite(delta0_by_family[family], f"delta0 for {family}")
        if d0 <= 0:
            raise ValueError("each frozen delta0 must be positive")
    if len(training_seeds) != 10 or len(set(training_seeds)) != 10 or any(type(x) is not int for x in training_seeds):
        raise ValueError("locked analysis requires the ten preregistered paired training seeds")
    selection = expected_provenance.get("validation_selection", {})
    if (selection.get("algorithm") != selected_cognition_arm
            or selection.get("sha256") != validation_selection_sha256
            or not isinstance(expected_provenance.get("checkpoints"), Mapping)
            or set(expected_provenance["checkpoints"]) != {f"{a}:{k}" for a in ARMS for k in training_seeds}
            or not isinstance(expected_provenance.get("source_sha256"), Mapping)
            or not isinstance(expected_provenance.get("analysis_sha256"), Mapping)
            or not isinstance(expected_provenance.get("audit_records"), Mapping)
            or not expected_provenance.get("authorization_id")):
        raise ValueError("frozen release/selection/auth/audit provenance is incomplete or inconsistent")
    index = _run_index(documents, expected_manifest_sha256=expected_manifest_sha256,
                       seeds=training_seeds, expected_worlds=worlds_by_family,
                       expected_provenance=expected_provenance)

    raw_specs: list[tuple[str, list[list[float]], float]] = []
    for arm in ARMS:
        for family in FAMILIES:
            worlds = worlds_by_family[family]
            trained = _matrix(index, arm=arm, family=family, seeds=training_seeds,
                              worlds=worlds, policy="trained")
            untrained = _matrix(index, arm=arm, family=family, seeds=training_seeds,
                                worlds=worlds, policy="untrained_same_seed")
            effects = [[a-b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(trained, untrained)]
            baseline_mean = sum(map(sum, untrained)) / (len(untrained) * len(untrained[0]))
            if family == "starter":
                margin = .05 * abs(baseline_mean) if abs(baseline_mean) >= .1 else float(delta0_by_family[family])
                raw_specs.append((f"{arm}:starter_utility", effects, margin))
            else:
                margin = .10 * abs(baseline_mean) if abs(baseline_mean) >= .1 else float(delta0_by_family[family])
                raw_specs.append((f"{arm}:{family}_non_harm", effects, -margin))
    a2c = _matrix(index, arm="a2c", family="starter", seeds=training_seeds,
                  worlds=worlds_by_family["starter"], policy="trained")
    reinforce = _matrix(index, arm="reinforce", family="starter", seeds=training_seeds,
                        worlds=worlds_by_family["starter"], policy="trained")
    raw_specs.append(("a2c_minus_reinforce:starter", [[a-b for a, b in zip(ar, rr)] for ar, rr in zip(a2c, reinforce)], 0.0))
    if len(raw_specs) != 9:
        raise AssertionError("confirmatory utility family schema drift")

    # Keep descriptive failure diagnostics for each checkpoint/family. These
    # metrics do not add tests or alter any preregistered threshold.
    diagnostics = []
    for arm in ARMS:
        for family in FAMILIES:
            for train_seed in training_seeds:
                rows = [index[(arm, int(train_seed), family, int(world), "trained")]
                        for world in worlds_by_family[family]]
                action_counts: dict[str, int] = {}
                navigation: dict[str, int] = {}
                drives: dict[str, list[float]] = {}
                resource_counts: dict[str, list[float]] = {}
                reward_trajectories = []
                invalid = 0
                entropies = []
                for row in rows:
                    counts = row.get("action_counts")
                    if not isinstance(counts, Mapping) or not counts:
                        raise ValueError("evaluation row is missing action-count diagnostics")
                    for action, count in counts.items():
                        if type(count) is not int or count < 0:
                            raise ValueError("action counts must be nonnegative integers")
                        action_counts[action] = action_counts.get(action, 0) + count
                    if type(row.get("invalid_actions")) is not int or row["invalid_actions"] < 0:
                        raise ValueError("invalid-action diagnostic must be a nonnegative integer")
                    invalid += row["invalid_actions"]
                    entropies.append(_finite(row.get("mean_policy_entropy_nats"), "policy entropy"))
                    for reason, count in row.get("navigation_reason_counts", {}).items():
                        if type(count) is not int or count < 0:
                            raise ValueError("navigation reason counts must be nonnegative integers")
                        navigation[reason] = navigation.get(reason, 0) + count
                    for drive, value in row.get("final_drive_means", {}).items():
                        drives.setdefault(drive, []).append(_finite(value, f"final drive {drive}"))
                    for resource, value in row.get("resource_state", {}).items():
                        resource_counts.setdefault(resource, []).append(_finite(value, f"resource state {resource}"))
                    trajectory = row.get("reward_trajectory_per_tick_all_animals")
                    if not isinstance(trajectory, list) or not trajectory:
                        raise ValueError("evaluation row is missing a reward trajectory")
                    reward_trajectories.append([_finite(v, "reward trajectory value") for v in trajectory])
                lengths = {len(row) for row in reward_trajectories}
                if len(lengths) != 1:
                    raise ValueError("reward trajectory lengths differ within a checkpoint/family")
                decisions = sum(action_counts.values())
                diagnostics.append({
                    "algorithm": arm, "training_seed": int(train_seed), "scenario_family": family,
                    "episode_cells": len(rows),
                    "mean_episode_total_reward": sum(_finite(r.get("total_reward"), "episode reward") for r in rows) / len(rows),
                    "action_counts": dict(sorted(action_counts.items())),
                    "action_frequencies": {a: n / decisions for a, n in sorted(action_counts.items())} if decisions else {},
                    "single_action_only": bool(decisions and max(action_counts.values()) == decisions),
                    "invalid_actions": invalid,
                    "mean_policy_entropy_nats": sum(entropies) / len(entropies),
                    "mean_final_drive_means": {key: sum(values) / len(values) for key, values in sorted(drives.items())},
                    "navigation_reason_counts": dict(sorted(navigation.items())),
                    "mean_resource_state": {key: sum(values) / len(values) for key, values in sorted(resource_counts.items())},
                    "mean_reward_trajectory_per_tick_all_animals": [
                        sum(values[tick] for values in reward_trajectories) / len(reward_trajectories)
                        for tick in range(next(iter(lengths)))],
                    "descriptive_only": True,
                })

    preliminary = [_test_record(name, cells, margin, seed=seed+i, resamples=bootstrap_resamples)
                   for i, (name, cells, margin) in enumerate(raw_specs)]
    raw_p = [x["raw_one_sided_p"] for x in preliminary]
    adjusted = holm_adjust(raw_p)
    utility_tests = []
    for i, ((name, cells, margin), row) in enumerate(zip(raw_specs, preliminary)):
        others = [p for j, p in enumerate(raw_p) if j != i]
        row["holm_adjusted_p"] = float(adjusted[i])
        row["holm_lower_bound"] = holm_inverted_lower_bound(cells, others, target_seed=seed+i,
                                                              resamples=bootstrap_resamples, alpha=.05)
        row["clears_margin"] = bool(row["holm_lower_bound"] is not None and row["holm_lower_bound"] > margin)
        utility_tests.append(row)

    cognition_rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    expected_cog = {(int(k), family, int(world)) for k in training_seeds
                    for family in FAMILIES for world in worlds_by_family[family]}
    for doc in documents:
        if doc.get("algorithm") != selected_cognition_arm:
            continue
        seed_id = int(doc["training_seed"])
        for row in doc.get("paired_cognition_probe_rows", []):
            if row.get("policy") != "trained":
                continue
            family, world = row.get("scenario_family"), row.get("world_seed")
            if family not in FAMILIES or type(world) is not int:
                raise ValueError("cognition row has an invalid family or world seed type")
            key = (seed_id, family, world)
            if key in cognition_rows:
                raise ValueError(f"duplicate selected-arm cognition cell {key}")
            cognition_rows[key] = row.get("probe", {})
    if set(cognition_rows) != expected_cog:
        raise ValueError(f"selected-arm cognition cohort incomplete; missing={len(expected_cog-set(cognition_rows))}, extra={len(set(cognition_rows)-expected_cog)}")

    cog_cells: dict[str, list[list[float]]] = {name: [] for name in COGNITION_TESTS}
    for seed_id in training_seeds:
        cell_values = {name: [] for name in COGNITION_TESTS}
        for family in FAMILIES:
            for world in worlds_by_family[family]:
                probe = cognition_rows[(int(seed_id), family, int(world))]
                contrasts = probe.get("contrasts", {})
                for label, key in (("memory_effect", "memory_success"), ("perception_effect", "perception_available")):
                    contrast = contrasts.get(key)
                    if not isinstance(contrast, Mapping):
                        raise ValueError("cognition probe is missing a preregistered paired contrast")
                    fidelity = contrast.get("fidelity", {})
                    sham = contrast.get("sham", {})
                    expected_direction = ("memory_success_minus_blocked" if key == "memory_success"
                                          else "water_available_feature_minus_unavailable_feature")
                    registered_features = ({"mem_water_success", "mem_water_blocked"} if key == "memory_success"
                                           else {"water_available"})
                    expected_fields = {"baseline", "changed", "baseline_ablation", "changed_ablation"}
                    input_hashes = contrast.get("input_sha256", {})
                    if (contrast.get("direction") != expected_direction
                            or set(contrast.get("changed_feature_names", [])) != registered_features
                            or not isinstance(input_hashes, Mapping)
                            or set(input_hashes) != expected_fields
                            or any(not isinstance(v, str) or len(v) != 64 or
                                   any(c not in "0123456789abcdef" for c in v)
                                   for v in input_hashes.values())):
                        raise ValueError("cognition intervention direction/input digest does not match the preregistered contrast")
                    if (sham.get("input_bit_identical") is not True
                            or sham.get("probabilities_bit_identical") is not True
                            or sham.get("input_sha256") != input_hashes.get("baseline")
                            or sham.get("repeat_input_sha256") != input_hashes.get("baseline")):
                        raise ValueError("cognition probe failed bit-identical sham")
                    if (set(fidelity.get("actual_features_changed", [])) != registered_features
                            or set(fidelity.get("expected_features_changed", [])) != registered_features):
                        raise ValueError("cognition intervention changed unexpected features")
                    _finite(contrast.get("drink_probability_effect"), f"{label} effect")
                    attenuation = contrast.get("ablation", {}).get("attenuation")
                    full_effect = _finite(contrast.get("drink_probability_effect"), f"{label} effect")
                    attenuation = _finite(attenuation, f"{label} attenuation")
                    cell_values[label].append(full_effect)
                    cell_values[label.replace("effect", "attenuation")].append(attenuation)
        for name in COGNITION_TESTS:
            cog_cells[name].append(cell_values[name])
    if any(not all(math.isfinite(v) for row in mat for v in row) for mat in cog_cells.values()):
        raise ValueError("cognition ablation attenuation fraction is missing/nonfinite")
    cog_raw_p = [centered_bootstrap_p_value(cog_cells[name], .05 if name.endswith("effect") else 0.0,
                                           resamples=bootstrap_resamples, seed=seed+100+i)
                 for i, name in enumerate(COGNITION_TESTS)]
    cog_adjusted = holm_adjust(cog_raw_p)
    cognition_tests = []
    for i, name in enumerate(COGNITION_TESTS):
        margin = .05 if name.endswith("effect") else 0.0
        others = [p for j, p in enumerate(cog_raw_p) if j != i]
        lower = holm_inverted_lower_bound(cog_cells[name], others, target_seed=seed+100+i,
                                           resamples=bootstrap_resamples, alpha=.05)
        point = sum(map(sum, cog_cells[name])) / sum(len(x) for x in cog_cells[name])
        point_fraction_gate = True
        if name.endswith("attenuation"):
            # Point attenuation must reach at least 50% of the paired full
            # effect, using the same checkpoint × world cell weighting.
            corresponding = "memory_effect" if name == "memory_attenuation" else "perception_effect"
            full_mean = sum(map(sum, cog_cells[corresponding])) / sum(len(x) for x in cog_cells[corresponding])
            point_fraction_gate = bool(full_mean > 0 and point >= .5 * full_mean)
        cognition_tests.append({"name": name, "estimate": float(point), "null_margin": margin,
                                "raw_one_sided_p": float(cog_raw_p[i]), "holm_adjusted_p": float(cog_adjusted[i]),
                                "holm_lower_bound": lower,
                                "clears_margin": bool(lower is not None and lower > margin),
                                "mean_cell_attenuation_fraction_at_least_0_5": point_fraction_gate})

    arm_gates = {}
    for arm in ARMS:
        tests = {row["name"].split(":", 1)[1]: row for row in utility_tests if row["name"].startswith(arm + ":")}
        arm_gates[arm] = bool(tests["starter_utility"]["clears_margin"] and
                              all(tests[f"{family}_non_harm"]["clears_margin"] for family in FAMILIES[1:]))
    return {"schema_version": 1, "scope": "synthetic simulator only; not biological validation",
            "manifest_sha256": expected_manifest_sha256,
            "input_payload_values_analyzed": True,
            "filesystem_accessed_by_analyzer": False,
            "provenance_validation": "compared to caller-supplied frozen bindings; signature, trust-root, and audit-chain verification must occur in guarded loader before these inputs are read",
            "bootstrap": {"method": "paired two-way crossed checkpoint × world resampling", "resamples": bootstrap_resamples,
                          "seed": seed, "confidence": "one-sided Holm-inverted familywise 95% lower bounds"},
            "utility_family": {"family_size": 9, "alpha": .05, "tests": utility_tests,
                              "per_arm_learned_utility_gate": arm_gates,
                              "overall_learned_utility_gate": bool(any(arm_gates.values()))},
            "cognition_family": {"family_size": 4, "alpha": .05, "selected_arm": selected_cognition_arm,
                                 "validation_selection_sha256": validation_selection_sha256,
                                 "tests": cognition_tests,
                                 "gate": bool(all(x["clears_margin"] for x in cognition_tests) and
                                              all(x["mean_cell_attenuation_fraction_at_least_0_5"] for x in cognition_tests[2:]))},
            "descriptive_checkpoint_family_diagnostics": diagnostics,
            "limitations": ["utility report is not released until pre-outcome freeze and signed-access governance pass",
                            "outputs remain synthetic simulation evidence; no biological validity claim"]}


def _read_audit_chain(path: Path) -> list[dict[str, Any]]:
    """Read/verify only the metadata audit chain; never read cohort payloads."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("locked analysis denied: audit file must be owner-only regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    payload = b"".join(chunks)
    if payload and not payload.endswith(b"\n"):
        raise ValueError("locked analysis denied: audit chain has an incomplete trailing record")
    previous = "0" * 64
    records = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"locked analysis denied: audit record {line_number} is invalid") from exc
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode()).hexdigest()
        if record.get("previous_record_sha256") != previous or record.get("record_sha256") != digest:
            raise ValueError(f"locked analysis denied: audit chain integrity failure at record {line_number}")
        records.append(record)
        previous = digest
    return records


def _read_regular_json_no_follow(path: Path, *, max_bytes: int = 256 * 1024 * 1024,
                                 expected_sha256: str | None = None) -> tuple[Any, str]:
    """Read/hash exact bounded bytes; optionally verify digest before JSON parsing."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("locked analysis denied: evaluator document must be a bounded regular file")
        payload = bytearray()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > max_bytes:
                raise ValueError("locked analysis denied: evaluator document exceeded the size limit")
    finally:
        os.close(fd)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("locked analysis denied: evaluator artifact bytes do not match audit")
    try:
        return json.loads(payload), digest
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("locked analysis denied: evaluator document is not valid JSON") from exc


def _verify_artifact_audit_record(record: Mapping[str, Any], *, key: str,
                                 authorization: Mapping[str, Any], manifest_sha256: str,
                                 freeze: Mapping[str, Any], access_record_sha256: str,
                                 artifact_name: str) -> str:
    """Validate the post-evaluation audit binding without touching outcome bytes."""
    expected_sha = freeze.get("checkpoint_sha256", {}).get(key)
    expected_manifest_sha = freeze.get("checkpoint_manifest_sha256", {}).get(key)
    digest = record.get("evaluation_artifact_sha256")
    if (record.get("action") != "locked_test_evaluation_artifact_committed"
            or record.get("selected_checkpoint_key") != key
            or record.get("authorization_id") != authorization.get("authorization_id")
            or record.get("authorization_sha256") != authorization.get("authorization_sha256")
            or record.get("preregistration_manifest_sha256") != manifest_sha256
            or record.get("selected_checkpoint_sha256") != expected_sha
            or record.get("selected_checkpoint_manifest_sha256") != expected_manifest_sha
            or record.get("analysis_sha256") != freeze.get("analysis_sha256")
            or record.get("evaluation_access_record_sha256") != access_record_sha256
            or record.get("evaluation_artifact_name") != artifact_name
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(f"locked analysis denied: evaluator artifact audit does not match frozen binding for {key}")
    return digest


def _read_verified_evaluator_document(path: Path, *, expected_sha256: str) -> tuple[Any, str]:
    """Parse only after exact on-disk bytes match the post-evaluation audit."""
    return _read_regular_json_no_follow(path, expected_sha256=expected_sha256)


def analyze_locked_files(*, manifest_path: str | Path, evaluation_files: Mapping[str, str | Path],
                         authorization_path: str | Path | None, authorization_root_path: str | Path | None,
                         access_audit_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Authenticate/audit before opening locked evaluator documents.

    The current trust-root allowlist is intentionally empty and historical v1
    freezes do not bind this analysis set/selection, so this wrapper currently
    fails closed. It is an integration skeleton for a separately reviewed
    protocol version; it must not be weakened to accept caller-supplied pins.
    """
    manifest_file = Path(manifest_path)
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise ValueError("locked analysis denied: preregistration manifest must be a regular file")
    manifest, _ = _read_regular_json_no_follow(manifest_file, max_bytes=16 * 1024 * 1024)
    from validation.evaluate_torch_mvp1 import append_locked_access_audit, verify_locked_authorization
    authorization = verify_locked_authorization(manifest, authorization_path=authorization_path,
                                                trusted_root_path=authorization_root_path)
    # These release-specific checks happen after signature verification and
    # before any evaluation document is opened.
    freeze = manifest.get("pre_outcome_freeze", {})
    frozen_analysis = freeze.get("analysis_sha256", {})
    if set(frozen_analysis) != set(LOCKED_ANALYSIS_FILES):
        raise ValueError("locked analysis denied: freeze does not bind the cohort analyzer/statistics code")
    from validation.preregister_torch_mvp1 import ROOT, sha as file_sha
    actual_analysis = {name: file_sha(ROOT / name) for name in LOCKED_ANALYSIS_FILES}
    if frozen_analysis != actual_analysis:
        raise ValueError("locked analysis denied: cohort analysis source differs from freeze")
    selection = freeze.get("validation_selection")
    if (not isinstance(selection, dict) or selection.get("algorithm") not in ARMS
            or not isinstance(selection.get("sha256"), str) or len(selection["sha256"]) != 64):
        raise ValueError("locked analysis denied: validation-only cognition selection is not frozen")
    analysis_config = freeze.get("analysis_config")
    if (not isinstance(analysis_config, dict) or set(analysis_config) != ANALYSIS_CONFIG_KEYS
            or analysis_config.get("bootstrap_method") != "paired two-way crossed checkpoint × world resampling"
            or type(analysis_config.get("bootstrap_resamples")) is not int
            or analysis_config["bootstrap_resamples"] < 10_000
            or type(analysis_config.get("bootstrap_seed")) is not int):
        raise ValueError("locked analysis denied: exact bootstrap method/count/seed are not frozen")
    seeds = manifest.get("seed_splits", {}).get("training_root_seeds", [])
    expected_names = {f"{arm}:{seed_id}" for arm in ARMS for seed_id in seeds}
    if set(evaluation_files) != expected_names:
        raise ValueError("locked analysis denied: every preregistered arm/checkpoint evaluation is required")
    # Validate path identity and regular-file status without opening outcome data.
    for name, raw_path in evaluation_files.items():
        candidate = Path(raw_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"locked analysis denied: evaluation document is not a regular file: {name}")
    audit_path = Path(access_audit_path)
    # append_locked_access_audit validates the prior chain, appends and fsyncs
    # this record. This is deliberately before the first evaluation-file read.
    audit_sha = append_locked_access_audit(audit_path, {
        "schema_version": 1, "request_identity": "cohort-" + authorization["authorization_id"],
        "action": "locked_cohort_analysis_access",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "trusted_root_sha256": authorization["trusted_root_sha256"],
        "preregistration_manifest_sha256": manifest["manifest_sha256"],
        "release_binding_sha256": hashlib.sha256(json.dumps({"source_sha256": freeze.get("source_sha256"),
            "analysis_sha256": frozen_analysis, "checkpoint_sha256": freeze.get("checkpoint_sha256"),
            "selection": selection}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "cohort_document_keys": sorted(evaluation_files),
    })
    audit_records = _read_audit_chain(audit_path)
    evaluation_audits = {}
    artifact_audits = {}
    for record in audit_records:
        if record.get("action") == "locked_test_evaluation_access":
            key = record.get("selected_checkpoint_key")
            if key in expected_names:
                if key in evaluation_audits:
                    raise ValueError("locked analysis denied: duplicate per-checkpoint locked evaluation audit")
                if (record.get("authorization_id") != authorization["authorization_id"]
                        or record.get("authorization_sha256") != authorization["authorization_sha256"]
                        or record.get("preregistration_manifest_sha256") != manifest["manifest_sha256"]
                        or record.get("selected_checkpoint_sha256") != freeze["checkpoint_sha256"].get(key)
                        or record.get("selected_checkpoint_manifest_sha256") != freeze["checkpoint_manifest_sha256"].get(key)
                        or record.get("analysis_sha256") != frozen_analysis):
                    raise ValueError("locked analysis denied: checkpoint audit record does not match the frozen release")
                evaluation_audits[key] = record.get("record_sha256")
        elif record.get("action") == "locked_test_evaluation_artifact_committed":
            key = record.get("selected_checkpoint_key")
            if key in expected_names:
                if key in artifact_audits:
                    raise ValueError("locked analysis denied: duplicate evaluator artifact audit")
                expected_artifact_sha = _verify_artifact_audit_record(
                    record, key=key, authorization=authorization,
                    manifest_sha256=manifest["manifest_sha256"], freeze=freeze,
                    access_record_sha256=evaluation_audits.get(key),
                    artifact_name=Path(evaluation_files[key]).name)
                if record["evaluation_artifact_sha256"] != expected_artifact_sha:
                    raise AssertionError("validated artifact digest changed")
                artifact_audits[key] = record
    if set(evaluation_audits) != expected_names:
        raise ValueError("locked analysis denied: prior per-checkpoint evaluation audit is incomplete")
    if set(artifact_audits) != expected_names:
        raise ValueError("locked analysis denied: per-checkpoint evaluator artifact hash audit is incomplete")
    documents = []
    document_hashes = {}
    for key in sorted(evaluation_files):
        path = Path(evaluation_files[key])
        if path.is_symlink():
            raise ValueError("locked analysis denied: evaluation document changed to symlink after audit")
        expected_artifact_sha = _verify_artifact_audit_record(
            artifact_audits[key], key=key, authorization=authorization,
            manifest_sha256=manifest["manifest_sha256"], freeze=freeze,
            access_record_sha256=evaluation_audits[key], artifact_name=path.name)
        document, content_sha = _read_verified_evaluator_document(path, expected_sha256=expected_artifact_sha)
        documents.append(document)
        document_hashes[key] = content_sha
    expected_provenance = {
        "source_sha256": freeze.get("source_sha256"), "analysis_sha256": frozen_analysis,
        "checkpoints": {key: {"checkpoint_key": key,
                              "checkpoint_sha256": freeze["checkpoint_sha256"][key],
                              "checkpoint_manifest_sha256": freeze["checkpoint_manifest_sha256"][key]}
                        for key in expected_names},
        "audit_records": {key: evaluation_audits[key] for key in expected_names if key in evaluation_audits},
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "trusted_root_sha256": authorization["trusted_root_sha256"],
        "validation_selection": selection,
    }
    report = analyze_locked_cohort(
        documents, expected_manifest_sha256=manifest["manifest_sha256"],
        expected_provenance=expected_provenance,
        training_seeds=seeds, worlds_by_family=manifest["seed_splits"]["locked_test"],
        delta0_by_family=manifest["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"],
        selected_cognition_arm=selection["algorithm"], validation_selection_sha256=selection["sha256"],
        bootstrap_resamples=analysis_config["bootstrap_resamples"], seed=analysis_config["bootstrap_seed"])
    report["input_payload_values_analyzed"] = True
    report["locked_payloads_verified"] = True
    report["evaluation_artifact_sha256"] = document_hashes
    report["bootstrap"]["method"] = analysis_config["bootstrap_method"]
    report["access_audit_record_sha256"] = audit_sha
    target = Path(output_path)
    if target.is_symlink():
        raise ValueError("locked analysis denied: report output path is a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
