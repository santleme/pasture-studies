#!/usr/bin/env python3
"""Held-out evaluation and paired cognition probes for optional Torch actors.

All outcomes use the repository's synthetic simulator/reward. This script does
not establish cattle biology or husbandry validity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import base64
import binascii
import datetime
import fcntl
import stat
import subprocess
import sys
import uuid
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from farm_sim.model import (Action, AnimalMemory, FEATURE_NAMES, MemoryEvent,
                            Position, ResourcePatch, ToyPhysiology)
from farm_sim.policy import HeuristicBaseline, load_checkpoint
import farm_sim.simulation as simulation_module
from farm_sim.simulation import Simulation, make_world
from farm_sim.perception import encode_features, perceive
from farm_sim.memory import retrieve
from farm_sim.torch_checkpoint import load_generation, resolve_generation
from farm_sim.torch_training import ACTIONS, ActorCritic, derive_seed, require_torch, set_deterministic
from farm_sim.execution_policy import require_kaggle_for_full_workload

TICKS = 72
FAMILIES = ("starter", "resource_scarce_relocated", "wide", "unseen_corridor")
REWARD_COMPONENTS = ("forage_success", "forage_depleted", "water_success", "water_blocked",
                     "rested", "ruminated", "herd_contact", "herd_alone", "walked",
                     "grazing_en_route", "drinking_en_route")
# Deliberately empty until an independent governance process pins an approved
# Ed25519 trust root. A caller-supplied PEM is only a candidate, never trust by
# itself. Do not populate this from a CLI flag, environment variable, or the
# authorization artifact.
TRUSTED_AUTHORIZATION_ROOT_SHA256: frozenset[str] = frozenset()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def make_scenario(family: str, seed: int):
    """Build predeclared synthetic scenario families; rewards are untouched."""
    if family == "starter":
        return make_world(seed)
    if family == "wide":
        return make_world(seed, width=20, height=14)
    if family == "resource_scarce_relocated":
        world = make_world(seed)
        return replace(world,
            forage=(ResourcePatch(Position(1, 8), True), ResourcePatch(Position(2, 2), False),
                     ResourcePatch(Position(3, 7), False)),
            water=(ResourcePatch(Position(12, 8), True),))
    if family == "unseen_corridor":
        # Deliberately new synthetic topology: a divider with two passages.
        # It is a software stress family, not a biological environment.
        world = make_world(seed, width=18, height=12)
        divider = {Position(9, y) for y in range(1, 11) if y not in {3, 8}}
        base_obstacles = set(world.obstacles) - {Position(9, 3), Position(9, 7)}
        obstacles = frozenset(base_obstacles | divider)
        free_cells = [Position(x, y) for y in range(1, 11) for x in range(1, 17)
                      if Position(x, y) not in obstacles]
        animals = list(world.animals)
        occupied = set()
        for idx, animal in enumerate(animals):
            if animal.position in obstacles or animal.position in occupied:
                destination = next(cell for cell in free_cells if cell not in occupied)
                animals[idx] = replace(animal, position=destination)
            occupied.add(animals[idx].position)
        forage = (ResourcePatch(Position(3, 3), True), ResourcePatch(Position(4, 8), True),
                  ResourcePatch(Position(15, 3), True))
        water = (ResourcePatch(Position(15, 9), True),)
        return replace(world, animals=tuple(animals), obstacles=obstacles, forage=forage, water=water)
    raise ValueError(f"unknown scenario family {family!r}; expected one of {FAMILIES}")


def _valid(action, probabilities) -> bool:
    return (action in Action and len(probabilities) == len(Action)
            and all(math.isfinite(float(p)) and 0 <= float(p) <= 1 for p in probabilities)
            and abs(sum(float(p) for p in probabilities) - 1.0) < 1e-6)


def evaluate_episode(policy, family: str, seed: int) -> dict[str, Any]:
    sim = Simulation(make_scenario(family, seed), policy)
    action_counts = Counter({a.value: 0 for a in Action})
    probabilities: list[np.ndarray] = []
    invalid = 0
    component_rewards = Counter({name: 0.0 for name in REWARD_COMPONENTS})
    component_counts = Counter({name: 0 for name in REWARD_COMPONENTS})
    reasons = Counter()
    subreasons = Counter()
    reward_trajectory: list[float] = []
    original_navigate = simulation_module.navigate

    def counted_navigate(*args, **kwargs):
        result = original_navigate(*args, **kwargs)
        reasons[str(result.reason)] += 1
        subreasons[str(result.subreason)] += 1
        return result

    simulation_module.navigate = counted_navigate
    try:
        for _ in range(TICKS):
            result = sim.step()
            reward_trajectory.append(float(sum(a.last_reward for a in result.world.animals)))
            for animal in result.world.animals:
                action_counts[animal.action.value] += 1
                probs = np.asarray(animal.last_probabilities, dtype=np.float64)
                if not _valid(animal.action, probs):
                    invalid += 1
                else:
                    probabilities.append(probs)
            for event in result.events:
                if event.kind in component_counts:
                    component_counts[event.kind] += 1
                    component_rewards[event.kind] += float(event.reward)
    finally:
        simulation_module.navigate = original_navigate
    probs = np.asarray(probabilities, dtype=np.float64).reshape((-1, len(Action))) if probabilities else np.zeros((0, len(Action)))
    n = TICKS * len(sim.world.animals)
    mean_probs = probs.mean(axis=0) if len(probs) else np.zeros(len(Action))
    entropy = -np.sum(np.where(mean_probs > 0, mean_probs * np.log(np.maximum(mean_probs, 1e-300)), 0.0))
    per_decision_entropy = -np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-300)), 0.0), axis=1) if len(probs) else np.array([])
    final = sim.world
    return {
        "scenario": family, "world_seed": int(seed), "ticks": TICKS,
        "total_reward": float(sum(reward_trajectory)),
        "reward_trajectory_per_tick_all_animals": reward_trajectory,
        "reward_components": {k: {"count": int(component_counts[k]), "labeled_reward_sum": float(component_rewards[k])} for k in REWARD_COMPONENTS},
        "action_counts": {a.value: int(action_counts[a.value]) for a in Action},
        "action_frequencies": {a.value: float(action_counts[a.value] / n) for a in Action},
        "mean_action_probabilities": {a.value: float(mean_probs[i]) for i, a in enumerate(Action)},
        "mean_policy_entropy_nats": float(per_decision_entropy.mean()) if len(per_decision_entropy) else 0.0,
        "entropy_of_mean_action_distribution_nats": float(entropy),
        "invalid_actions": int(invalid),
        "navigation_reason_counts": dict(sorted(reasons.items())),
        "navigation_subreason_counts": dict(sorted(subreasons.items())),
        "final_drive_means": {key: float(np.mean([getattr(a.physiology, key) for a in final.animals])) for key in ("hunger", "thirst", "fatigue", "social_need")},
        "final_drives_by_animal": {a.animal_id: {key: float(getattr(a.physiology, key)) for key in ("hunger", "thirst", "fatigue", "social_need")} for a in final.animals},
        "resource_state": {"forage_available": int(sum(p.available for p in final.forage)), "forage_total": len(final.forage), "water_available": int(sum(p.available for p in final.water)), "water_total": len(final.water)},
        "final_positions": {a.animal_id: [a.position.x, a.position.y] for a in final.animals},
        "reward_total_note": "summed from per-animal synthetic step rewards once per tick; component label sums are separately descriptive",
    }


def _initial_water_fixture(seed: int = 83831):
    """Actual encoder-path fixture with high thirst and visible available water."""
    world = make_world(seed)
    animals = list(world.animals)
    cow = animals[0]
    cow = replace(cow, position=Position(2, 2), physiology=ToyPhysiology(.35, .85, .25, .45),
                  memory=AnimalMemory((MemoryEvent(0, "water_access", True, Position(4, 2)),)))
    animals[0] = cow
    return replace(world, animals=tuple(animals), tick=1,
                   water=(ResourcePatch(Position(4, 2), True),),
                   forage=(ResourcePatch(Position(3, 2), True),))


def cognition_probe(model, fixture_seed: int = 83831) -> dict[str, Any]:
    """Paired first-decision probabilities with fidelity hashes and exact shams."""
    water_success = FEATURE_NAMES.index("mem_water_success")
    water_blocked = FEATURE_NAMES.index("mem_water_blocked")
    water_available = FEATURE_NAMES.index("water_available")
    thirst_index = FEATURE_NAMES.index("thirst")
    world = _initial_water_fixture(fixture_seed)
    animal = world.animals[0]

    def vector(w):
        a = next(a for a in w.animals if a.animal_id == animal.animal_id)
        return np.asarray(encode_features(perceive(w, a), retrieve(a.memory, w.tick), a, w.width, w.height), dtype=np.float32)

    success = vector(world)
    blocked_animals = list(world.animals)
    a = blocked_animals[0]
    blocked_events = tuple(replace(e, success=False) if e.kind == "water_access" else e for e in a.memory.events)
    blocked_animals[0] = replace(a, memory=AnimalMemory(blocked_events))
    blocked_world = replace(world, animals=tuple(blocked_animals))
    blocked = vector(blocked_world)
    # Feature changes from the actual encoder are asserted to be confined to the
    # corresponding water-memory channels; causality is input-level only.
    memory_changed = [FEATURE_NAMES[i] for i, (x, y) in enumerate(zip(blocked, success)) if x != y]
    if set(memory_changed) - {"mem_water_success", "mem_water_blocked", "mem_event_recency"}:
        raise AssertionError(f"memory intervention changed unexpected features: {memory_changed}")
    if success[thirst_index] < 0.75 or success[water_available] != 1.0:
        raise AssertionError("water cognition fixture is not high-thirst with visible available water")

    available = success.copy()
    unavailable = success.copy()
    unavailable[water_available] = 0.0
    cases = {
        # Positive direction is always the preregistered cue-positive minus cue-negative.
        "memory_success": (blocked, success, "mem_water_success", "mem_water_blocked"),
        "perception_available": (unavailable, available, "water_available"),
    }
    device = next(model.parameters()).device
    torch = require_torch()

    def prob(row):
        with torch.no_grad():
            x = torch.as_tensor(row, dtype=torch.float32, device=device)
            return model.probabilities(x).detach().cpu().numpy().astype(np.float64)

    def digest(row):
        canonical = np.asarray(row, dtype="<f4").tobytes(order="C")
        return _sha_bytes(canonical)

    output = {}
    for name, values in cases.items():
        base, changed, *mask_features = values
        p_base = prob(base); p_changed = prob(changed)
        # Feature-off ablation clears the exact named channel(s) in both arms.
        base_off = base.copy(); changed_off = changed.copy()
        for feature in mask_features:
            idx = FEATURE_NAMES.index(feature)
            base_off[idx] = 0.0; changed_off[idx] = 0.0
        p_base_off = prob(base_off); p_changed_off = prob(changed_off)
        effect = float(p_changed[ACTIONS.index(Action.DRINK)] - p_base[ACTIONS.index(Action.DRINK)])
        off_effect = float(p_changed_off[ACTIONS.index(Action.DRINK)] - p_base_off[ACTIONS.index(Action.DRINK)])
        sham = prob(base.copy())
        output[name] = {
            "contrast": "changed_minus_baseline",
            "direction": "memory_success_minus_blocked" if name == "memory_success" else "water_available_feature_minus_unavailable_feature",
            "baseline_probabilities": {a.value: float(p_base[i]) for i, a in enumerate(ACTIONS)},
            "changed_probabilities": {a.value: float(p_changed[i]) for i, a in enumerate(ACTIONS)},
            "drink_probability_effect": effect,
            "required_effect_margin_descriptive_only": 0.05,
            "effect_meets_plus_0_05_margin": bool(effect >= 0.05),
            "ablation": {"features_zeroed": list(mask_features), "baseline_drink_probability": float(p_base_off[ACTIONS.index(Action.DRINK)]),
                         "changed_drink_probability": float(p_changed_off[ACTIONS.index(Action.DRINK)]),
                         "drink_probability_effect": off_effect, "attenuation": float(effect - off_effect),
                         "attenuation_fraction_of_full": float((effect - off_effect) / effect) if effect != 0 else None},
            "input_sha256": {"baseline": digest(base), "changed": digest(changed), "baseline_ablation": digest(base_off), "changed_ablation": digest(changed_off)},
            "changed_feature_names": [FEATURE_NAMES[i] for i, (x, y) in enumerate(zip(base, changed)) if x != y],
            "sham": {"input_bit_identical": bool(np.array_equal(base, base.copy())), "probabilities_bit_identical": bool(np.array_equal(p_base, sham)),
                     "input_sha256": digest(base), "repeat_input_sha256": digest(base.copy())},
            "fidelity": {"expected_features_changed": list(mask_features), "actual_features_changed": [FEATURE_NAMES[i] for i, (x, y) in enumerate(zip(base, changed)) if x != y]},
        }
    return {"scope": "paired first-decision policy probability intervention; not a full trajectory causal effect",
            "fixture": {"fixture_seed": int(fixture_seed), "thirst": float(success[thirst_index]), "water_visible": float(success[FEATURE_NAMES.index("water_visible")]),
                        "water_available": float(success[water_available]), "memory_encoder_changed": memory_changed},
            "feature_names": list(FEATURE_NAMES), "contrasts": output,
            "interpretation": "The +0.05 margin is reported as predeclared; these single-checkpoint effects are descriptive and do not satisfy the preregistered clustered/Holm acceptance gate."}


def load_checkpoint_actor(generation: str | Path, algorithm: str, seed: int, device: str):
    torch = require_torch()
    set_deterministic(seed, device)
    target = resolve_generation(generation)
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest.get("algorithm") != algorithm:
        raise ValueError(f"requested algorithm {algorithm!r} but checkpoint is {manifest.get('algorithm')!r}")
    config = manifest.get("config", {})
    if int(config.get("seed", -1)) != int(seed):
        raise ValueError(f"requested seed {seed} does not match checkpoint config seed {config.get('seed')}")
    model = ActorCritic(value_head=algorithm == "a2c", seed=seed, dtype=torch.float32).to(device)
    load_generation(target, model=model, map_location=device, expected_algorithm=algorithm)
    model.eval()
    return model, manifest, target


def evaluation_world_schedule(*, eval_seed: int | None, episodes_per_family: int,
                              seed_manifest: dict[str, Any] | None = None,
                              split_name: str = "validation") -> list[tuple[str, int]]:
    if split_name not in {"validation", "locked_test"}:
        raise ValueError("evaluator supports validation or locked_test; pilot data requires a masked precision-only workflow")
    if seed_manifest is not None:
        from validation.preregister_torch_mvp1 import verify_manifest
        # Validation is intentionally usable with the approved pending manifest
        # even after analysis code evolves. Locked runs below require exact live
        # source and analysis bindings before any locked world is instantiated.
        verify_manifest(seed_manifest, verify_current_sources=(split_name == "locked_test"))
        if split_name == "locked_test":
            status = seed_manifest.get("status")
            delta0 = seed_manifest["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"]
            if status != "LOCKED_PRE_OUTCOME" or not delta0:
                raise ValueError("locked evaluation prohibited until pilot delta0 and locked checkpoint/source analysis are frozen")
        field = "locked_test" if split_name == "locked_test" else "validation"
        return [(family, int(world_seed)) for family in FAMILIES
                for world_seed in seed_manifest["seed_splits"][field][family]]
    if split_name == "locked_test":
        raise ValueError("locked evaluation requires the pre-outcome seed/source manifest")
    if eval_seed is None:
        raise ValueError("eval_seed is required unless a pre-registered seed manifest is supplied")
    return [(family, derive_seed(eval_seed, "heldout", family, idx))
            for family in FAMILIES for idx in range(episodes_per_family)]


def verify_locked_checkpoint_binding(seed_manifest: dict[str, Any], *, generation_path: Path,
                                     generation_manifest: dict[str, Any], algorithm: str, seed: int) -> dict[str, Any]:
    """Fail closed unless this exact committed checkpoint was frozen pre-outcome."""
    from validation.preregister_torch_mvp1 import (TRAINING_SOURCE_PATHS, verify_manifest)
    verify_manifest(seed_manifest, verify_current_sources=True)
    if seed_manifest.get("status") != "LOCKED_PRE_OUTCOME":
        raise ValueError("locked checkpoint verification requires a LOCKED_PRE_OUTCOME manifest")
    key = f"{algorithm}:{seed}"
    expected = seed_manifest["pre_outcome_freeze"]["checkpoint_sha256"].get(key)
    expected_manifest = seed_manifest["pre_outcome_freeze"]["checkpoint_manifest_sha256"].get(key)
    if not isinstance(expected, str):
        raise ValueError(f"locked manifest has no selected checkpoint binding for {key}")
    if not isinstance(expected_manifest, str):
        raise ValueError(f"locked manifest has no selected generation-manifest binding for {key}")
    if generation_manifest.get("algorithm") != algorithm:
        raise ValueError("loaded checkpoint algorithm does not match the locked arm")
    config = generation_manifest.get("config", {})
    if not isinstance(config, dict) or config.get("seed") != seed:
        raise ValueError("loaded checkpoint training seed does not match the locked arm")
    expected_training_sources = {name: seed_manifest["source_sha256"][name]
                                 for name in TRAINING_SOURCE_PATHS}
    if config.get("source_hashes") != expected_training_sources:
        raise ValueError("loaded checkpoint training source hashes do not match the frozen pre-outcome source manifest")
    state_name = generation_manifest.get("state_file")
    if state_name != "training.pt":
        raise ValueError("loaded checkpoint state filename is invalid")
    actual = sha_file(generation_path / state_name)
    if actual != generation_manifest.get("state_sha256"):
        raise ValueError("loaded checkpoint state hash does not match its generation manifest")
    if actual != expected:
        raise ValueError(f"loaded checkpoint hash does not match the frozen selected artifact {key}")
    actual_generation_manifest = sha_file(generation_path / "manifest.json")
    if actual_generation_manifest != expected_manifest:
        raise ValueError(f"loaded checkpoint generation manifest does not match frozen selected artifact {key}")
    # Keep these comparisons explicit at the evaluation boundary: neither a
    # stale preregistration nor a substituted analysis file may reach a world
    # constructor, even if the lock verifier is later refactored.
    from validation.preregister_torch_mvp1 import ROOT as REPO_ROOT, sha as prereg_sha
    frozen_sources = seed_manifest["pre_outcome_freeze"]["source_sha256"]
    if frozen_sources != seed_manifest.get("source_sha256"):
        raise ValueError("locked source hashes do not match the pre-outcome source manifest")
    actual_sources = {name: prereg_sha(REPO_ROOT / name) for name in frozen_sources}
    if actual_sources != frozen_sources:
        raise ValueError("current training source hashes do not match the frozen pre-outcome source manifest")
    frozen_analysis = seed_manifest["pre_outcome_freeze"]["analysis_sha256"]
    actual_analysis = {name: prereg_sha(REPO_ROOT / name) for name in frozen_analysis}
    if actual_analysis != frozen_analysis:
        raise ValueError("current evaluator/fixed-corpus analysis hashes do not match the frozen pre-outcome analysis")
    return {"checkpoint_key": key, "checkpoint_sha256": actual,
            "checkpoint_manifest_sha256": actual_generation_manifest,
            "analysis_sha256": actual_analysis, "source_sha256": actual_sources}


def verify_locked_authorization(seed_manifest: dict[str, Any], *, authorization_path: str | Path | None,
                                trusted_root_path: str | Path | None) -> dict[str, Any]:
    """Verify separately signed permission to access locked outcomes.

    The authorization is intentionally detached from the preregistration. The
    public key must be explicitly supplied and its SHA-256 fingerprint must
    match the code-pinned allowlist. The allowlist is intentionally empty until
    an independent governance process configures it. This function reads only
    authorization and preregistration metadata, never locked outcome data.
    """
    if authorization_path is None or trusted_root_path is None:
        raise ValueError("locked evaluation denied: explicit signed authorization and trusted root are required")
    from validation.preregister_torch_mvp1 import verify_manifest
    verify_manifest(seed_manifest, verify_current_sources=True)
    if seed_manifest.get("status") != "LOCKED_PRE_OUTCOME":
        raise ValueError("locked evaluation denied: preregistration is not LOCKED_PRE_OUTCOME")
    root = Path(trusted_root_path)
    if not root.is_file():
        raise ValueError("locked evaluation denied: candidate authorization root is missing")
    root_fingerprint = sha_file(root)
    if root_fingerprint not in TRUSTED_AUTHORIZATION_ROOT_SHA256:
        raise ValueError("locked evaluation denied: candidate authorization root fingerprint is not allowlisted")
    try:
        doc = json.loads(Path(authorization_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("locked evaluation denied: authorization artifact cannot be read") from exc
    required = {"schema_version", "authorization_id", "action", "preregistration_manifest_sha256",
                "selected_checkpoint_sha256", "analysis_sha256", "signature_ed25519_base64"}
    if not isinstance(doc, dict) or set(doc) != required:
        raise ValueError("locked evaluation denied: authorization artifact has an invalid schema")
    if doc["schema_version"] != 1 or not isinstance(doc["authorization_id"], str) or not doc["authorization_id"].strip():
        raise ValueError("locked evaluation denied: authorization identity/schema is invalid")
    if doc["action"] != "evaluate_locked_test":
        raise ValueError("locked evaluation denied: authorization scope is not evaluate_locked_test")
    freeze = seed_manifest["pre_outcome_freeze"]
    if doc["preregistration_manifest_sha256"] != seed_manifest["manifest_sha256"]:
        raise ValueError("locked evaluation denied: authorization is bound to a different preregistration")
    if doc["selected_checkpoint_sha256"] != freeze["checkpoint_sha256"]:
        raise ValueError("locked evaluation denied: authorization selected-checkpoint hashes do not match freeze")
    if doc["analysis_sha256"] != freeze["analysis_sha256"]:
        raise ValueError("locked evaluation denied: authorization analysis hashes do not match freeze")
    try:
        signature = base64.b64decode(doc["signature_ed25519_base64"], validate=True)
    except (TypeError, binascii.Error) as exc:
        raise ValueError("locked evaluation denied: authorization signature encoding is invalid") from exc
    body = {key: value for key, value in doc.items() if key != "signature_ed25519_base64"}
    signed_bytes = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # OpenSSL's Ed25519 verifier is used as a system primitive, avoiding a
    # silently selected/bundled trust key or an additional Python crypto API.
    # pkeyutl accepts separate input and signature files, so use a private
    # temporary directory rather than shell paths or a combined stdin stream.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pasture-auth-") as temporary:
        message_path = Path(temporary) / "authorization.json"
        signature_path = Path(temporary) / "authorization.sig"
        message_path.write_bytes(signed_bytes)
        signature_path.write_bytes(signature)
        try:
            verified = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(root), "-rawin",
                 "-in", str(message_path), "-sigfile", str(signature_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
        except OSError as exc:
            raise ValueError("locked evaluation denied: OpenSSL Ed25519 verifier is unavailable") from exc
    if verified.returncode != 0:
        raise ValueError("locked evaluation denied: authorization signature is invalid for configured trusted root")
    return {"authorization_id": doc["authorization_id"], "action": doc["action"],
            "trusted_root_sha256": root_fingerprint, "authorization_sha256": sha_file(Path(authorization_path)),
            "preregistration_manifest_sha256": seed_manifest["manifest_sha256"]}


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha_bytes(encoded)


def append_locked_access_audit(path: str | Path, entry: dict[str, Any]) -> str:
    """Append and fsync one hash-chained locked-access audit record.

    A corrupted or non-regular existing log blocks evaluation. This is an
    append-only application protocol; operational storage must additionally
    protect the log from truncation/replacement.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("locked evaluation denied: audit log is not a regular file")
        if info.st_mode & 0o077:
            raise ValueError("locked evaluation denied: audit log permissions must be owner-only")
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        existing = bytearray()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            existing.extend(chunk)
        previous = "0" * 64
        if existing:
            if not existing.endswith(b"\n"):
                raise ValueError("locked evaluation denied: audit log has an incomplete trailing record")
            for line_number, raw_line in enumerate(existing.splitlines(), start=1):
                try:
                    prior = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"locked evaluation denied: audit log record {line_number} is invalid") from exc
                if not isinstance(prior, dict) or prior.get("previous_record_sha256") != previous:
                    raise ValueError(f"locked evaluation denied: audit log hash chain is broken at record {line_number}")
                claimed = prior.get("record_sha256")
                body = {key: value for key, value in prior.items() if key != "record_sha256"}
                if claimed != _canonical_json_sha256(body):
                    raise ValueError(f"locked evaluation denied: audit log record {line_number} hash mismatch")
                previous = claimed
        body = {**entry, "previous_record_sha256": previous}
        record = {**body, "record_sha256": _canonical_json_sha256(body)}
        line = (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        os.lseek(fd, 0, os.SEEK_END)
        written = os.write(fd, line)
        if written != len(line):
            raise OSError("short write while appending locked access audit record")
        os.fsync(fd)
        return record["record_sha256"]
    finally:
        os.close(fd)


def evaluate(generation: str | Path, *, algorithm: str, seed: int, eval_seed: int | None,
             output: str | Path, episodes_per_family: int = 3, device: str = "cpu",
             seed_manifest_path: str | Path | None = None, split_name: str = "validation",
             locked_authorization_path: str | Path | None = None,
             authorization_root_path: str | Path | None = None,
             access_audit_path: str | Path | None = None,
             request_identity: str | None = None) -> dict[str, Any]:
    if split_name == "locked_test" and access_audit_path is None:
        raise ValueError("locked evaluation denied: durable access audit path is required")
    torch = require_torch()
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else device
    model, manifest, generation_path = load_checkpoint_actor(generation, algorithm, seed, device)
    untrained = ActorCritic(value_head=algorithm == "a2c", seed=seed, dtype=torch.float32).to(device).eval()
    manifest_doc = None
    locked_binding = None
    if seed_manifest_path is not None:
        manifest_doc = json.loads(Path(seed_manifest_path).read_text())
        if int(seed) not in set(manifest_doc["seed_splits"]["training_root_seeds"]):
            raise ValueError("checkpoint seed is not in the preregistered training split")
        if split_name == "locked_test":
            # Authorization must pass before checkpoint loading can lead to
            # any schedule construction or outcome/world access.
            locked_binding = verify_locked_authorization(
                manifest_doc, authorization_path=locked_authorization_path,
                trusted_root_path=authorization_root_path)
            generation_manifest = json.loads((generation_path / "manifest.json").read_text())
            checkpoint_binding = verify_locked_checkpoint_binding(manifest_doc, generation_path=generation_path,
                generation_manifest=generation_manifest, algorithm=algorithm, seed=seed)
            locked_binding["checkpoint_binding"] = checkpoint_binding
            authorization_info = locked_binding
            prereg_freeze = manifest_doc["pre_outcome_freeze"]
            release_binding = {
                "source_sha256": prereg_freeze["source_sha256"],
                "analysis_sha256": prereg_freeze["analysis_sha256"],
                "checkpoint_sha256": prereg_freeze["checkpoint_sha256"],
                "checkpoint_manifest_sha256": prereg_freeze["checkpoint_manifest_sha256"],
                "preregistration_manifest_sha256": manifest_doc["manifest_sha256"],
            }
            audit_hash = append_locked_access_audit(access_audit_path, {
                "schema_version": 1,
                "request_identity": request_identity or uuid.uuid4().hex,
                "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "action": "locked_test_evaluation_access",
                "authorization_id": authorization_info["authorization_id"],
                "authorization_sha256": authorization_info["authorization_sha256"],
                "trusted_root_sha256": authorization_info["trusted_root_sha256"],
                "preregistration_manifest_sha256": manifest_doc["manifest_sha256"],
                "release_binding_sha256": _canonical_json_sha256(release_binding),
                "release_binding": release_binding,
                "selected_checkpoint_key": f"{algorithm}:{seed}",
                "selected_checkpoint_sha256": checkpoint_binding["checkpoint_sha256"],
                "selected_checkpoint_manifest_sha256": checkpoint_binding["checkpoint_manifest_sha256"],
                "analysis_sha256": checkpoint_binding["analysis_sha256"],
            })
            locked_binding["access_audit_record_sha256"] = audit_hash
    families = evaluation_world_schedule(eval_seed=eval_seed, episodes_per_family=episodes_per_family,
                                         seed_manifest=manifest_doc, split_name=split_name)
    train_worlds = {derive_seed(seed, "train-world", i) for i in range(int(manifest.get("config", {}).get("episodes", 0)))}
    eval_worlds = {world_seed for _, world_seed in families}
    if len(eval_worlds) != len(families) or train_worlds.intersection(eval_worlds):
        raise ValueError("evaluation seeds collide with each other or the checkpoint's dynamic-train split")
    src_names = ("validation/evaluate_torch_mvp1.py", "farm_sim/torch_training.py", "farm_sim/torch_checkpoint.py",
                 "farm_sim/simulation.py", "farm_sim/model.py", "farm_sim/navigation.py",
                 "farm_sim/perception.py", "farm_sim/memory.py", "farm_sim/physiology.py")
    sources = {name: sha_file(ROOT / name) for name in src_names}
    split_doc = ({"manifest_sha256": manifest_doc["manifest_sha256"],
                  "train_root_seed": seed,
                  "evaluation_split": split_name,
                  "validation_world_seeds": manifest_doc["seed_splits"]["validation"],
                  "precision_pilot_world_seeds": manifest_doc["seed_splits"]["precision_pilot"],
                  "locked_world_seeds": manifest_doc["seed_splits"]["locked_test"],
                  "reserve_world_seeds": manifest_doc["seed_splits"]["precision_reserve"],
                  "families": list(FAMILIES)} if manifest_doc else
                 {"train_root_seed": seed, "train_world_seed_derivation": "sha256(root,train-world,episode-index)",
                  "evaluation_root_seed": eval_seed, "evaluation_world_seed_derivation": "sha256(root,heldout,family,index)",
                  "families": list(FAMILIES)})
    split_doc["world_seeds"] = [{"family": f, "seed": s} for f, s in families]
    rows = []
    for family, world_seed in families:
        for name, policy in (("trained", model), ("untrained_same_seed", untrained)):
            result = evaluate_episode(policy, family, world_seed)
            result["policy"] = name
            rows.append(result)
        # Heuristic is deliberately an independently labeled benchmark arm.
        baseline = evaluate_episode(HeuristicBaseline(), family, world_seed)
        baseline["policy"] = "heuristic_baseline"
        rows.append(baseline)
    cognition_rows = []
    for family, world_seed in families:
        for policy_name, probe_model in (("trained", model), ("untrained_same_seed", untrained)):
            cognition_rows.append({"policy": policy_name, "scenario_family": family,
                                   "world_seed": int(world_seed),
                                   "probe": cognition_probe(probe_model, fixture_seed=world_seed)})
    doc = {"schema_version": 1, "title": "Pasture Studies Torch held-out evaluation",
           "algorithm": algorithm, "training_seed": seed, "generation": str(generation_path),
           "generation_manifest_sha256": sha_file(generation_path / "manifest.json"),
           "locked_pre_outcome_binding": locked_binding,
           "scenario_scope": "synthetic simulator stress/generalization cases; not biological validation",
           "ticks_per_day": 24, "days_per_episode": 3, "paired_world_seeds": True,
           "seed_split": {**split_doc, "sha256": _sha_bytes(json.dumps(split_doc, sort_keys=True).encode())},
           "training_validation_test_disjoint_check": {"training_vs_eval_collision_count": len(train_worlds.intersection(eval_worlds)),
                "all_registered_splits_pairwise_disjoint": manifest_doc is not None,
                "pass": True},
           "source_sha256": sources, "runs": rows,
           "cognition_influence": cognition_probe(model),
           "paired_cognition_probe_rows": cognition_rows,
           "split_limitation": ("Pre-registered train/validation/pilot/locked/reserve split manifest hash-verified; pilot-derived delta0 remains pending, so utility acceptance remains inconclusive." if manifest_doc else "No separate validation-world manifest supplied. Evaluator derives an independent held-out split and verifies it is disjoint from all train world IDs represented by the checkpoint episode count; this descriptive run is not eligible for locked acceptance."),
           "limitation": "This evaluator supports paired first-decision probabilities and descriptive held-out rollouts; it does not perform the preregistered multi-checkpoint clustered bootstrap/Holm locked cognition test."}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return {"output": str(output_path), "episodes": len(rows), "cognition_channels": list(doc["cognition_influence"]["contrasts"]),
            "report_sha256": sha_file(output_path), "gpu_device": str(device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True, help="committed generation dir or checkpoint root containing LATEST")
    parser.add_argument("--algorithm", required=True, choices=("reinforce", "a2c"))
    parser.add_argument("--seed", type=int, required=True, help="must match training checkpoint seed")
    parser.add_argument("--eval-seed", type=int, default=None, help="independent held-out split root seed (descriptive mode only)")
    parser.add_argument("--seed-manifest", type=Path, default=None,
                        help="pre-outcome preregistration JSON; uses its locked test seeds and verifies all bound hashes")
    parser.add_argument("--split", choices=("validation", "locked_test"), default="validation",
                        help="validation is for selection; locked_test additionally requires a separately signed authorization")
    parser.add_argument("--locked-authorization", type=Path, default=None,
                        help="separate signed authorization JSON required for locked_test")
    parser.add_argument("--authorization-root", type=Path, default=None,
                        help="candidate Ed25519 public-key PEM; accepted only when its SHA-256 fingerprint is code-allowlisted (currently none)")
    parser.add_argument("--access-audit", type=Path, default=None,
                        help="append-only, owner-only hash-chained audit JSONL path; required for locked_test")
    parser.add_argument("--request-identity", default=None,
                        help="optional caller correlation ID; generated if omitted")
    parser.add_argument("--episodes-per-family", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="explicit one-episode-per-family local validation smoke; never accesses locked test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.smoke:
        if args.episodes_per_family != 3 or args.split != "validation":
            parser.error("--smoke fixes one episode per family and permits only the validation split")
        args.episodes_per_family = 1
    else:
        try:
            require_kaggle_for_full_workload(
                "locked-test evaluation" if args.split == "locked_test" else "policy evaluation"
            )
        except RuntimeError as exc:
            parser.error(str(exc))
    if args.episodes_per_family <= 0:
        parser.error("--episodes-per-family must be positive")
    print(json.dumps(evaluate(args.generation, algorithm=args.algorithm, seed=args.seed,
                              eval_seed=args.eval_seed, output=args.output,
                              episodes_per_family=args.episodes_per_family, device=args.device,
                              seed_manifest_path=args.seed_manifest, split_name=args.split,
                              locked_authorization_path=args.locked_authorization,
                              authorization_root_path=args.authorization_root,
                              access_audit_path=args.access_audit,
                              request_identity=args.request_identity), indent=2))


if __name__ == "__main__":
    main()
