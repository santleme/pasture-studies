from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from validation.evaluate_torch_mvp1 import FAMILIES, evaluation_world_schedule
from validation.preregister_torch_mvp1 import (acceptance_spec, build_manifest,
    canonical_sha, finalize_manifest, historical_seed_ids, verify_manifest, FAMILIES, TRAIN_SEEDS,
    SOURCE_PATHS, TRAINING_SOURCE_PATHS, ROOT, sha)
import hashlib


def checkpoint_spec(root: Path, algorithm: str, seed: int) -> str:
    generation = root / algorithm / f"seed-{seed}" / "generation-00000001"
    generation.mkdir(parents=True)
    state = generation / "training.pt"
    state.write_bytes(f"checkpoint:{algorithm}:{seed}".encode())
    manifest = {"format_version": 1, "algorithm": algorithm,
                "config": {"seed": seed,
                           "source_hashes": {name: sha(ROOT / name) for name in TRAINING_SOURCE_PATHS}},
                "state_file": "training.pt",
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(), "generation": 1}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    return f"{algorithm}:{seed}={state}"


def analysis_specs() -> list[str]:
    from validation.preregister_torch_mvp1 import ROOT
    return [f"{name}={ROOT / name}" for name in
            ("validation/evaluate_torch_mvp1.py", "validation/benchmark_torch_fixed_corpus.py")]


class TorchPreregistrationTests(unittest.TestCase):
    def test_manifest_freezes_disjoint_train_validation_pilot_locked_and_reserve_seeds(self):
        manifest = build_manifest()
        verify_manifest(manifest)
        splits = manifest["seed_splits"]
        self.assertEqual(len(splits["training_root_seeds"]), 10)
        self.assertEqual(len(splits["validation"]["starter"]), 30)
        self.assertEqual(len(splits["precision_pilot"]["starter"]), 60)
        self.assertEqual(len(splits["locked_test"]["unseen_corridor"]), 30)
        self.assertEqual(len(splits["precision_reserve"]["starter"]), 30)
        self.assertTrue(manifest["seed_overlap_audit"]["disjoint"])
        self.assertEqual(manifest["seed_overlap_audit"]["overlap_count"], 0)
        all_ids = set(splits["training_root_seeds"])
        for key in ("validation", "precision_pilot", "locked_test", "precision_reserve"):
            for family in FAMILIES:
                values = set(splits[key][family])
                self.assertFalse(all_ids & values)
                all_ids |= values
        self.assertFalse(all_ids & historical_seed_ids())

    def test_manifest_copies_approved_thresholds_and_keeps_delta0_inconclusive(self):
        spec = acceptance_spec()
        self.assertEqual(spec["hardware_fixed_corpus"]["gpu_compute_speedup_ci95_lower_vs_best_cpu"], 2.0)
        self.assertEqual(spec["hardware_fixed_corpus"]["fixed_work_elapsed_reduction_min_fraction"], .30)
        self.assertEqual(spec["hardware_fixed_corpus"]["t4x2_speedup_ci95_lower_vs_t4x1"], 1.20)
        self.assertEqual(spec["cognition"]["minimum_each_primary_effect_absolute_probability"], .05)
        self.assertIsNone(spec["training_validity"]["near_zero_margin_delta0_by_family"])
        self.assertIn("INCONCLUSIVE", spec["training_validity"]["delta0_status"])

    def test_provenance_binds_profiler_and_vectorization_tests_but_not_generated_profile(self):
        self.assertIn("validation/profile_training_pipeline.py", SOURCE_PATHS)
        self.assertIn("validation/profile_torch_training.py", SOURCE_PATHS)
        self.assertIn("tests/test_profile_training_pipeline.py", SOURCE_PATHS)
        self.assertIn("tests/test_vectorized_world_batch.py", SOURCE_PATHS)
        self.assertNotIn("validation/PROFILE_TORCH_PIPELINE_LOCAL.json", SOURCE_PATHS)
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(all((root / path).is_file() for path in SOURCE_PATHS))

    def test_manifest_generation_succeeds_without_access_to_ignored_omx_files(self):
        original_open = Path.open
        original_is_file = Path.is_file

        def reject_omx(path, *args, **kwargs):
            if ".omx" in path.parts:
                raise AssertionError(f"runtime unexpectedly opened ignored approval artifact: {path}")
            return original_open(path, *args, **kwargs)

        def absent_omx(path, *args, **kwargs):
            if ".omx" in path.parts:
                return False
            return original_is_file(path, *args, **kwargs)

        with patch.object(Path, "open", reject_omx), patch.object(Path, "is_file", absent_omx):
            manifest = build_manifest()
            verify_manifest(manifest)
        self.assertIn("approved_artifacts", json.loads((ROOT / "validation/kaggle-mvp1/approved-plan-hashes.json").read_text()))

    def test_recomputed_manifest_hash_cannot_hide_train_world_seed_collision(self):
        manifest = build_manifest()
        collision = manifest["seed_splits"]["locked_test"]["starter"][0]
        manifest["seed_splits"]["training_world_seeds_by_root"]["71001"][0] = collision
        manifest["manifest_sha256"] = canonical_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "training world seeds do not match|disjoint"):
            verify_manifest(manifest)

    def test_evaluator_consumes_manifest_validation_seeds_and_blocks_locked_test_until_freeze(self):
        manifest = build_manifest()
        scheduled = evaluation_world_schedule(eval_seed=None, episodes_per_family=3,
                                              seed_manifest=manifest, split_name="validation")
        self.assertEqual(len(scheduled), 30 * len(FAMILIES))
        self.assertEqual(scheduled[:30], [("starter", x) for x in manifest["seed_splits"]["validation"]["starter"]])
        with self.assertRaisesRegex(ValueError, "locked evaluation prohibited"):
            evaluation_world_schedule(eval_seed=None, episodes_per_family=3,
                                      seed_manifest=manifest, split_name="locked_test")
        with self.assertRaisesRegex(ValueError, "requires the pre-outcome"):
            evaluation_world_schedule(eval_seed=10, episodes_per_family=1,
                                      seed_manifest=None, split_name="locked_test")

    def test_pending_manifest_can_be_safely_finalized_from_masked_pilot_and_frozen_artifacts(self):
        pending = build_manifest()
        seed_hash = canonical_sha(pending["seed_splits"]["precision_pilot"])
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"],
                 "pilot_seed_ids_sha256": seed_hash,
                 "sd_untrained_returns_by_family": {family: 2.0 + i for i, family in enumerate(FAMILIES)}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint_specs = []
            for algorithm in ("reinforce", "a2c"):
                for seed in TRAIN_SEEDS:
                    checkpoint_specs.append(checkpoint_spec(root, algorithm, seed))
            locked = finalize_manifest(pending, pilot, checkpoint_artifacts=checkpoint_specs,
                                       analysis_artifacts=analysis_specs())

        self.assertEqual(pending["status"], "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0")
        self.assertIsNone(pending["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"])
        self.assertEqual(locked["status"], "LOCKED_PRE_OUTCOME")
        self.assertAlmostEqual(locked["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"]["starter"], .1)
        self.assertEqual(len(locked["pre_outcome_freeze"]["checkpoint_sha256"]), 20)
        self.assertFalse(locked["pre_outcome_freeze"]["locked_outcomes_read"])
        verify_manifest(locked)

    def test_lock_rejects_unmasked_outcomes_invalid_pilot_and_incomplete_artifact_freeze(self):
        pending = build_manifest()
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"],
                 "pilot_seed_ids_sha256": canonical_sha(pending["seed_splits"]["precision_pilot"]),
                 "sd_untrained_returns_by_family": {family: 1.0 for family in FAMILIES}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = analysis_specs()
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                finalize_manifest(pending, {**pilot, "mean_rewards": [1]},
                                  checkpoint_artifacts=[], analysis_artifacts=analysis)
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                finalize_manifest(pending, {**pilot, "sd_untrained_returns_by_family": {**pilot["sd_untrained_returns_by_family"], "starter": 0}},
                                  checkpoint_artifacts=[], analysis_artifacts=analysis)
            with self.assertRaisesRegex(ValueError, "every selected"):
                finalize_manifest(pending, pilot, checkpoint_artifacts=[],
                                  analysis_artifacts=analysis)

    def test_locked_manifest_rejects_delta_mutation_or_missing_freeze_binding(self):
        pending = build_manifest()
        seeds = canonical_sha(pending["seed_splits"]["precision_pilot"])
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"], "pilot_seed_ids_sha256": seeds,
                 "sd_untrained_returns_by_family": {family: 1.0 for family in FAMILIES}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cp = []
            for alg in ("reinforce", "a2c"):
                for seed in TRAIN_SEEDS: cp.append(checkpoint_spec(root, alg, seed))
            locked = finalize_manifest(pending, pilot, checkpoint_artifacts=cp, analysis_artifacts=analysis_specs())
        tampered = {**locked, "acceptance_thresholds": json.loads(json.dumps(locked["acceptance_thresholds"]))}
        tampered["acceptance_thresholds"]["training_validity"]["near_zero_margin_delta0_by_family"]["starter"] = 99
        tampered["manifest_sha256"] = canonical_sha({k: v for k, v in tampered.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "checksum|delta0"):
            verify_manifest(tampered)
        no_binding = json.loads(json.dumps(locked))
        no_binding["pre_outcome_freeze"]["analysis_sha256"].pop("validation/evaluate_torch_mvp1.py")
        no_binding["manifest_sha256"] = canonical_sha({k: v for k, v in no_binding.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "exactly evaluator"):
            verify_manifest(no_binding)
        no_generation_binding = json.loads(json.dumps(locked))
        no_generation_binding["pre_outcome_freeze"]["checkpoint_manifest_sha256"].pop("a2c:71001")
        no_generation_binding["manifest_sha256"] = canonical_sha({k: v for k, v in no_generation_binding.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "generation manifest"):
            verify_manifest(no_generation_binding)

    def test_freeze_rejects_wrong_checkpoint_identity_and_substituted_analysis_file(self):
        pending = build_manifest()
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"],
                 "pilot_seed_ids_sha256": canonical_sha(pending["seed_splits"]["precision_pilot"]),
                 "sd_untrained_returns_by_family": {family: 1.0 for family in FAMILIES}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = [checkpoint_spec(root, alg, seed) for alg in ("reinforce", "a2c") for seed in TRAIN_SEEDS]
            # Substitute a checkpoint generation carrying the wrong algorithm identity.
            a2c_manifest_path = Path(specs[10].split("=", 1)[1]).parent / "manifest.json"
            wrong_manifest = json.loads(a2c_manifest_path.read_text())
            wrong_manifest["algorithm"] = "reinforce"
            a2c_manifest_path.write_text(json.dumps(wrong_manifest))
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                finalize_manifest(pending, pilot, checkpoint_artifacts=specs, analysis_artifacts=analysis_specs())

            root2 = root / "second"
            specs = [checkpoint_spec(root2, alg, seed) for alg in ("reinforce", "a2c") for seed in TRAIN_SEEDS]
            substitute = root / "fake-evaluator.py"
            substitute.write_text("# substituted analysis\n")
            bad_analysis = analysis_specs()
            bad_analysis[0] = f"validation/evaluate_torch_mvp1.py={substitute}"
            with self.assertRaisesRegex(ValueError, "actual current evaluator"):
                finalize_manifest(pending, pilot, checkpoint_artifacts=specs, analysis_artifacts=bad_analysis)

    def test_freeze_rejects_checkpoint_trained_from_different_source_hashes(self):
        pending = build_manifest()
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"],
                 "pilot_seed_ids_sha256": canonical_sha(pending["seed_splits"]["precision_pilot"]),
                 "sd_untrained_returns_by_family": {family: 1.0 for family in FAMILIES}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = [checkpoint_spec(root, alg, seed) for alg in ("reinforce", "a2c") for seed in TRAIN_SEEDS]
            manifest_path = Path(specs[0].split("=", 1)[1]).parent / "manifest.json"
            generation = json.loads(manifest_path.read_text())
            source_hashes = generation["config"]["source_hashes"]
            source_hashes[TRAINING_SOURCE_PATHS[0]] = "f" * 64
            manifest_path.write_text(json.dumps(generation))
            with self.assertRaisesRegex(ValueError, "training source hashes do not match"):
                finalize_manifest(pending, pilot, checkpoint_artifacts=specs, analysis_artifacts=analysis_specs())

    def test_freeze_rejects_checkpoint_trained_from_different_source_hashes(self):
        pending = build_manifest()
        pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                 "manifest_sha256": pending["manifest_sha256"],
                 "pilot_seed_ids_sha256": canonical_sha(pending["seed_splits"]["precision_pilot"]),
                 "sd_untrained_returns_by_family": {family: 1.0 for family in FAMILIES}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = [checkpoint_spec(root, alg, seed) for alg in ("reinforce", "a2c") for seed in TRAIN_SEEDS]
            manifest_path = Path(specs[0].split("=", 1)[1]).parent / "manifest.json"
            generation = json.loads(manifest_path.read_text())
            source_hashes = generation["config"]["source_hashes"]
            source_hashes[TRAINING_SOURCE_PATHS[0]] = "f" * 64
            manifest_path.write_text(json.dumps(generation))
            with self.assertRaisesRegex(ValueError, "training source hashes do not match"):
                finalize_manifest(pending, pilot, checkpoint_artifacts=specs, analysis_artifacts=analysis_specs())


if __name__ == "__main__":
    unittest.main()
