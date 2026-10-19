"""Synthetic-only tests for the preregistered Torch cohort analyzer."""
from __future__ import annotations

import unittest

from validation.analyze_torch_cohort import FAMILIES, analyze_locked_cohort, analyze_locked_files


def _documents():
    seeds = tuple(range(71001, 71011))
    worlds = {family: tuple(range(74001 + i * 100, 74031 + i * 100))
              for i, family in enumerate(FAMILIES)}
    docs = []
    source_hashes = {"train.py": "c" * 64}
    analysis_hashes = {"analysis.py": "d" * 64}
    checkpoint_bindings = {f"{arm}:{seed}": {"checkpoint_key": f"{arm}:{seed}",
                                               "checkpoint_sha256": f"{seed:064x}",
                                               "checkpoint_manifest_sha256": f"{seed+1:064x}"}
                          for arm in ("reinforce", "a2c") for seed in seeds}
    audit_records = {f"{arm}:{seed}": f"{seed+2:064x}" for arm in ("reinforce", "a2c") for seed in seeds}
    provenance = {"source_sha256": source_hashes, "analysis_sha256": analysis_hashes,
                  "checkpoints": checkpoint_bindings, "audit_records": audit_records,
                  "authorization_id": "synthetic-test-authorization",
                  "validation_selection": {"algorithm": "a2c", "sha256": "b" * 64}}
    for arm in ("reinforce", "a2c"):
        for seed in seeds:
            runs = []
            cognition = []
            for family in FAMILIES:
                for world in worlds[family]:
                    baseline = -0.5
                    trained = baseline + (0.3 if arm == "a2c" else 0.2)
                    for policy, value in (("trained", trained), ("untrained_same_seed", baseline),
                                          ("heuristic_baseline", -0.4)):
                        runs.append({"scenario": family, "world_seed": world, "policy": policy,
                                     "total_reward": value,
                                     "action_counts": {"WALK": 24, "GRAZE": 24, "DRINK": 24},
                                     "invalid_actions": 0,
                                     "mean_policy_entropy_nats": .7,
                                     "final_drive_means": {"hunger": .2, "thirst": .3},
                                     "navigation_reason_counts": {"reached": 2},
                                     "resource_state": {"water_available": 1},
                                     "reward_trajectory_per_tick_all_animals": [0.1] * 72})
                    cognition.append({"policy": "trained", "scenario_family": family,
                                      "world_seed": world,
                                      "probe": {"contrasts": {
                                          "memory_success": _contrast(.10, .06, ["mem_water_success", "mem_water_blocked"], memory=True),
                                          "perception_available": _contrast(.12, .07, ["water_available"]),
                                      }}})
            docs.append({"algorithm": arm, "training_seed": seed,
                         "locked_pre_outcome_binding": {
                             "preregistration_manifest_sha256": "a" * 64,
                             "authorization_id": provenance["authorization_id"],
                             "access_audit_record_sha256": audit_records[f"{arm}:{seed}"],
                             "checkpoint_binding": {**checkpoint_bindings[f"{arm}:{seed}"],
                                                     "source_sha256": source_hashes,
                                                     "analysis_sha256": analysis_hashes}},
                         "seed_split": {"evaluation_split": "locked_test"}, "runs": runs,
                         "paired_cognition_probe_rows": cognition})
    return docs, seeds, worlds, provenance


def _contrast(effect, attenuation, features, *, memory=False):
    digest = "e" * 64
    return {"drink_probability_effect": effect,
            "direction": "memory_success_minus_blocked" if memory else "water_available_feature_minus_unavailable_feature",
            "changed_feature_names": features,
            "input_sha256": {"baseline": digest, "changed": "f" * 64,
                             "baseline_ablation": "1" * 64, "changed_ablation": "2" * 64},
            "ablation": {"attenuation": attenuation},
            "fidelity": {"expected_features_changed": features,
                         "actual_features_changed": features},
            "sham": {"input_bit_identical": True, "probabilities_bit_identical": True,
                     "input_sha256": digest, "repeat_input_sha256": digest}}


class TorchCohortAnalysisTests(unittest.TestCase):
    def analyze(self, docs, resamples=200):
        _, seeds, worlds, provenance = _documents()
        return analyze_locked_cohort(docs, expected_manifest_sha256="a" * 64,
                                     expected_provenance=provenance,
                                     training_seeds=seeds, worlds_by_family=worlds,
                                     delta0_by_family={f: .1 for f in FAMILIES},
                                     selected_cognition_arm="a2c",
                                     validation_selection_sha256="b" * 64,
                                     bootstrap_resamples=resamples, seed=19)

    def test_computes_exact_nine_and_four_confirmatory_families(self):
        docs, _, _, _ = _documents()
        result = self.analyze(docs)
        utility = result["utility_family"]
        cognition = result["cognition_family"]
        self.assertEqual(utility["family_size"], 9)
        self.assertEqual(len(utility["tests"]), 9)
        self.assertEqual(cognition["family_size"], 4)
        self.assertEqual(len(cognition["tests"]), 4)
        self.assertEqual(result["cognition_family"]["selected_arm"], "a2c")
        self.assertTrue(utility["overall_learned_utility_gate"])
        self.assertTrue(cognition["gate"])
        self.assertEqual(len(result["descriptive_checkpoint_family_diagnostics"]), 2 * 4 * 10)
        self.assertFalse(result["descriptive_checkpoint_family_diagnostics"][0]["single_action_only"])

    def test_rejects_missing_cell_wrong_manifest_sham_and_fidelity(self):
        docs, _, _, _ = _documents()
        with self.assertRaisesRegex(ValueError, "exactly every arm"):
            self.analyze(docs[:-1])
        docs, _, _, _ = _documents()
        docs[0]["locked_pre_outcome_binding"]["preregistration_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "different preregistration"):
            self.analyze(docs)
        docs, _, _, _ = _documents()
        docs[-1]["paired_cognition_probe_rows"][0]["probe"]["contrasts"]["memory_success"]["sham"]["probabilities_bit_identical"] = False
        with self.assertRaisesRegex(ValueError, "sham"):
            self.analyze(docs)

    def test_rejects_nonfinite_episode_outcome(self):
        docs, _, _, _ = _documents()
        docs[0]["runs"][0]["total_reward"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.analyze(docs)

    def test_rejects_mislabeled_intervention_and_coerced_seed(self):
        docs, _, _, _ = _documents()
        probe = docs[-1]["paired_cognition_probe_rows"][0]["probe"]
        probe["contrasts"]["memory_success"]["fidelity"]["expected_features_changed"] = ["thirst"]
        probe["contrasts"]["memory_success"]["fidelity"]["actual_features_changed"] = ["thirst"]
        with self.assertRaisesRegex(ValueError, "unexpected features"):
            self.analyze(docs)
        docs, _, _, _ = _documents()
        docs[0]["training_seed"] = float(docs[0]["training_seed"])
        with self.assertRaisesRegex(ValueError, "unexpected arm or training seed"):
            self.analyze(docs)

    def test_low_bootstrap_resolution_cannot_manufacture_holm_pass(self):
        docs, _, _, _ = _documents()
        result = self.analyze(docs, resamples=20)
        self.assertFalse(result["utility_family"]["overall_learned_utility_gate"])
        self.assertFalse(result["cognition_family"]["gate"])
        self.assertTrue(any(row["holm_lower_bound"] is None
                            for row in result["utility_family"]["tests"]))

    def test_locked_file_loader_denies_before_opening_outcome_documents(self):
        import tempfile
        from pathlib import Path
        from validation.preregister_torch_mvp1 import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            outcome = Path(tmp) / "locked-eval.json"
            outcome.write_text("this is deliberately not JSON")
            with self.assertRaisesRegex(ValueError, "explicit signed authorization"):
                analyze_locked_files(manifest_path=ROOT / "validation/kaggle-mvp1/preregistration.json",
                                     evaluation_files={"reinforce:71001": outcome}, authorization_path=None,
                                     authorization_root_path=None,
                                     access_audit_path=Path(tmp) / "audit.jsonl",
                                     output_path=Path(tmp) / "report.json")
            # Authorization is rejected from manifest/authorization metadata;
            # malformed locked payload is never parsed.
            self.assertEqual(outcome.read_text(), "this is deliberately not JSON")

    def test_post_evaluation_artifact_audit_binds_bytes_checkpoint_auth_and_filename(self):
        import hashlib
        from validation.analyze_torch_cohort import (_read_regular_json_no_follow,
            _read_verified_evaluator_document, _verify_artifact_audit_record)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "a2c-71001.json"
            payload = b'{"episode": 1}\n'
            artifact.write_bytes(payload)
            parsed, digest = _read_regular_json_no_follow(artifact)
            self.assertEqual(parsed, {"episode": 1})
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(_read_verified_evaluator_document(artifact, expected_sha256=digest)[0], parsed)
            # The mismatched bytes are not parsed as JSON first.
            artifact.write_bytes(b"not even json")
            with self.assertRaisesRegex(ValueError, "bytes do not match audit"):
                _read_verified_evaluator_document(artifact, expected_sha256=digest)
            artifact.write_bytes(payload)
            freeze = {"checkpoint_sha256": {"a2c:71001": "a" * 64},
                      "checkpoint_manifest_sha256": {"a2c:71001": "b" * 64},
                      "analysis_sha256": {"analysis.py": "c" * 64}}
            auth = {"authorization_id": "auth-1", "authorization_sha256": "d" * 64}
            record = {"action": "locked_test_evaluation_artifact_committed",
                      "selected_checkpoint_key": "a2c:71001", "authorization_id": "auth-1",
                      "authorization_sha256": "d" * 64, "preregistration_manifest_sha256": "e" * 64,
                      "selected_checkpoint_sha256": "a" * 64,
                      "selected_checkpoint_manifest_sha256": "b" * 64,
                      "analysis_sha256": freeze["analysis_sha256"],
                      "evaluation_access_record_sha256": "f" * 64,
                      "evaluation_artifact_name": artifact.name,
                      "evaluation_artifact_sha256": digest}
            self.assertEqual(_verify_artifact_audit_record(record, key="a2c:71001", authorization=auth,
                manifest_sha256="e" * 64, freeze=freeze, access_record_sha256="f" * 64,
                artifact_name=artifact.name), digest)
            with self.assertRaisesRegex(ValueError, "artifact audit"):
                _verify_artifact_audit_record(record, key="a2c:71001", authorization=auth,
                    manifest_sha256="e" * 64, freeze=freeze, access_record_sha256="f" * 64,
                    artifact_name="substituted.json")
            altered_digest = _verify_artifact_audit_record({**record, "evaluation_artifact_sha256": "0" * 64},
                    key="a2c:71001", authorization=auth, manifest_sha256="e" * 64,
                    freeze=freeze, access_record_sha256="f" * 64, artifact_name=artifact.name)
            self.assertNotEqual(altered_digest, digest)


if __name__ == "__main__":
    unittest.main()
