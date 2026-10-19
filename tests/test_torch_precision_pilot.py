from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from validation import run_torch_precision_pilot as pilot


def fixture_manifest():
    families = list(pilot.FAMILIES)
    return {
        "schema_version": 1,
        "status": "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0",
        "manifest_sha256": "a" * 64,
        "source_sha256": pilot._source_hashes(),
        "seed_splits": {
            "training_root_seeds": [101, 202],
            "precision_pilot": {family: [1000 + i * 10, 1001 + i * 10]
                                 for i, family in enumerate(families)},
        },
    }


class TorchPrecisionPilotTests(unittest.TestCase):
    def test_plan_binds_exact_seed_schedule_and_is_outcome_free(self):
        manifest = fixture_manifest()
        plan = pilot.build_plan(manifest, manifest_file_sha256="b" * 64)
        self.assertEqual(plan["pilot_seed_ids_sha256"], pilot.canonical_sha(manifest["seed_splits"]["precision_pilot"]))
        self.assertEqual(plan["episodes_per_family"], 4)
        self.assertEqual(plan["sd_convention"], "sample standard deviation (ddof=1) across registered training-root × pilot-world cells")
        self.assertFalse(any(word in json.dumps(plan).lower() for word in ("mean_return", "per_seed_return", "trained_difference")))

    def test_manifest_requires_exact_file_hash_and_pending_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending.json"
            manifest = fixture_manifest()
            path.write_text(json.dumps(manifest))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(pilot, "verify_manifest") as verify:
                loaded = pilot.load_pending_manifest(path, digest)
            verify.assert_called_once_with(manifest, verify_current_sources=True)
            self.assertEqual(loaded, manifest)
            with patch.object(pilot, "verify_manifest"):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    pilot.load_pending_manifest(path, "c" * 64)
                manifest["status"] = "LOCKED_PRE_OUTCOME"
                path.write_text(json.dumps(manifest))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "requires the pending"):
                    pilot.load_pending_manifest(path, digest)

    def test_manifest_is_hashed_and_parsed_from_one_byte_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending.json"
            manifest = fixture_manifest()
            payload = json.dumps(manifest).encode()
            path.write_bytes(payload)
            with patch.object(Path, "read_bytes", autospec=True,
                              side_effect=lambda instance: payload) as read_bytes, \
                    patch.object(Path, "read_text", side_effect=AssertionError("must not reopen")), \
                    patch.object(pilot, "verify_manifest"):
                loaded = pilot.load_pending_manifest(path, hashlib.sha256(payload).hexdigest())
            self.assertEqual(loaded, manifest)
            read_bytes.assert_called_once_with(path)

    def test_run_emits_only_masked_statistics_and_separate_provenance(self):
        totals = {family: [1.0, 2.0, 4.0] for family in pilot.FAMILIES}
        summary = pilot._synthetic_sd_summary_for_test(totals)
        self.assertTrue(summary["synthetic_test_only"])
        self.assertEqual(set(summary["sample_sd_by_family"]), set(pilot.FAMILIES))
        self.assertAlmostEqual(summary["sample_sd_by_family"][pilot.FAMILIES[0]], np.std([1., 2., 4.], ddof=1))

    def test_rejects_nonfinite_and_zero_variance(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pilot._synthetic_sd_summary_for_test({family: [1.0, float("nan")] for family in pilot.FAMILIES})
        with self.assertRaisesRegex(ValueError, "nonzero"):
            pilot._synthetic_sd_summary_for_test({family: [7.0, 7.0] for family in pilot.FAMILIES})
        with self.assertRaisesRegex(ValueError, "exactly"):
            pilot._synthetic_sd_summary_for_test({"starter": [1., 2.]})

    def test_plan_only_cli_never_calls_runner_or_writes_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "pending.json"
            manifest = fixture_manifest()
            manifest_path.write_text(json.dumps(manifest))
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            output = Path(tmp) / "pilot.json"
            with patch.object(pilot, "verify_manifest"), \
                    patch.object(pilot, "run_precision_pilot", side_effect=AssertionError("must not execute")), \
                    patch("sys.stdout") as stdout:
                self.assertEqual(pilot.main([
                    "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                    "--output", str(output), "--plan-only",
                ]), 0)
            self.assertFalse(output.exists())

    def test_execute_cli_is_denied_without_independently_approved_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "pending.json"
            manifest = fixture_manifest()
            manifest_path.write_text(json.dumps(manifest))
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            output = Path(tmp) / "pilot.json"
            with patch.object(pilot, "verify_manifest"), \
                    patch.object(pilot, "run_precision_pilot", side_effect=AssertionError("must not execute")):
                with self.assertRaises(SystemExit):
                    pilot.main(["--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                                "--output", str(output), "--execute"])
            self.assertFalse(output.exists())

    def test_python_production_api_denies_before_evaluator_runs(self):
        called = []
        with self.assertRaisesRegex(ValueError, "independent protocol approval"):
            pilot.run_precision_pilot(evaluator=lambda *args: called.append(args),
                                      actor_factory=lambda seed: seed)
        self.assertEqual(called, [])

    def test_test_statistics_helper_cannot_run_evaluator_or_emit_pilot_artifact(self):
        # Regression for the former injected-evaluator test helper bypass.
        called = []
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pilot.json"
            with self.assertRaises(TypeError):
                pilot._synthetic_sd_summary_for_test(evaluator=lambda *args: called.append(args),
                                                     actor_factory=lambda seed: seed,
                                                     output_path=output)
            self.assertEqual(called, [])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
