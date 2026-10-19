"""Synthetic and fail-closed tests for the separately versioned pilot finalizer."""
from __future__ import annotations

import json
import base64
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from validation.analyze_torch_cohort import FAMILIES
from validation.finalize_precision_pilot_v2 import (
    _validate_cells,
    finalize_precision_pilot_v2,
)
from validation.finalize_precision_pilot_v2 import _canonical, _sha
from validation.finalize_precision_pilot_v2 import _verify_signature, TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256
from validation.preregister_torch_mvp1 import canonical_sha
from validation.run_torch_precision_pilot import PILOT_SOURCES

ROOT = Path(__file__).resolve().parents[1]


def _public_key_fingerprint(public_path: Path) -> str:
    der = subprocess.run(["openssl", "pkey", "-pubin", "-in", str(public_path), "-outform", "DER"],
                         check=True, capture_output=True).stdout
    return hashlib.sha256(der).hexdigest()


def _fixture():
    train_seeds = [11, 22]
    worlds = {family: [101 + i * 10, 102 + i * 10] for i, family in enumerate(FAMILIES)}
    manifest = {"manifest_sha256": "a" * 64, "protocol_id": "candidate-v2",
                "pilot_seed_ids_sha256": "b" * 64,
                "seed_splits": {"training_root_seeds": train_seeds, "precision_pilot": worlds}}
    schedule_sha = "c" * 64
    candidate_sha = "d" * 64
    release_sha = "e" * 64
    cells = []
    for family_idx, family in enumerate(FAMILIES):
        for train_seed in train_seeds:
            for world_seed in worlds[family]:
                cells.append({"family": family, "training_root_seed": train_seed,
                              "world_seed": world_seed,
                              "episode_total_return": float(family_idx + train_seed / 10 + world_seed / 1000)})
    doc = {"schema_version": 2, "status": "RAW_PRECISION_PILOT_CELLS",
           "protocol_id": manifest["protocol_id"], "candidate_sha256": candidate_sha,
           "release_manifest_sha256": release_sha,
           "manifest_sha256": manifest["manifest_sha256"],
           "pilot_seed_ids_sha256": manifest["pilot_seed_ids_sha256"],
           "schedule_sha256": schedule_sha, "cells": cells}
    return manifest, doc, schedule_sha, candidate_sha, release_sha


class TorchPrecisionPilotV2Tests(unittest.TestCase):
    def test_recomputes_masked_sample_sd_from_exact_synthetic_cell_schedule(self):
        manifest, doc, schedule_sha, candidate_sha, release_sha = _fixture()
        result = _validate_cells(doc, manifest=manifest, schedule_sha256=schedule_sha,
                                 candidate_sha256=candidate_sha,
                                 release_manifest_sha256=release_sha)
        self.assertEqual(set(result), set(FAMILIES))
        self.assertTrue(all(sd > 0 for sd in result.values()))

    def test_rejects_sd_only_caller_artifact_and_extra_cell_fields(self):
        manifest, doc, schedule_sha, candidate_sha, release_sha = _fixture()
        with self.assertRaisesRegex(ValueError, "schema is invalid"):
            _validate_cells({"sd_untrained_returns_by_family": {f: 1.0 for f in FAMILIES}},
                            manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)
        doc["cells"][0]["mean"] = 3.0
        with self.assertRaisesRegex(ValueError, "unsupported or missing"):
            _validate_cells(doc, manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)

    def test_rejects_duplicate_missing_nonfinite_and_boolean_cells(self):
        manifest, doc, schedule_sha, candidate_sha, release_sha = _fixture()
        duplicate = json.loads(json.dumps(doc))
        duplicate["cells"][-1] = duplicate["cells"][0]
        with self.assertRaisesRegex(ValueError, "duplicate schedule cell"):
            _validate_cells(duplicate, manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)
        incomplete = json.loads(json.dumps(doc))
        incomplete["cells"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete or contain extras"):
            _validate_cells(incomplete, manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)
        nonfinite = json.loads(json.dumps(doc))
        nonfinite["cells"][0]["episode_total_return"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            _validate_cells(nonfinite, manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)
        boolean = json.loads(json.dumps(doc))
        boolean["cells"][0]["episode_total_return"] = True
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            _validate_cells(boolean, manifest=manifest, schedule_sha256=schedule_sha,
                            candidate_sha256=candidate_sha,
                            release_manifest_sha256=release_sha)

    def test_draft_candidate_denies_before_reading_raw_input_or_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            raw.write_text("not JSON and must not be opened")
            output = root / "masked.json"
            with self.assertRaisesRegex(ValueError, "not independently approved"):
                finalize_precision_pilot_v2(
                    candidate_path=ROOT / "validation/protocols/pasture-study-analysis-v2-candidate.json",
                    release_manifest_path=root / "missing-release.json",
                    pending_manifest_path=root / "missing-pending.json",
                    authorization_path=root / "missing-auth.json",
                    authorization_root_path=root / "missing-auth-root.pem",
                    run_attestation_path=root / "missing-attestation.json",
                    attestation_root_path=root / "missing-attestation-root.pem",
                    protocol_review_path=root / "missing-review.json",
                    protocol_review_root_path=root / "missing-review-root.pem",
                    raw_results_path=raw, output_path=output)
            self.assertEqual(raw.read_text(), "not JSON and must not be opened")
            self.assertFalse(output.exists())
            self.assertFalse((root / "audit.jsonl").exists())

    def test_signed_synthetic_release_recomputes_only_masked_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signers = {}
            for role in ("reviewer", "authorizer", "attester"):
                private = root / f"{role}-private.pem"
                public = root / f"{role}-public.pem"
                subprocess.run(["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
                               check=True, capture_output=True)
                subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                               check=True, capture_output=True)
                signers[role] = {"private": private, "public": public,
                                 "sha": _public_key_fingerprint(public)}
            sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in PILOT_SOURCES}
            runtime = {"python": "3.13.15", "numpy": "2.1.3", "pytorch": "2.11.0+cu128",
                       "cuda_runtime": "12.8", "cuda_available": True, "gpu_count": 2,
                       "gpu_names": ["Tesla T4", "Tesla T4"]}
            publication = {"kernel": "test-owner/test-kernel", "version": 9,
                           "content_sha256": "1" * 64}
            candidate = {"candidate_id": "candidate-v2", "status": "APPROVED_PRE_OUTCOME_PROTOCOL",
                         "release_eligible": True,
                         "approval": {"pilot_authorized": True, "independent_review": "PASS",
                                      "explicit_user_authorization_for_outcomes": True},
                         "provenance": {"source_sha256": sources},
                         "runtime_profile_observed": runtime,
                         "notebook_publication": publication}
            candidate["candidate_sha256"] = canonical_sha(candidate)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(json.dumps(candidate, sort_keys=True) + "\n")
            commit = "2" * 40
            release = {"protocol_candidate_sha256": candidate["candidate_sha256"],
                       "source_commit_sha": commit, "source_sha256": sources,
                       "runtime_profile": runtime, "notebook_publication": publication}
            release["release_manifest_sha256"] = canonical_sha(release)
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release, sort_keys=True) + "\n")
            release_file_sha = _sha(release_path.read_bytes())

            train_seeds = [11, 22]
            worlds = {family: [201 + i * 10, 202 + i * 10] for i, family in enumerate(FAMILIES)}
            seed_ids_sha = canonical_sha(worlds)
            manifest = {"status": "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0",
                        "protocol_id": candidate["candidate_id"],
                        "source_sha256": sources,
                        "seed_splits": {"training_root_seeds": train_seeds, "precision_pilot": worlds},
                        "pilot_seed_ids_sha256": seed_ids_sha}
            manifest["manifest_sha256"] = canonical_sha(manifest)
            manifest_path = root / "pending.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            manifest_file_sha = _sha(manifest_path.read_bytes())
            schedule_sha = canonical_sha({"training_root_seeds": train_seeds,
                "precision_pilot_world_seeds": worlds,
                "actor_policy": "untrained ActorCritic(value_head=False, seed=training_root_seed)",
                "evaluation_order": "family-major, world-seed, then training-root-seed"})

            def signed(body, name, role):
                signer = signers[role]
                message_path = root / f"{name}.msg"
                signature_path = root / f"{name}.sig"
                message_path.write_bytes(_canonical(body))
                subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(signer["private"]), "-rawin",
                                "-in", str(message_path), "-out", str(signature_path)],
                               check=True, capture_output=True)
                return {**body, "signature_ed25519_base64": base64.b64encode(signature_path.read_bytes()).decode()}

            review = signed({"schema_version": 1, "action": "approve_precision_pilot_protocol",
                "candidate_sha256": candidate["candidate_sha256"],
                "candidate_file_sha256": _sha(candidate_path.read_bytes()),
                "release_manifest_sha256": release_file_sha,
                "pending_manifest_file_sha256": manifest_file_sha,
                "pending_manifest_sha256": manifest["manifest_sha256"], "source_commit_sha": commit,
                "schedule_sha256": schedule_sha,
                "notebook_publication_sha256": canonical_sha(publication)}, "review", "reviewer")
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review, sort_keys=True) + "\n")

            cells = []
            expected_sds = {}
            for i, family in enumerate(FAMILIES):
                values = [1.0 + i, 2.0 + i, 4.0 + i, 8.0 + i]
                expected_sds[family] = float(np.std(values, ddof=1))
                for train_seed, world_seed, value in zip(
                        [11, 11, 22, 22], worlds[family] * 2, values):
                    cells.append({"family": family, "training_root_seed": train_seed,
                                  "world_seed": world_seed, "episode_total_return": value})
            raw = {"schema_version": 2, "status": "RAW_PRECISION_PILOT_CELLS",
                   "protocol_id": candidate["candidate_id"],
                   "candidate_sha256": candidate["candidate_sha256"],
                   "release_manifest_sha256": release_file_sha,
                   "manifest_sha256": manifest["manifest_sha256"],
                   "pilot_seed_ids_sha256": seed_ids_sha,
                   "schedule_sha256": schedule_sha, "cells": cells}
            raw_path = root / "cells.json"
            raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
            raw_sha = _sha(raw_path.read_bytes())

            auth = signed({"schema_version": 1, "authorization_id": "synthetic-auth",
                "action": "collect_precision_pilot", "candidate_sha256": candidate["candidate_sha256"],
                "candidate_file_sha256": _sha(candidate_path.read_bytes()),
                "release_manifest_sha256": release_file_sha,
                "pending_manifest_file_sha256": manifest_file_sha,
                "pending_manifest_sha256": manifest["manifest_sha256"], "source_commit_sha": commit,
                "schedule_sha256": schedule_sha, "runtime_sha256": canonical_sha(runtime),
                "notebook_publication_sha256": canonical_sha(publication),
                "expires_at_utc": "2999-01-01T00:00:00+00:00"}, "auth", "authorizer")
            auth_path = root / "auth.json"
            auth_path.write_text(json.dumps(auth, sort_keys=True) + "\n")
            auth_file_sha = _sha(auth_path.read_bytes())
            attestation = signed({"schema_version": 1, "run_id": "synthetic-run",
                "action": "attest_precision_pilot_results", "authorization_id": "synthetic-auth",
                "authorization_sha256": auth_file_sha,
                "candidate_sha256": candidate["candidate_sha256"],
                "release_manifest_sha256": release_file_sha,
                "pending_manifest_file_sha256": manifest_file_sha,
                "pending_manifest_sha256": manifest["manifest_sha256"], "source_commit_sha": commit,
                "schedule_sha256": schedule_sha, "runtime_sha256": canonical_sha(runtime),
                "notebook_publication_sha256": canonical_sha(publication),
                "raw_results_sha256": raw_sha, "raw_results_bytes": raw_path.stat().st_size}, "attestation", "attester")
            attestation_path = root / "attestation.json"
            attestation_path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
            output = root / "masked.json"
            with patch("validation.finalize_precision_pilot_v2._git_state", return_value=(commit, False)), \
                 patch("validation.finalize_precision_pilot_v2._current_runtime", return_value=runtime), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256",
                       frozenset({signers["reviewer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256",
                       frozenset({signers["authorizer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_ATTESTATION_ROOT_SHA256",
                       frozenset({signers["attester"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.PILOT_AUDIT_STORE_PATH", root / "audit.jsonl"):
                result = finalize_precision_pilot_v2(
                    candidate_path=candidate_path, release_manifest_path=release_path,
                    pending_manifest_path=manifest_path, authorization_path=auth_path,
                    authorization_root_path=signers["authorizer"]["public"],
                    run_attestation_path=attestation_path,
                    attestation_root_path=signers["attester"]["public"],
                    protocol_review_path=review_path,
                    protocol_review_root_path=signers["reviewer"]["public"],
                    raw_results_path=raw_path, output_path=output)
            self.assertEqual(result["sd_untrained_returns_by_family"], expected_sds)
            self.assertEqual(result["delta0_by_family"], {family: .05 * sd for family, sd in expected_sds.items()})
            self.assertFalse(result["locked_outcomes_read"])
            serialized = output.read_text()
            self.assertNotIn('"cells"', serialized)
            self.assertNotIn('"episode_total_return"', serialized)
            self.assertNotIn('"mean_return"', serialized)

            # The caller cannot select a fresh audit file to replay an authorization.
            replay_output = root / "replayed.json"
            with patch("validation.finalize_precision_pilot_v2._git_state", return_value=(commit, False)), \
                 patch("validation.finalize_precision_pilot_v2._current_runtime", return_value=runtime), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256",
                       frozenset({signers["reviewer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256",
                       frozenset({signers["authorizer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_ATTESTATION_ROOT_SHA256",
                       frozenset({signers["attester"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.PILOT_AUDIT_STORE_PATH", root / "audit.jsonl"):
                with self.assertRaisesRegex(ValueError, "already been consumed"):
                    finalize_precision_pilot_v2(
                        candidate_path=candidate_path, release_manifest_path=release_path,
                        pending_manifest_path=manifest_path, authorization_path=auth_path,
                        authorization_root_path=signers["authorizer"]["public"],
                        run_attestation_path=attestation_path,
                        attestation_root_path=signers["attester"]["public"],
                        protocol_review_path=review_path,
                        protocol_review_root_path=signers["reviewer"]["public"],
                        raw_results_path=raw_path, output_path=replay_output)
            self.assertFalse(replay_output.exists())

            # A separate authorization over the original digest detects
            # same-size raw-file substitution before parsing/output.
            tampered_bytes = raw_path.read_bytes().replace(b'"episode_total_return": 1.0',
                                                           b'"episode_total_return": 9.0', 1)
            if len(tampered_bytes) == raw_path.stat().st_size:
                raw_path.write_bytes(tampered_bytes)
            else:
                # JSON whitespace differs by serializer versions; use a digit
                # replacement that preserves exact byte length.
                original = raw_path.read_bytes()
                index = original.index(b'"episode_total_return": 1.0')
                raw_path.write_bytes(original[:index] + original[index:].replace(
                    b'"episode_total_return": 1.0', b'"episode_total_return": 9.0', 1))
            new_auth = signed({"schema_version": 1, "authorization_id": "synthetic-auth-tamper",
                "action": "collect_precision_pilot", "candidate_sha256": candidate["candidate_sha256"],
                "candidate_file_sha256": _sha(candidate_path.read_bytes()),
                "release_manifest_sha256": release_file_sha,
                "pending_manifest_file_sha256": manifest_file_sha,
                "pending_manifest_sha256": manifest["manifest_sha256"], "source_commit_sha": commit,
                "schedule_sha256": schedule_sha, "runtime_sha256": canonical_sha(runtime),
                "notebook_publication_sha256": canonical_sha(publication),
                "expires_at_utc": "2999-01-01T00:00:00+00:00"}, "tamper-auth", "authorizer")
            new_auth_path = root / "tamper-auth.json"
            new_auth_path.write_text(json.dumps(new_auth, sort_keys=True) + "\n")
            new_attestation = signed({"schema_version": 1, "run_id": "synthetic-run-tamper",
                "action": "attest_precision_pilot_results", "authorization_id": "synthetic-auth-tamper",
                "authorization_sha256": _sha(new_auth_path.read_bytes()),
                "candidate_sha256": candidate["candidate_sha256"],
                "release_manifest_sha256": release_file_sha,
                "pending_manifest_file_sha256": manifest_file_sha,
                "pending_manifest_sha256": manifest["manifest_sha256"], "source_commit_sha": commit,
                "schedule_sha256": schedule_sha, "runtime_sha256": canonical_sha(runtime),
                "notebook_publication_sha256": canonical_sha(publication),
                "raw_results_sha256": raw_sha, "raw_results_bytes": len(tampered_bytes)},
                "tamper-attestation", "attester")
            new_attestation_path = root / "tamper-attestation.json"
            new_attestation_path.write_text(json.dumps(new_attestation, sort_keys=True) + "\n")
            tamper_output = root / "tampered-masked.json"
            with patch("validation.finalize_precision_pilot_v2._git_state", return_value=(commit, False)), \
                 patch("validation.finalize_precision_pilot_v2._current_runtime", return_value=runtime), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256",
                       frozenset({signers["reviewer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256",
                       frozenset({signers["authorizer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_ATTESTATION_ROOT_SHA256",
                       frozenset({signers["attester"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.PILOT_AUDIT_STORE_PATH", root / "audit.jsonl"):
                with self.assertRaisesRegex(ValueError, "raw result bytes differ"):
                    finalize_precision_pilot_v2(
                        candidate_path=candidate_path, release_manifest_path=release_path,
                        pending_manifest_path=manifest_path, authorization_path=new_auth_path,
                        authorization_root_path=signers["authorizer"]["public"],
                        run_attestation_path=new_attestation_path,
                        attestation_root_path=signers["attester"]["public"],
                        protocol_review_path=review_path,
                        protocol_review_root_path=signers["reviewer"]["public"],
                        raw_results_path=raw_path, output_path=tamper_output)
            self.assertFalse(tamper_output.exists())

            # Re-sign the run with the authorizer key: even a key present in
            # both allowlists cannot satisfy the distinct-attester invariant.
            # Add a newline to prove PEM re-encoding does not disguise reuse.
            alternate_authorizer_pem = root / "authorizer-reencoded-public.pem"
            alternate_authorizer_pem.write_bytes(signers["authorizer"]["public"].read_bytes() + b"\n")
            self.assertEqual(_public_key_fingerprint(alternate_authorizer_pem),
                             signers["authorizer"]["sha"])
            attestation_body = {key: value for key, value in attestation.items()
                                if key != "signature_ed25519_base64"}
            reused_attestation = signed(attestation_body, "reused-attestation", "authorizer")
            attestation_path.write_text(json.dumps(reused_attestation, sort_keys=True) + "\n")
            reused_output = root / "reused-signer.json"
            with patch("validation.finalize_precision_pilot_v2._git_state", return_value=(commit, False)), \
                 patch("validation.finalize_precision_pilot_v2._current_runtime", return_value=runtime), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PROTOCOL_REVIEW_ROOT_SHA256",
                       frozenset({signers["reviewer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256",
                       frozenset({signers["authorizer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.TRUSTED_PILOT_ATTESTATION_ROOT_SHA256",
                       frozenset({signers["authorizer"]["sha"]})), \
                 patch("validation.finalize_precision_pilot_v2.PILOT_AUDIT_STORE_PATH", root / "unused-audit.jsonl"):
                with self.assertRaisesRegex(ValueError, "must be distinct signers"):
                    finalize_precision_pilot_v2(
                        candidate_path=candidate_path, release_manifest_path=release_path,
                        pending_manifest_path=manifest_path, authorization_path=auth_path,
                        authorization_root_path=signers["authorizer"]["public"],
                        run_attestation_path=attestation_path,
                        attestation_root_path=alternate_authorizer_pem,
                        protocol_review_path=review_path,
                        protocol_review_root_path=signers["reviewer"]["public"],
                        raw_results_path=raw_path, output_path=reused_output)
            self.assertFalse(reused_output.exists())

    def test_empty_trusted_root_allowlist_rejects_even_validly_signed_data(self):
        self.assertEqual(TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256, frozenset())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "private.pem"
            public = root / "public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
                           check=True, capture_output=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                           check=True, capture_output=True)
            signature = base64.b64encode(b"not-read-after-root-denial").decode()
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                _verify_signature({"signature_ed25519_base64": signature}, root_path=public,
                                  allowed_fingerprints=TRUSTED_PILOT_AUTHORIZATION_ROOT_SHA256,
                                  label="pilot authorization")


if __name__ == "__main__":
    unittest.main()
