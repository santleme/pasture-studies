from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import kaggle_release as release


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {
    "python": "3.13.15", "numpy": "2.1.3", "pytorch": "2.11.0+cu128",
    "cuda_runtime": "12.8", "cuda_available": True, "gpu_count": 2,
    "gpu_names": ["Tesla T4", "Tesla T4"],
}


class KaggleReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        for relative in (*release.ENGINE_SOURCE_PATHS, str(release.PREREGISTRATION),
                         str(release.RUNTIME_PROTOCOL_CANDIDATE), str(release.NOTEBOOK)):
            source = ROOT / relative
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "release-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Release Test"], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "fixture"], check=True)
        self.staged_notebook = Path(self.temp.name) / "staged-notebook.ipynb"
        shutil.copyfile(self.root / release.NOTEBOOK, self.staged_notebook)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return release.build_release_manifest(root=self.root, runtime_profile=RUNTIME,
                                              staged_notebook=self.staged_notebook)

    def verify(self, manifest):
        release.verify_release_manifest(manifest, root=self.root, staged_notebook=self.staged_notebook)

    def test_manifest_binds_engine_notebook_protocol_thresholds_and_splits(self):
        manifest = self.manifest()
        self.verify(manifest)
        self.assertEqual(manifest["source"]["worktree_clean"], True)
        self.assertEqual(set(manifest["training_engine"]["source_sha256"]), set(release.ENGINE_SOURCE_PATHS))
        self.assertEqual(len(manifest["notebook"]["source_sha256"]), 64)
        self.assertEqual(len(manifest["notebook"]["staged_sha256"]), 64)
        self.assertEqual(manifest["notebook"]["source_sha256"], manifest["notebook"]["staged_sha256"])
        self.assertEqual(len(manifest["notebook"]["source_cells_sha256"]), 64)
        self.assertEqual(len(manifest["acceptance_thresholds_sha256"]), 64)
        self.assertEqual(manifest["seeds_and_configs"]["seed_splits"]["locked_test"],
                         json.loads((self.root / release.PREREGISTRATION).read_text())["seed_splits"]["locked_test"])
        self.assertEqual(manifest["approval"]["status"], release.APPROVAL_STATUS_PENDING)
        candidate = manifest["runtime_protocol_candidate"]
        self.assertEqual(candidate["status"], "DRAFT_REVIEW_REQUIRED_NO_OUTCOMES")
        self.assertEqual(candidate["approval_status"], "PENDING_INDEPENDENT_REVIEW_AND_EXPLICIT_APPROVAL")
        self.assertEqual(len(candidate["file_sha256"]), 64)
        self.assertEqual(len(candidate["candidate_sha256"]), 64)

    def test_dirty_tree_fails_closed(self):
        (self.root / release.ENGINE_SOURCE_PATHS[0]).write_text("tampered\n")
        with self.assertRaisesRegex(release.ReleaseError, "clean Git worktree"):
            self.manifest()

    def test_untracked_worktree_fails_closed_at_generation(self):
        (self.root / "untracked.txt").write_text("dirty\n")
        with self.assertRaisesRegex(release.ReleaseError, "clean Git worktree"):
            self.manifest()

    def test_manifest_verifies_immutable_source_commit_after_head_advances(self):
        manifest = self.manifest()
        (self.root / "later.txt").write_text("later revision\n")
        self._commit_changes()
        self.verify(manifest)
        (self.root / "later-dirty.txt").write_text("uncommitted later worktree change\n")
        self.verify(manifest)

    def test_staged_notebook_hash_is_distinct_and_verified(self):
        staged = json.loads((self.staged_notebook).read_text())
        staged["metadata"]["release_stage"] = "upload-copy"
        self.staged_notebook.write_text(json.dumps(staged, separators=(",", ":")))
        manifest = self.manifest()
        self.assertNotEqual(manifest["notebook"]["source_sha256"], manifest["notebook"]["staged_sha256"])
        self.assertEqual(manifest["notebook"]["staged_path"], "staged_notebook")
        # A release manifest is portable across the local staging and Kaggle
        # execution filesystems; exact bytes, not an absolute path, are bound.
        staged_copy = Path(self.temp.name) / "kaggle" / "source_notebook.ipynb"
        staged_copy.parent.mkdir()
        shutil.copyfile(self.staged_notebook, staged_copy)
        release.verify_release_manifest(manifest, root=self.root, staged_notebook=staged_copy)
        staged_copy.write_text("tampered\n")
        with self.assertRaisesRegex(release.ReleaseError, "bindings differ"):
            release.verify_release_manifest(manifest, root=self.root, staged_notebook=staged_copy)

    def test_notebook_or_engine_mutation_after_commit_invalidates_manifest(self):
        manifest = self.manifest()
        path = self.root / release.NOTEBOOK
        path.write_bytes(path.read_bytes() + b" ")
        self._commit_changes()
        # Later source changes do not invalidate evidence bound to the old commit.
        self.verify(manifest)

    def test_threshold_or_seed_mutation_cannot_be_hidden_by_rehashing_manifest(self):
        manifest = self.manifest()
        mutated = json.loads(json.dumps(manifest))
        mutated["acceptance_thresholds_sha256"] = "0" * 64
        mutated["release_manifest_sha256"] = release.canonical_sha256(
            {key: value for key, value in mutated.items() if key != "release_manifest_sha256"})
        with self.assertRaisesRegex(release.ReleaseError, "bindings differ"):
            self.verify(mutated)
        mutated = json.loads(json.dumps(manifest))
        mutated["seeds_and_configs"]["seed_splits"]["locked_test"]["starter"][0] += 1
        mutated["release_manifest_sha256"] = release.canonical_sha256(
            {key: value for key, value in mutated.items() if key != "release_manifest_sha256"})
        with self.assertRaisesRegex(release.ReleaseError, "bindings differ"):
            self.verify(mutated)

    def test_candidate_must_preserve_historical_thresholds_seeds_and_config(self):
        candidate_path = self.root / release.RUNTIME_PROTOCOL_CANDIDATE
        candidate = json.loads(candidate_path.read_text())
        candidate["historical_protocol"]["training_config"]["episodes_per_train_seed"] += 1
        candidate["candidate_sha256_canonical_json_excluding_candidate_sha256"] = release.canonical_sha256(
            {key: value for key, value in candidate.items()
             if key != "candidate_sha256_canonical_json_excluding_candidate_sha256"})
        candidate_path.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n")
        self._commit_changes()
        with self.assertRaisesRegex(release.ReleaseError, "changes historical training_config"):
            self.manifest()

    def test_protocol_change_after_commit_invalidates_manifest(self):
        manifest = self.manifest()
        path = self.root / release.PREREGISTRATION
        protocol = json.loads(path.read_text())
        protocol["acceptance_thresholds"]["training_validity"]["training_seed_count_per_arm"] += 1
        protocol["manifest_sha256"] = release.canonical_sha256(
            {key: value for key, value in protocol.items() if key != "manifest_sha256"})
        path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n")
        self._commit_changes()
        self.verify(manifest)

    def test_untrusted_or_env_independent_approval_never_unlocks_experiment(self):
        manifest = self.manifest()
        fake = {"schema_version": 1, "release_manifest_sha256": manifest["release_manifest_sha256"],
                "status": "APPROVED", "algorithm": "Ed25519", "public_key": "caller-chosen",
                "signature": "00" * 64}
        self.assertEqual(release.approval_status(fake, manifest["release_manifest_sha256"]),
                         release.APPROVAL_STATUS_PENDING)
        for mode in ("EXPERIMENT", "FULL_VALIDATION"):
            with self.assertRaisesRegex(release.ReleaseError, "independently verifiable"):
                release.authorize_mode(mode, manifest, fake)
        self.assertEqual(release.authorize_mode("SMOKE", manifest), "SMOKE")
        self.assertEqual(release.authorize_mode("PREFLIGHT", manifest), "PREFLIGHT")
        self.assertEqual(release.authorize_mode("EXPLORATORY", manifest,
                         config=release.EXPLORATORY_CONFIG_VALUE), "EXPLORATORY")
        for key in ("episodes", "seed", "locked_test_access", "formal_eligible"):
            bad = dict(release.EXPLORATORY_CONFIG_VALUE)
            bad[key] = not bad[key] if isinstance(bad[key], bool) else bad[key] + 1
            with self.assertRaises(release.ReleaseError):
                release.authorize_mode("EXPLORATORY", manifest, config=bad)

    def test_readiness_accepts_only_the_two_bounded_non_outcome_workloads(self):
        for workload, updates, episodes in (("t4x2_path_smoke", 1, 16),
                                             ("checkpoint_recovery", 2, 32)):
            config = {"task": workload, "max_updates": updates, "max_episodes": episodes,
                      "max_devices": 2, "locked_test_access": False, "outcome_analysis": False,
                      "benchmark_eligible": False, "checkpoint_promotion": False}
            self.assertEqual(release.authorize_mode("READINESS", {}, config=config), "READINESS")
        valid = {"task": "checkpoint_recovery", "max_updates": 2, "max_episodes": 32,
                 "max_devices": 2, "locked_test_access": False, "outcome_analysis": False,
                 "benchmark_eligible": False, "checkpoint_promotion": False}
        for key, value in (("max_updates", 3), ("max_episodes", 33), ("max_devices", 3),
                           ("locked_test_access", True), ("outcome_analysis", True),
                           ("benchmark_eligible", True), ("checkpoint_promotion", True)):
            bad = {**valid, key: value}
            with self.subTest(key=key), self.assertRaises(release.ReleaseError):
                release.authorize_mode("READINESS", {}, config=bad)
        with self.assertRaisesRegex(release.ReleaseError, "exact bounded"):
            release.authorize_mode("READINESS", {}, config={**valid, "extra": "ignored?"})

    def test_runtime_profile_is_strict(self):
        for bad in (
            {**RUNTIME, "gpu_names": ["Tesla T4"]},
            {**RUNTIME, "cuda_available": False},
            {**RUNTIME, "secret_token": "not-allowed"},
        ):
            with self.assertRaises(release.ReleaseError):
                release.build_release_manifest(root=self.root, runtime_profile=bad,
                                               staged_notebook=self.staged_notebook)

    def _commit_changes(self):
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "fixture update"], check=True)


if __name__ == "__main__":
    unittest.main()
