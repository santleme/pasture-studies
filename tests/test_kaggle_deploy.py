from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import kaggle_deploy as deploy


def result(stdout):
    return type("Result", (), {"stdout": stdout, "stderr": "", "returncode": 0})()


class KaggleDeployTests(unittest.TestCase):
    def setUp(self):
        self._real_source_commit = deploy._source_commit
        self._patches = [
            patch.object(deploy, "_source_commit", return_value="a" * 40),
            patch.object(deploy, "_head_commit", return_value="a" * 40),
            patch.object(deploy, "_committed_notebook_bytes", return_value=deploy.NOTEBOOK.read_bytes()),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def test_prepare_rejects_dirty_source_checkout(self):
        with patch.object(deploy, "run", return_value=result(" M notebooks/pasture_studies_kaggle.ipynb")):
            with self.assertRaisesRegex(deploy.DeploymentError, "must be clean and committed"):
                self._real_source_commit()

    def test_modes_allow_only_safe_readiness_and_legacy_modes(self):
        for mode in ("PREFLIGHT", "SMOKE", "READINESS"):
            self.assertEqual(deploy.validate_mode(f" {mode.lower()} "), mode)
        for mode in ("EXPERIMENT", "FULL_VALIDATION", "", "other"):
            with self.subTest(mode=mode), self.assertRaises(deploy.DeploymentError):
                deploy.validate_mode(mode)

    def test_metadata_uses_private_t4x2_schema(self):
        metadata = deploy.metadata_for("sample_user", gpu=True)
        self.assertEqual(metadata["id"], "sample_user/pasture-studies")
        self.assertEqual(metadata["code_file"], "pasture_studies_kaggle.ipynb")
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_internet"], "true")
        with self.assertRaises(deploy.DeploymentError):
            deploy.metadata_for("not/a/user")

    def test_prepare_preserves_exact_notebook_bytes_and_binds_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            deploy.prepare(stage, "sample_user", mode="PREFLIGHT")
            staged = stage / deploy.NOTEBOOK.name
            config = json.loads(deploy.deployment_record_path(stage).read_text())
            self.assertNotEqual(staged.read_bytes(), deploy.NOTEBOOK.read_bytes())
            self.assertEqual({item.name for item in stage.iterdir()}, {"kernel-metadata.json", deploy.NOTEBOOK.name})
            self.assertEqual(config["mode"], "PREFLIGHT")
            self.assertEqual(config["notebook_sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())
            self.assertEqual(config["source_commit"], deploy._source_commit())
            self.assertEqual(deploy.validate_staging_folder(stage)[0]["is_private"], "true")
            staged.write_text(staged.read_text() + " ")
            with self.assertRaisesRegex(deploy.DeploymentError, "exact staged notebook"):
                deploy.validate_staging_folder(stage)

    def test_mutable_detached_hash_cannot_authorize_staged_notebook_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = {"python": "3.13.15", "numpy": "2.1.3", "pytorch": "2.11.0+cu128",
                "cuda_runtime": "12.8", "cuda_available": True, "gpu_count": 2,
                "gpu_names": ["Tesla T4", "Tesla T4"]}
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime))
            for mode in ("PREFLIGHT", "SMOKE", "READINESS"):
                with self.subTest(mode=mode):
                    stage = root / mode.lower()
                    if mode == "READINESS":
                        deploy.prepare_release(stage, "sample_user", runtime_profile_path=runtime_path)
                    else:
                        deploy.prepare(stage, "sample_user", mode=mode)
                    staged_path = stage / deploy.NOTEBOOK.name
                    staged_path.write_bytes(staged_path.read_bytes() + b" ")
                    config_path = deploy.deployment_record_path(stage)
                    config = json.loads(config_path.read_text())
                    config["notebook_sha256"] = hashlib.sha256(staged_path.read_bytes()).hexdigest()
                    config_path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(deploy.DeploymentError, "exact committed source plus approved config"):
                        deploy.validate_staging_folder(stage)

    def test_readiness_requires_gpu_verified_manifest_and_exact_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = {"python": "3.13.15", "numpy": "2.1.3", "pytorch": "2.11.0+cu128",
                "cuda_runtime": "12.8", "cuda_available": True, "gpu_count": 2,
                "gpu_names": ["Tesla T4", "Tesla T4"]}
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime))
            stage = root / "staged"
            with patch.object(deploy, "_source_commit", return_value="a" * 40):
                deploy.prepare_release(stage, "sample_user", runtime_profile_path=runtime_path)
            staged = stage / deploy.NOTEBOOK.name
            config = json.loads(deploy.deployment_record_path(stage).read_text())
            self.assertEqual(config["mode"], "READINESS")
            self.assertEqual(deploy.embedded_deployment_config(staged)["mode"], "READINESS")
            self.assertEqual({p.name for p in stage.iterdir()}, {"kernel-metadata.json", deploy.NOTEBOOK.name})
            manifest = {"source": {"commit_sha": "a" * 40}, "notebook": {
                "source_sha256": config["source_notebook_sha256"],
                "staged_sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}}
            release = root / "release.json"
            release.write_text(json.dumps(manifest))
            with patch.object(deploy, "_manifest_digest", return_value=hashlib.sha256(release.read_bytes()).hexdigest()):
                digest = deploy.bind_release(stage, release)
                metadata, staged_notebook = deploy.validate_staging_folder(stage)
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(staged_notebook, staged)
            self.assertEqual(config["source_commit"], "a" * 40)
            self.assertEqual(digest, json.loads(deploy.deployment_record_path(stage).read_text())["release_manifest_sha256"])
            self.assertEqual(deploy.detached_manifest_path(stage).read_bytes(), release.read_bytes())
            self.assertEqual({p.name for p in stage.iterdir()}, {"kernel-metadata.json", deploy.NOTEBOOK.name})
            staged.write_text(staged.read_text() + " ")
            with self.assertRaisesRegex(deploy.DeploymentError, "exact staged notebook"):
                deploy.validate_staging_folder(stage)

    def test_readiness_rejects_manifest_mismatch_on_staged_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = {"python": "3.13.15", "numpy": "2.1.3", "pytorch": "2.11.0+cu128",
                "cuda_runtime": "12.8", "cuda_available": True, "gpu_count": 2,
                "gpu_names": ["Tesla T4", "Tesla T4"]}
            runtime_path = root / "runtime.json"
            runtime_path.write_text(json.dumps(runtime))
            stage = root / "stage"
            with patch.object(deploy, "_source_commit", return_value="a" * 40):
                deploy.prepare_release(stage, "sample_user", runtime_profile_path=runtime_path)
            release = root / "release.json"
            release.write_text("{}")
            with patch.object(deploy, "_manifest_digest", return_value="1" * 64):
                with self.assertRaises(deploy.DeploymentError):
                    deploy.bind_release(stage, release)

    def test_prepare_rejects_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()
            secret = stage / "kaggle.json"
            secret.write_text('{"key":"secret"}')
            with self.assertRaisesRegex(deploy.DeploymentError, "must be empty"):
                deploy.prepare(stage, "sample_user")
            self.assertEqual(list(stage.iterdir()), [secret])

    @patch.object(deploy, "run")
    def test_publish_fails_closed_without_no_run(self, run_mock):
        run_mock.side_effect = [result("Kaggle CLI 2.2.4"), result("usage: kernels push [-h] -p PATH")]
        with tempfile.TemporaryDirectory() as tmp, patch.object(deploy, "_source_commit", return_value="a" * 40):
            stage = Path(tmp) / "stage"
            deploy.prepare(stage, "sample_user")
            with self.assertRaisesRegex(deploy.DeploymentError, "refusing push"):
                deploy.publish(stage)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[-1].args[0], ["kaggle", "kernels", "push", "--help"])

    @patch.object(deploy, "write_publication_attestation", return_value=Path("attestation.json"))
    @patch.object(deploy, "check_cli", return_value=("9.0.0", "--no-run"))
    @patch.object(deploy, "pull_and_compare", return_value={"exact_match": True, "revision": 5})
    @patch.object(deploy, "run")
    def test_publish_no_run_then_numeric_remote_revision_compare(self, run_mock, compare_mock, cli_mock, attest_mock):
        run_mock.return_value = result("Kernel version 5 successfully pushed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(deploy, "_source_commit", return_value="a" * 40):
            stage = Path(tmp) / "stage"
            deploy.prepare(stage, "sample_user")
            result_doc = deploy.publish(stage)
            self.assertTrue(result_doc["exact_match"])
            compare_mock.assert_called_once_with(stage, "sample_user/pasture-studies", 5)
            attest_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.args[0][-1], "--no-run")

    @patch.object(deploy, "write_publication_attestation", return_value=Path("attestation.json"))
    @patch.object(deploy, "check_cli", return_value=("2.2.4", "help"))
    @patch.object(deploy, "pull_and_compare", return_value={"exact_match": True, "revision": 5})
    @patch.object(deploy, "run")
    def test_execute_requires_separate_gpu_consent_and_source_mode(self, run_mock, compare_mock, cli_mock, attest_mock):
        run_mock.return_value = result("Kernel version 5 successfully pushed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(deploy, "_source_commit", return_value="a" * 40):
            stage = Path(tmp) / "stage"
            deploy.prepare(stage, "sample_user", mode="SMOKE", gpu=True)
            with self.assertRaisesRegex(deploy.DeploymentError, "confirm-execution"):
                deploy.execute(stage, mode="SMOKE", confirm_execution=False, confirm_gpu=True)
            with self.assertRaisesRegex(deploy.DeploymentError, "confirm-gpu"):
                deploy.execute(stage, mode="SMOKE", confirm_execution=True)
            deploy.execute(stage, mode="SMOKE", confirm_execution=True, confirm_gpu=True)
            self.assertEqual(run_mock.call_args.args[0][-2:], ["--accelerator", "NvidiaTeslaT4"])
            compare_mock.assert_called_once_with(stage, "sample_user/pasture-studies", 5)

    def test_revision_is_taken_from_exact_numeric_kaggle_push_output(self):
        self.assertEqual(deploy.revision_from_push_output("Kernel version 7 successfully pushed. Please check progress"), 7)
        for output in ("", "Kernel version successfully pushed", "Kernel version 7 successfully pushed\nKernel version 8 successfully pushed"):
            with self.subTest(output=output), self.assertRaisesRegex(deploy.DeploymentError, "numeric kernel version"):
                deploy.revision_from_push_output(output)

    def test_remote_pull_uses_numeric_revision_and_compares_exact_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            deploy.prepare(stage, "sample_user", mode="PREFLIGHT")
            def fake_run(args):
                self.assertIn("sample_user/pasture-studies/13", args)
                target = Path(args[args.index("--path") + 1]) / deploy.NOTEBOOK.name
                target.write_bytes((stage / deploy.NOTEBOOK.name).read_bytes())
                return result("pulled")
            with patch.object(deploy, "run", side_effect=fake_run):
                evidence = deploy.pull_and_compare(stage, "sample_user/pasture-studies", 13)
            self.assertTrue(evidence["exact_match"])
            self.assertEqual(evidence["revision"], 13)
            with self.assertRaises(deploy.DeploymentError):
                deploy.pull_and_compare(stage, "sample_user/pasture-studies", 0)

    def test_remote_numeric_pull_content_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            deploy.prepare(stage, "sample_user", mode="PREFLIGHT")
            def fake_run(args):
                target = Path(args[args.index("--path") + 1]) / deploy.NOTEBOOK.name
                remote_doc = json.loads((stage / deploy.NOTEBOOK.name).read_text())
                remote_doc["cells"][0]["source"][0] += "\n# remote change\n"
                target.write_text(json.dumps(remote_doc, indent=1) + "\n")
                return result("pulled")
            with patch.object(deploy, "run", side_effect=fake_run):
                with self.assertRaisesRegex(deploy.DeploymentError, "differs from the staged notebook"):
                    deploy.pull_and_compare(stage, "sample_user/pasture-studies", 14)

    def test_publication_attestation_is_detached_and_binds_exact_content(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(deploy, "_source_commit", return_value="a" * 40):
            parent = Path(tmp)
            stage = parent / "staging"
            deploy.prepare(stage, "sample_user", mode="PREFLIGHT")
            config = json.loads(deploy.deployment_record_path(stage).read_text())
            evidence = {"exact_match": True, "revision": 12, "source_commit": "a" * 40,
                "notebook_sha256": config["notebook_sha256"], "remote_notebook_sha256": config["notebook_sha256"],
                "source_cells_sha256": "b" * 64}
            attestation = deploy.write_publication_attestation(stage, evidence, cli_version="2.2.4")
            self.assertEqual(attestation, parent / "staging-publication-attestation.json")
            self.assertFalse(attestation.parent == stage)
            record = json.loads(attestation.read_text())
            self.assertEqual(record["kernel_version"], 12)
            self.assertEqual(record["source_commit"], "a" * 40)
            self.assertEqual(record["staged_notebook_sha256"], config["notebook_sha256"])
            self.assertEqual(record["remote_notebook_sha256"], config["notebook_sha256"])
            self.assertTrue(record["exact_content_match"])
            self.assertNotIn("key", record)
            evidence["remote_notebook_sha256"] = "c" * 64
            with self.assertRaisesRegex(deploy.DeploymentError, "exact source-bound"):
                deploy.write_publication_attestation(stage, evidence, cli_version="2.2.4")

    @patch.object(deploy, "pull_and_compare", return_value={"exact_match": True,
        "source_cells_sha256": "b" * 64, "remote_notebook_sha256": "c" * 64})
    def test_deployment_record_requires_remote_exact_compare(self, compare_mock):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage"
            deploy.prepare(stage, "sample_user", mode="PREFLIGHT")
            record = root / "record.json"
            deploy.write_deployment_record(record, kernel="sample_user/pasture-studies", version="2.2.4",
                kernel_version="7", source_commit=deploy._source_commit(), mode="PREFLIGHT",
                metadata_path=stage / "kernel-metadata.json")
            doc = json.loads(record.read_text())
            self.assertEqual(doc["kaggle_kernel_version"], "7")
            self.assertTrue(doc["remote_exact_content_match"])
            self.assertEqual(doc["remote_notebook_sha256"], "c" * 64)
            compare_mock.assert_called_once()
            with self.assertRaises(deploy.DeploymentError):
                deploy.write_deployment_record(record, kernel="sample_user/pasture-studies", version="2.2.4",
                    kernel_version="latest", source_commit=deploy._source_commit(), mode="PREFLIGHT",
                    metadata_path=stage / "kernel-metadata.json")

    def test_artifact_manifest_hash_size_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.bin"
            artifact.write_bytes(b"artifact")
            manifest = root / "artifact.json"
            manifest.write_text(json.dumps({"files": [{"path": "model.bin",
                "sha256": hashlib.sha256(b"artifact").hexdigest(), "size_bytes": 8}]}))
            self.assertEqual(deploy.verify_manifest(manifest, root), ["model.bin"])
            manifest.write_text(json.dumps({"files": [{"path": "../outside", "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(deploy.DeploymentError, "Unsafe artifact path"):
                deploy.verify_manifest(manifest, root)

    @patch.object(deploy, "run")
    def test_retrieval_redacts_secrets_and_requires_numeric_revision(self, run_mock):
        run_mock.return_value = result("Authorization: Bearer topsecret\nkey=secret123")
        output = deploy.retrieve("logs", "sample_user/pasture-studies/5")
        self.assertNotIn("topsecret", output)
        self.assertNotIn("secret123", output)
        self.assertEqual(run_mock.call_args.args[0], ["kaggle", "kernels", "logs", "sample_user/pasture-studies/5"])
        with self.assertRaises(deploy.DeploymentError):
            deploy.retrieve("status", "sample_user/pasture-studies/latest")

    def test_auth_preflight_does_not_probe_or_disclose_credentials(self):
        auth = deploy.auth_preflight()
        self.assertEqual(auth["auth_status"], "not_verified")
        self.assertEqual(auth["auth_check"], "not_performed")
        self.assertNotIn("credentials_available", auth)


if __name__ == "__main__":
    unittest.main()
