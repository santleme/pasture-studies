from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import kaggle_deploy as deploy


class KaggleDeployTests(unittest.TestCase):
    def test_modes_only_allow_preflight_and_smoke(self):
        self.assertEqual(deploy.validate_mode(" smoke "), "SMOKE")
        self.assertEqual(deploy.validate_mode("preflight"), "PREFLIGHT")
        for mode in ("EXPERIMENT", "FULL_VALIDATION", "", "other"):
            with self.subTest(mode=mode), self.assertRaises(deploy.DeploymentError):
                deploy.validate_mode(mode)

    def test_metadata_uses_valid_kernel_schema_and_t4x2_selection(self):
        metadata = deploy.metadata_for("sample_user", gpu=True)
        self.assertEqual(metadata["id"], "sample_user/pasture-studies")
        self.assertEqual(metadata["code_file"], "pasture_studies_kaggle.ipynb")
        self.assertEqual(metadata["language"], "python")
        self.assertEqual(metadata["kernel_type"], "notebook")
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["enable_tpu"], "false")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["dataset_sources"], [])
        with self.assertRaises(deploy.DeploymentError):
            deploy.metadata_for("not/a/user")

    def test_prepare_copies_notebook_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "kernel"
            source_hash = hashlib.sha256(deploy.NOTEBOOK.read_bytes()).hexdigest()
            metadata_path = deploy.prepare(output, "sample_user", mode="PREFLIGHT")
            self.assertTrue((output / deploy.NOTEBOOK.name).is_file())
            self.assertEqual(json.loads(metadata_path.read_text())["id"], "sample_user/pasture-studies")
            staged = json.loads((output / deploy.NOTEBOOK.name).read_text())
            self.assertTrue(any('os.environ["PASTURE_MODE"] = "PREFLIGHT"' in "".join(cell["source"])
                                for cell in staged["cells"]))
            self.assertEqual(hashlib.sha256(deploy.NOTEBOOK.read_bytes()).hexdigest(), source_hash)
            with self.assertRaises(deploy.DeploymentError):
                deploy.prepare(output, "sample_user", mode="EXPERIMENT")

    def test_gpu_staging_injects_fail_closed_cuda_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            deploy.prepare(stage, "sample_user", mode="SMOKE", gpu=True)
            staged = json.loads((stage / deploy.NOTEBOOK.name).read_text())
            code = "\n".join("".join(cell["source"]) for cell in staged["cells"])
            self.assertIn('os.environ["PASTURE_REQUIRE_CUDA"] = "1"', code)
            metadata, _ = deploy.validate_staging_folder(stage)
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            notebook_path = stage / deploy.NOTEBOOK.name
            for cell in staged["cells"]:
                cell["source"] = [line.replace('os.environ["PASTURE_REQUIRE_CUDA"] = "1"\n', "")
                                   for line in cell.get("source", [])]
            notebook_path.write_text(json.dumps(staged))
            with self.assertRaisesRegex(deploy.DeploymentError, "fail-closed"):
                deploy.validate_staging_folder(stage)

    def test_prepare_refuses_stale_files_in_staging_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "kernel"
            output.mkdir()
            secret = output / "kaggle.json"
            secret.write_text('{"key":"must not upload"}')
            with self.assertRaisesRegex(deploy.DeploymentError, "must be empty"):
                deploy.prepare(output, "sample_user")
            self.assertEqual(list(output.iterdir()), [secret])

    @patch.object(deploy, "run")
    def test_publish_fails_closed_without_no_run_and_never_calls_push(self, run_mock):
        run_mock.side_effect = [
            type("Result", (), {"stdout": "Kaggle CLI 2.2.4", "stderr": ""})(),
            type("Result", (), {"stdout": "usage: kaggle kernels push [-h] [-p PATH]", "stderr": ""})(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            deploy.prepare(stage, "sample_user")
            with self.assertRaisesRegex(deploy.DeploymentError, "refusing push"):
                deploy.publish(stage)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual([call.args[0] for call in run_mock.call_args_list][-1],
                         ["kaggle", "kernels", "push", "--help"])

    @patch.object(deploy, "run")
    def test_publish_uses_no_run_only_when_advertised(self, run_mock):
        run_mock.side_effect = [
            type("Result", (), {"stdout": "Kaggle CLI 9.0.0", "stderr": ""})(),
            type("Result", (), {"stdout": "--no-run Do not execute", "stderr": ""})(),
            type("Result", (), {"stdout": "submitted", "stderr": ""})(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            deploy.prepare(stage, "sample_user")
            deploy.publish(stage)
            self.assertEqual(run_mock.call_args.args[0],
                             ["kaggle", "kernels", "push", "--path", str(stage), "--no-run"])

    @patch.object(deploy, "run")
    def test_execute_requires_consent_and_uses_t4x2_accelerator(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            deploy.prepare(stage, "sample_user", mode="SMOKE", gpu=True)
            with self.assertRaisesRegex(deploy.DeploymentError, "confirm-execution"):
                deploy.execute(stage, mode="SMOKE", confirm_execution=False)
            with self.assertRaisesRegex(deploy.DeploymentError, "confirm-gpu"):
                deploy.execute(stage, mode="SMOKE", confirm_execution=True)
            with self.assertRaises(deploy.DeploymentError):
                deploy.execute(stage, mode="EXPERIMENT", confirm_execution=True, confirm_gpu=True)
            deploy.execute(stage, mode="SMOKE", confirm_execution=True, confirm_gpu=True)
            run_mock.assert_called_once_with(["kaggle", "kernels", "push", "--path", str(stage),
                                              "--accelerator", "NvidiaTeslaT4"])

    @patch.object(deploy, "run")
    def test_publish_requires_private_exact_staging_contents(self, run_mock):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "kernel"
            metadata_path = deploy.prepare(stage, "sample_user")
            metadata = json.loads(metadata_path.read_text())
            metadata["is_private"] = "false"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(deploy.DeploymentError, "not explicitly private"):
                deploy.publish(stage)
            self.assertEqual(run_mock.call_count, 0)

    @patch.object(deploy, "run")
    def test_status_logs_and_outputs_use_official_cli_commands(self, run_mock):
        run_mock.return_value = type("Result", (), {"stdout": "Authorization: Bearer topsecret\nkey=abc123secret", "stderr": ""})()
        for kind, expected in (("status", "status"), ("logs", "logs"), ("output", "output")):
            with self.subTest(kind=kind):
                result = deploy.retrieve(kind, "sample_user/pasture-studies/42")
                self.assertNotIn("topsecret", result)
                self.assertNotIn("abc123secret", result)
                self.assertEqual(run_mock.call_args.args[0][:4],
                                 ["kaggle", "kernels", expected, "sample_user/pasture-studies/42"])
        with self.assertRaises(deploy.DeploymentError):
            deploy.retrieve("status", "bad kernel")
        with self.assertRaises(deploy.DeploymentError):
            deploy.retrieve("status", "sample_user/pasture-studies/latest")
        deploy.retrieve("status", "sample_user/pasture-studies")
        self.assertEqual(run_mock.call_args.args[0],
                         ["kaggle", "kernels", "status", "sample_user/pasture-studies"])

    def test_manifest_verifies_hash_size_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.bin"
            artifact.write_bytes(b"artifact")
            manifest = root / "artifact_manifest.json"
            manifest.write_text(json.dumps({"files": [{"path": "model.bin",
                "sha256": hashlib.sha256(b"artifact").hexdigest(), "size_bytes": 8}]}))
            self.assertEqual(deploy.verify_manifest(manifest, root), ["model.bin"])
            manifest.write_text(json.dumps({"files": []}))
            with self.assertRaisesRegex(deploy.DeploymentError, "no files to verify"):
                deploy.verify_manifest(manifest, root)
            manifest.write_text(json.dumps({"files": [{"path": "../outside",
                "sha256": hashlib.sha256(b"artifact").hexdigest()}]}))
            with self.assertRaisesRegex(deploy.DeploymentError, "Unsafe artifact path"):
                deploy.verify_manifest(manifest, root)
            manifest.write_text(json.dumps({"files": [{"path": "model.bin", "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(deploy.DeploymentError, "SHA-256 mismatch"):
                deploy.verify_manifest(manifest, root)

    def test_deployment_record_has_commit_and_cli_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "kernel-metadata.json"
            notebook = root / deploy.NOTEBOOK.name
            notebook.write_text(json.dumps({"cells": [{"source": ['os.environ["PASTURE_MODE"] = "SMOKE"']}]}))
            metadata.write_text(json.dumps({"id": "sample_user/pasture-studies", "code_file": notebook.name}))
            record = root / "deployment.json"
            commit = "a" * 40
            deploy.write_deployment_record(record, kernel="sample_user/pasture-studies",
                version="2.2.4", kernel_version="7", source_commit=commit, mode="SMOKE", metadata_path=metadata)
            result = json.loads(record.read_text())
            self.assertEqual(result["source_commit"], commit)
            self.assertEqual(result["kaggle_cli_version"], "2.2.4")
            self.assertEqual(result["kaggle_kernel_version"], "7")
            self.assertEqual(result["staged_notebook_sha256"], hashlib.sha256(notebook.read_bytes()).hexdigest())
            self.assertEqual(result["metadata_sha256"], hashlib.sha256(metadata.read_bytes()).hexdigest())
            with self.assertRaises(deploy.DeploymentError):
                deploy.write_deployment_record(record, kernel="k", version="x", source_commit="short",
                    kernel_version="7", mode="SMOKE", metadata_path=metadata)

    @patch.object(deploy, "subprocess")
    def test_cli_failures_suppress_server_error_details(self, subprocess_mock):
        subprocess_mock.run.return_value = type("Result", (), {
            "returncode": 1, "stderr": "Authorization: Bearer secret-token", "stdout": ""})()
        with self.assertRaises(deploy.DeploymentError) as error:
            deploy.run(["kaggle", "kernels", "status", "owner/kernel"])
        self.assertNotIn("secret-token", str(error.exception))

    def test_auth_preflight_explicitly_reports_unverified_without_oauth(self):
        result = deploy.auth_preflight()
        self.assertEqual(result["auth_status"], "not_verified")
        self.assertEqual(result["auth_check"], "not_performed")
        self.assertNotIn("credentials_available", result)

if __name__ == "__main__":
    unittest.main()
