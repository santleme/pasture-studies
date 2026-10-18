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
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["dataset_sources"], [])
        with self.assertRaises(deploy.DeploymentError):
            deploy.metadata_for("not/a/user")

    def test_prepare_copies_notebook_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "kernel"
            metadata_path = deploy.prepare(output, "sample_user", mode="PREFLIGHT")
            self.assertTrue((output / deploy.NOTEBOOK.name).is_file())
            self.assertEqual(json.loads(metadata_path.read_text())["id"], "sample_user/pasture-studies")
            with self.assertRaises(deploy.DeploymentError):
                deploy.prepare(output, "sample_user", mode="EXPERIMENT")

    @patch.object(deploy, "run")
    def test_publish_fails_closed_without_no_run_and_never_calls_push(self, run_mock):
        run_mock.side_effect = [
            type("Result", (), {"stdout": "Kaggle CLI 2.2.4", "stderr": ""})(),
            type("Result", (), {"stdout": "usage: kaggle kernels push [-h] [-p PATH]", "stderr": ""})(),
        ]
        with self.assertRaisesRegex(deploy.DeploymentError, "refusing push"):
            deploy.publish(Path("prepared"))
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args.args[0], ["kaggle", "kernels", "push", "--help"])

    @patch.object(deploy, "run")
    def test_publish_uses_no_run_only_when_advertised(self, run_mock):
        run_mock.side_effect = [
            type("Result", (), {"stdout": "Kaggle CLI 9.0.0", "stderr": ""})(),
            type("Result", (), {"stdout": "--no-run Do not execute", "stderr": ""})(),
            type("Result", (), {"stdout": "submitted", "stderr": ""})(),
        ]
        deploy.publish(Path("prepared"))
        self.assertEqual(run_mock.call_args.args[0],
                         ["kaggle", "kernels", "push", "--path", "prepared", "--no-run"])

    @patch.object(deploy, "run")
    def test_status_logs_and_outputs_use_official_cli_commands(self, run_mock):
        run_mock.return_value = type("Result", (), {"stdout": "ok", "stderr": ""})()
        for kind, expected in (("status", "status"), ("logs", "logs"), ("output", "output")):
            with self.subTest(kind=kind):
                self.assertEqual(deploy.retrieve(kind, "sample_user/pasture-studies"), "ok")
                self.assertEqual(run_mock.call_args.args[0][:3], ["kaggle", "kernels", expected])
        with self.assertRaises(deploy.DeploymentError):
            deploy.retrieve("status", "bad kernel")

    def test_manifest_verifies_hash_size_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.bin"
            artifact.write_bytes(b"artifact")
            manifest = root / "artifact_manifest.json"
            manifest.write_text(json.dumps({"files": [{"path": "model.bin",
                "sha256": hashlib.sha256(b"artifact").hexdigest(), "size_bytes": 8}]}))
            self.assertEqual(deploy.verify_manifest(manifest, root), ["model.bin"])
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
            metadata.write_text('{"id":"sample_user/pasture-studies"}')
            record = root / "deployment.json"
            commit = "a" * 40
            deploy.write_deployment_record(record, kernel="sample_user/pasture-studies",
                version="2.2.4", source_commit=commit, mode="SMOKE", metadata_path=metadata)
            result = json.loads(record.read_text())
            self.assertEqual(result["source_commit"], commit)
            self.assertEqual(result["kaggle_cli_version"], "2.2.4")
            self.assertEqual(result["metadata_sha256"], hashlib.sha256(metadata.read_bytes()).hexdigest())
            with self.assertRaises(deploy.DeploymentError):
                deploy.write_deployment_record(record, kernel="k", version="x", source_commit="short",
                    mode="SMOKE", metadata_path=metadata)

    @patch.object(deploy, "KAGGLE_CONFIG", Path("/does/not/exist"))
    def test_auth_preflight_does_not_start_oauth_and_detects_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(deploy.DeploymentError, "never starts OAuth"):
            deploy.auth_preflight()
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "test-token"}, clear=True):
            self.assertEqual(deploy.auth_preflight()["credentials_source"], "environment")


if __name__ == "__main__":
    unittest.main()
