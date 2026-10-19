"""Fresh-kernel execution and fail-closed checks for the Kaggle launcher.

Run explicitly with:
  PASTURE_RUN_NOTEBOOK_SMOKE=1 python -m unittest discover -s tests \
    -p test_pasture_notebook_smoke.py -v

The e2e case runs only disposable CPU SMOKE. READINESS is source/release-bound
and CUDA-only; it is verified structurally here and is never run from local CI.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import kaggle_deploy as deploy

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/pasture_studies_kaggle.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text())


def _code_sources(notebook):
    return ["".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"]


class KaggleNotebookReadinessTests(unittest.TestCase):
    def test_launcher_does_not_accept_env_claims_and_hard_blocks_experiments(self):
        source = "\n".join(_code_sources(NOTEBOOK))
        self.assertNotIn("PASTURE_NOTEBOOK_REVISION", source)
        self.assertNotIn("PASTURE_NOTEBOOK_CONTENT_SHA256", source)
        self.assertIn('if MODE in {"EXPERIMENT", "FULL_VALIDATION"}', source)
        self.assertIn("authorize_readiness", source)
        self.assertIn("PASTURE_DEPLOYMENT = None  # KAGGLE_DEPLOYMENT_CONFIG_MARKER", source)
        self.assertNotIn("seed_splits", source)
        self.assertNotIn("locked_test_seed", source)

    def test_readiness_has_bounded_recovery_ddp_and_no_evaluation(self):
        source = "\n".join(_code_sources(NOTEBOOK))
        for required in ("--readiness-ddp", "--nproc_per_node=2", "--max-updates", "--resume", "latest",
                        "diagnostic_seed = 37331", "evaluation_performed", "locked_split_accessed",
                        "cuda_device_index", "_is_complete_generation"):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertIn('"benchmark_eligible": False', source)

    def test_caller_mode_cannot_enable_experiment(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        marker = "PASTURE_DEPLOYMENT = None  # KAGGLE_DEPLOYMENT_CONFIG_MARKER"
        cell = next(c for c in notebook["cells"] if marker in "".join(c.get("source", [])))
        source = "".join(cell["source"])
        source = source.replace(marker, "PASTURE_DEPLOYMENT = {\"mode\": \"EXPERIMENT\"}", 1)
        cell["source"] = source.splitlines(keepends=True)
        code = "".join(cell["source"])
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                os.chdir(tmp)
                with self.assertRaisesRegex(RuntimeError, "Experimental execution is hard-disabled"):
                    exec(compile(code, str(NOTEBOOK_PATH), "exec"), {})
            finally:
                os.chdir(cwd)

    @unittest.skipUnless(os.environ.get("PASTURE_RUN_NOTEBOOK_SMOKE") == "1",
                         "opt-in fresh-kernel CPU SMOKE; see module docstring")
    def test_fresh_kernel_cpu_smoke_exports_hash_verified_artifacts(self):
        self.assertTrue(shutil.which("git"))
        with tempfile.TemporaryDirectory(prefix="pasture-notebook-smoke-") as temp:
            tmp = Path(temp)
            work = tmp / "work"
            work.mkdir()
            run_id = ""  # The notebook supplies a fresh unpredictable run identifier.
            source_sha = "a" * 40
            stage_dir = tmp / "staged"
            with patch.object(deploy, "_source_commit", return_value=source_sha), \
                    patch.object(deploy, "_committed_notebook_bytes", return_value=NOTEBOOK_PATH.read_bytes()):
                deploy.prepare(stage_dir, "test_owner", mode="SMOKE")
            staged_notebook = stage_dir / NOTEBOOK_PATH.name

            # Test a fresh checkout from an isolated source snapshot without network
            # or mutable GitHub state. The wrapper models only clone/fetch/checkout.
            source_snapshot = tmp / "source-snapshot"
            source_snapshot.mkdir()
            for name in ("farm_sim", "validation"):
                shutil.copytree(ROOT / name, source_snapshot / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(ROOT / "scripts", source_snapshot / "scripts",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (source_snapshot / "notebooks").mkdir()
            shutil.copy2(NOTEBOOK_PATH, source_snapshot / "notebooks/pasture_studies_kaggle.ipynb")
            shutil.copy2(ROOT / "requirements-training.txt", source_snapshot / "requirements-training.txt")

            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            git_wrapper = bin_dir / "git"
            git_wrapper.write_text("""#!/usr/bin/env python3
import os, shutil, sys
args = sys.argv[1:]
if args and args[0] == 'clone':
    shutil.copytree(os.environ['PASTURE_TEST_SOURCE'], args[-1])
    raise SystemExit(0)
if len(args) >= 3 and args[0] == '-C' and args[2] in {'fetch', 'checkout'}:
    raise SystemExit(0)
if len(args) >= 3 and args[0] == '-C' and args[2:] == ['rev-parse', 'HEAD']:
    print(os.environ['PASTURE_TEST_COMMIT'])
    raise SystemExit(0)
raise SystemExit('unexpected git wrapper args: ' + repr(args))
""")
            git_wrapper.chmod(0o755)
            env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
                "PASTURE_TEST_SOURCE": str(source_snapshot), "PASTURE_TEST_COMMIT": source_sha,
                "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
            kernels = tmp / "jupyter/kernels/pasture-test"
            kernels.mkdir(parents=True)
            from jupyter_client import KernelManager
            from jupyter_client.kernelspec import KernelSpec
            import nbformat
            from nbclient import NotebookClient

            (kernels / "kernel.json").write_text(KernelSpec(
                argv=[os.sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Pasture smoke isolated", language="python").to_json())
            notebook = nbformat.read(staged_notebook, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.execution_count = None
                    cell.outputs = []
            env["JUPYTER_PATH"] = str(tmp / "jupyter")
            with patch.dict(os.environ, env, clear=True):
                manager = KernelManager(kernel_name="pasture-test")
                executed = NotebookClient(notebook, km=manager, timeout=600,
                    resources={"metadata": {"path": str(tmp)}, "path": str(tmp)},
                    allow_errors=False).execute()
            errors = [out.get("evalue", "") for cell in executed.cells if cell.cell_type == "code"
                      for out in cell.get("outputs", []) if out.output_type == "error"]
            self.assertEqual(errors, [], f"fresh notebook errors: {errors}")
            run_dirs = list(tmp.glob("pasture-release-smoke-*"))
            self.assertEqual(len(run_dirs), 1)
            artifact_root = run_dirs[0]
            manifest = json.loads((artifact_root / "artifact-manifest.json").read_text())
            self.assertEqual(manifest["mode"], "SMOKE")
            self.assertFalse(manifest["benchmark_eligible"])
            self.assertFalse(manifest["locked_outcomes_accessed"])
            self.assertGreaterEqual(len(deploy.verify_manifest(artifact_root / "artifact-manifest.json", artifact_root)), 1)
            archive = artifact_root / "pasture-artifacts.zip"
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as zf:
                self.assertIsNone(zf.testzip())
                for entry in manifest["files"]:
                    self.assertIn(entry["path"], zf.namelist())


if __name__ == "__main__":
    unittest.main()
