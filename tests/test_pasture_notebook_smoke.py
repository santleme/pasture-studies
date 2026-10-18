"""Fresh-kernel, top-to-bottom SMOKE validation of the published Kaggle notebook.

Run explicitly with:
  PASTURE_RUN_NOTEBOOK_SMOKE=1 python -m unittest discover -s tests \
    -p test_pasture_notebook_smoke.py -v

This test clones the notebook's immutable public source pin and runs disposable
CPU smoke training only. It never runs experimental/full-validation arms.
"""

from __future__ import annotations

import hashlib
import ast
import json
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"
_pipeline_cell = "\n".join("".join(cell.get("source", []))
                            for cell in json.loads(NOTEBOOK_PATH.read_text())["cells"])
_pipeline_match = re.search(r'^PASTURE_PIPELINE_COMMIT\s*=\s*"([0-9a-f]{40})"$',
                            _pipeline_cell, re.MULTILINE)
if not _pipeline_match:
    raise RuntimeError("Notebook must declare one immutable 40-character pipeline commit")
EXPECTED_SOURCE_COMMIT = _pipeline_match.group(1)


class FreshKernelNotebookSmokeTests(unittest.TestCase):
    @staticmethod
    def execute_notebook_provenance_gate(mode: str, *, revision_claim: str = "",
                                         sha256_claim: str = "", arbitrary_approval: dict | None = None) -> dict:
        """Execute the actual hard-disabled gate with caller-controlled claims."""
        source = nbformat.read(NOTEBOOK_PATH, as_version=4).cells[7].source
        claims_start = source.index("# Caller-provided revision/hash strings are recorded only as unverified claims.")
        prereg_start = source.index("# Generate outcome-blind preregistration", claims_start)
        gate_start = source.index("EXPERIMENT_REQUESTED = MODE", prereg_start)
        gate_end = source.index("dependency_versions =", gate_start)
        claims_source = source[claims_start:prereg_start]
        gate_source = source[gate_start:gate_end]
        namespace = {
            "MODE": mode,
            "EXECUTION_NOTEBOOK_REVISION_CLAIM": revision_claim,
            "EXECUTION_NOTEBOOK_SHA256_CLAIM": sha256_claim,
            "json": json,
            "hashlib": hashlib,
            # Deliberately self-asserted caller-controlled values: these must not
            # turn into a trusted approval or affect any execution flag.
            "PASTURE_EXECUTION_CONFIG_PATH": "/caller/controlled/config.json",
            "EXECUTION_CONFIG": {"status": "APPROVED", "revision": revision_claim},
            "APPROVAL_ARTIFACT": arbitrary_approval,
            "EXECUTION_APPROVAL_SHA256": "f" * 64,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(claims_source, str(NOTEBOOK_PATH), "exec"), namespace)
            exec(compile(gate_source, str(NOTEBOOK_PATH), "exec"), namespace)
        return namespace

    def test_caller_claims_cannot_enable_experimental_modes(self) -> None:
        adversarial_claims = {
            "revision_claim": "self-declared-kaggle-revision",
            "sha256_claim": "a" * 64,
            "arbitrary_approval": {"status": "APPROVED", "approved_by": "self", "approval_id": "fake"},
        }
        missing_approval = {"revision_claim": "", "sha256_claim": "", "arbitrary_approval": None}
        for mode in ("EXPERIMENT", "FULL_VALIDATION"):
            for case, claims in (("self-asserted", adversarial_claims), ("missing approval", missing_approval)):
                with self.subTest(mode=mode, case=case):
                    namespace = self.execute_notebook_provenance_gate(mode, **claims)
                    self.assertFalse(namespace["RUN_EXPERIMENT"])
                    self.assertFalse(namespace["RUN_FULL"])
                    self.assertFalse(namespace["RUN_GPU_BENCHMARK"])
                    self.assertFalse(namespace["RUN_FIXED_BENCHMARK"])
                    self.assertEqual(namespace["NOTEBOOK_PROVENANCE_STATUS"],
                                     "BLOCKED_PENDING_TRUSTED_SOURCE_BOUND_APPROVAL")
                    self.assertEqual(namespace["EXECUTION_APPROVAL_STATUS"],
                                     "BLOCKED_PENDING_TRUSTED_SOURCE_BOUND_APPROVAL")

        for mode in ("SMOKE", "PREFLIGHT"):
            with self.subTest(mode=mode):
                namespace = self.execute_notebook_provenance_gate(mode, **adversarial_claims)
                self.assertFalse(namespace["RUN_EXPERIMENT"])
                self.assertFalse(namespace["RUN_FULL"])
                self.assertFalse(namespace["RUN_GPU_BENCHMARK"])
                self.assertFalse(namespace["RUN_FIXED_BENCHMARK"])
        notebook_init = "".join(nbformat.read(NOTEBOOK_PATH, as_version=4).cells[1].source)
        self.assertIn('MODE = os.environ.get("PASTURE_MODE", "SMOKE").strip().upper()', notebook_init)

    def test_train_helper_resumes_latest_in_same_run_directory(self) -> None:
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        # Extract the actual notebook function, not a duplicated approximation.
        # This isolated test calls a mocked process runner and never trains.
        cell_source = notebook.cells[11].source
        module = ast.parse(cell_source)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_one"
        )
        isolated = ast.Module(body=[function], type_ignores=[])
        namespace = {
            "Path": Path,
            "sys": sys,
            "time": __import__("time"),
            "EPISODES_PER_SEED": 278,
            "CHECKPOINT_EVERY": 16,
            "commit": EXPECTED_SOURCE_COMMIT,
            "record_run_timing": lambda *_args, **_kwargs: None,
        }
        exec(compile(isolated, str(NOTEBOOK_PATH), "exec"), namespace)
        with tempfile.TemporaryDirectory(prefix="pasture-resume-helper-") as temp:
            run_dir = Path(temp) / "cpu" / "a2c" / "seed-71001"
            (run_dir / "a2c").mkdir(parents=True)
            (run_dir / "a2c" / "LATEST").write_text("generation-00000002\n")
            expected_metrics = run_dir / "metrics.jsonl"

            observed = []

            def recording_run(argv):
                observed.append(list(map(str, argv)))
                expected_metrics.write_text('{"update": 3}\n')

            namespace["run"] = recording_run
            returned = namespace["train_one"]("a2c", "cpu", 71001, run_dir)
            self.assertEqual(returned, expected_metrics)
            self.assertEqual(len(observed), 1)
            argv = observed[0]
            self.assertIn("--resume", argv)
            self.assertEqual(argv[argv.index("--resume") + 1], "latest")
            self.assertEqual(argv[argv.index("--output") + 1], str(run_dir))
            self.assertEqual(argv[argv.index("--metrics") + 1], str(expected_metrics))
            self.assertEqual(argv[argv.index("--algorithm") + 1], "a2c")
            self.assertEqual(argv[argv.index("--device") + 1], "cpu")
            self.assertEqual(argv[argv.index("--seed") + 1], "71001")

    @unittest.skipUnless(
        os.environ.get("PASTURE_RUN_NOTEBOOK_SMOKE") == "1",
        "opt-in end-to-end notebook execution; see test module docstring",
    )
    def test_top_to_bottom_smoke_exports_verifiable_artifacts(self) -> None:
        self.assertTrue(shutil.which("git"), "git is required for immutable source checkout")
        self.assertTrue(NOTEBOOK_PATH.is_file())

        with tempfile.TemporaryDirectory(prefix="pasture-notebook-smoke-") as temp:
            temp_root = Path(temp)
            work_root = temp_root / "work"
            work_root.mkdir()
            run_id = f"ci-smoke-{os.getpid()}"

            notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.execution_count = None
                    cell.outputs = []

            # Use an explicit kernelspec pointing at this test interpreter so
            # the notebook runs in a new process/kernel with the pinned test
            # packages and the repository's installed CPU PyTorch.
            kernels_dir = temp_root / "jupyter" / "kernels" / "pasture-test"
            kernels_dir.mkdir(parents=True)
            kernel_spec = KernelSpec(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Pasture isolated test kernel",
                language="python",
            )
            (kernels_dir / "kernel.json").write_text(kernel_spec.to_json())
            env = {
                **os.environ,
                "PASTURE_MODE": "SMOKE",
                "PASTURE_WORK_ROOT": str(work_root),
                "PASTURE_RUN_ID": run_id,
                "PASTURE_NOTEBOOK_REVISION": "",
                "PASTURE_NOTEBOOK_CONTENT_SHA256": "",
                "PASTURE_REQUIRE_CUDA": "0",
                "CUDA_VISIBLE_DEVICES": "",
                "MPLBACKEND": "Agg",
                "JUPYTER_PATH": str(temp_root / "jupyter"),
            }
            # NotebookClient starts a genuinely fresh kernel using the explicit
            # kernelspec above. patch.dict scopes the workload variables to the
            # launch so this runner's process environment is left unchanged.
            with patch.dict(os.environ, env, clear=True):
                kernel_manager = KernelManager(kernel_name="pasture-test")
                client = NotebookClient(
                    notebook,
                    km=kernel_manager,
                    timeout=900,
                    resources={"metadata": {"path": str(temp_root)}},
                    allow_errors=False,
                )
                executed = client.execute()

            artifact_roots = list(work_root.glob(f"pasture-training-artifacts-{run_id[:39]}-s-*"))
            self.assertEqual(len(artifact_roots), 1, "SMOKE must create a fresh isolated output namespace")
            artifact_root = artifact_roots[0]

            errors = [
                output.get("evalue", "")
                for cell in executed.cells
                if cell.cell_type == "code"
                for output in cell.get("outputs", [])
                if output.output_type == "error"
            ]
            self.assertEqual(errors, [], f"notebook execution errors: {errors}")
            self.assertTrue(
                all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code"),
                "notebook did not execute every code cell top-to-bottom",
            )
            self.assertTrue(artifact_root.is_dir(), "notebook did not create its run directory")

            source_record = artifact_root / "source_commit.txt"
            self.assertTrue(source_record.is_file())
            self.assertIn(f"commit={EXPECTED_SOURCE_COMMIT}", source_record.read_text())

            prereg_path = artifact_root / "preregistration.json"
            binding_path = artifact_root / "preregistration_binding.json"
            self.assertTrue(prereg_path.is_file())
            self.assertTrue(binding_path.is_file())
            prereg = json.loads(prereg_path.read_text())
            binding = json.loads(binding_path.read_text())
            manifest_hash = prereg["manifest_sha256"]
            self.assertEqual(prereg["status"], "PRE_OUTCOME_PREREGISTRATION_PENDING_PILOT_DELTA0")
            self.assertEqual(binding["manifest_sha256"], manifest_hash)
            blocked = "BLOCKED_PENDING_TRUSTED_SOURCE_BOUND_APPROVAL"
            self.assertEqual(binding["execution_notebook_provenance_status"], blocked)
            self.assertIsNone(binding["execution_notebook_revision_claim_unverified"])
            self.assertIsNone(binding["execution_notebook_sha256_claim_unverified"])
            self.assertEqual(binding["execution_approval_status"], blocked)
            self.assertIsNone(binding["execution_approval_sha256"])
            self.assertFalse(binding["experiment_eligible"])
            self.assertTrue(binding["experiment_block_reason"])

            for algorithm in ("reinforce", "a2c"):
                output = artifact_root / "smoke" / "cpu" / algorithm
                latest = output / "smoke" / algorithm / "LATEST"
                metrics = output / "metrics.jsonl"
                self.assertTrue(latest.is_file(), f"missing {algorithm} smoke checkpoint pointer")
                generation_name = latest.read_text().strip()
                self.assertTrue((latest.parent / generation_name).is_dir(), f"invalid {algorithm} checkpoint pointer")
                self.assertTrue(metrics.is_file(), f"missing {algorithm} metrics")
                self.assertGreater(metrics.stat().st_size, 0)
                metric_rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                self.assertTrue(metric_rows, f"{algorithm} metrics contain no rows")

            export_dir = artifact_root / "export"
            manifest_path = export_dir / "artifact_manifest.json"
            archive_path = export_dir / "pasture-training-artifacts.zip"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(archive_path.is_file())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["commit"], EXPECTED_SOURCE_COMMIT)
            self.assertEqual(manifest["execution_mode"], "SMOKE")
            self.assertEqual(manifest["notebook_provenance_status"], blocked)
            self.assertEqual(manifest["execution_approval_status"], blocked)
            self.assertFalse(manifest["experiment_eligibility"], "SMOKE must not authorize experiment runs")
            self.assertTrue(manifest["experiment_block_reason"])
            self.assertIsNone(manifest["notebook_revision_claim_unverified"])
            self.assertIsNone(manifest["notebook_content_sha256_claim_unverified"])
            self.assertIsNone(manifest["execution_approval_sha256"])
            self.assertEqual(manifest["preregistration_sha256"], manifest_hash)
            self.assertEqual(manifest["acceptance_thresholds_sha256"], binding["acceptance_thresholds_sha256"])
            self.assertEqual(manifest["cuda_forward_backward_smoke"], "UNAVAILABLE")
            self.assertFalse(manifest["cuda_available"], "CPU-only test must not claim CUDA availability")
            self.assertEqual(manifest["cuda_device_count"], 0)
            self.assertEqual(manifest["cuda_device_names"], [])
            self.assertEqual(manifest["scientific_dependency_pins"], {"numpy": "1.26.4", "torch": "2.14.0"})
            self.assertIsInstance(manifest["runtime_dependency_mismatches"], list)
            self.assertFalse((artifact_root / "fixed_corpus_benchmark" / "kaggle-a2c-fixed-corpus-v1.npz").exists())
            self.assertFalse(any((artifact_root / "matched_cpu").rglob("metrics.jsonl")))

            entries = {entry["path"]: entry for entry in manifest["files"]}
            self.assertIn("preregistration.json", entries)
            self.assertIn("smoke/cpu/reinforce/metrics.jsonl", entries)
            self.assertIn("smoke/cpu/a2c/metrics.jsonl", entries)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertTrue(archive.testzip() is None, "artifact archive CRC validation failed")
                members = set(archive.namelist())
                self.assertIn("export/artifact_manifest.json", members)
                for name, entry in entries.items():
                    self.assertIn(name, members, f"manifest artifact missing from archive: {name}")
                    payload = archive.read(name)
                    self.assertEqual(len(payload), entry["size_bytes"], f"size mismatch: {name}")
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        entry["sha256"],
                        f"SHA-256 mismatch: {name}",
                    )


if __name__ == "__main__":
    unittest.main()
