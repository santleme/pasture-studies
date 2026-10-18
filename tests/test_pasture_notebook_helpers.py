from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from validation.benchmark_torch_fixed_corpus import compare_reports, summarize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"


def notebook_helper(name: str, cell_index: int, namespace: dict) -> object:
    """Load one notebook helper without executing the surrounding Kaggle cell."""
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][cell_index]["source"])
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if not matches:
        raise AssertionError(f"{name} is missing from notebook cell {cell_index}")
    module = ast.Module(body=[matches[-1]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


class PastureNotebookHelperTests(unittest.TestCase):
    def test_smoke_runtime_mismatches_use_installed_stack_without_installing_pins(self):
        helper = notebook_helper("should_use_installed_smoke_runtime", 5,
                                 {"print": lambda *_args, **_kwargs: None,
                                  "RuntimeError": RuntimeError})
        mismatch = [{"package": "torch", "scientific_pin": "2.14.0",
                     "runtime_version": "2.11.0+cu128"}]
        self.assertTrue(helper("SMOKE", mismatch, True, True))
        self.assertFalse(helper("PREFLIGHT", mismatch, True, True))
        with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
            helper("SMOKE", mismatch, True, False)

    def test_smoke_job_refuses_stale_checkpoint_instead_of_skipping(self):
        run_mock = Mock()
        namespace = {"Path": Path, "json": json, "sys": sys, "RuntimeError": RuntimeError,
                     "run": run_mock}
        run_smoke_job = notebook_helper("run_smoke_job", 9, namespace)
        with tempfile.TemporaryDirectory() as temp:
            smoke_dir = Path(temp) / "smoke"
            out = smoke_dir / "cpu" / "a2c"
            stale = out / "smoke" / "a2c" / "LATEST"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale-generation")
            with self.assertRaisesRegex(RuntimeError, "pre-existing SMOKE outputs"):
                run_smoke_job("cpu", "a2c", "cpu", smoke_dir=smoke_dir,
                              seed=10, repo_dir=Path(temp))
        run_mock.assert_not_called()

    def test_smoke_job_requires_latest_generation_and_hashed_exports(self):
        def exercise(*, latest="generation-00000001", tamper_hash=False):
            with tempfile.TemporaryDirectory() as temp:
                smoke_dir = Path(temp) / "smoke"
                out = smoke_dir / "cuda" / "a2c"
                generation_root = out / "smoke" / "a2c"
                generation = generation_root / "generation-00000001"
                checkpoint = generation_root / "LATEST"
                npz = generation_root / "policy.npz"
                sidecar = generation_root / "policy.json"

                def fake_run(_argv, *, cwd):
                    del cwd
                    generation.mkdir(parents=True)
                    (generation / "manifest.json").write_text("{}\n")
                    checkpoint.write_text(latest + "\n")
                    (out / "metrics.jsonl").write_text('{"update":1}\n')
                    npz.write_bytes(b"inference-model")
                    sidecar.write_text('{"schema":"test"}\n')
                    manifest = {"training_generation": "generation-00000001",
                        "policy_npz": "policy.npz", "policy_json": "policy.json",
                        "policy_npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                        "policy_json_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}
                    if tamper_hash:
                        manifest["policy_json_sha256"] = "0" * 64
                    (generation_root / "artifact-manifest.json").write_text(json.dumps(manifest))

                namespace = {"Path": Path, "json": json, "hashlib": hashlib, "sys": sys,
                             "RuntimeError": RuntimeError, "run": fake_run}
                helper = notebook_helper("run_smoke_job", 9, namespace)
                result = helper("cuda", "a2c", "cuda", smoke_dir=smoke_dir,
                                seed=11, repo_dir=Path(temp))
                self.assertEqual(result, out)
                return True

        self.assertTrue(exercise())
        with self.assertRaisesRegex(RuntimeError, "sole fresh smoke generation"):
            exercise(latest="missing-generation")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            exercise(tamper_hash=True)

    def test_fixed_corpus_comparison_schema_and_pairing(self):
        left = {"device": "cpu", "workload": "fixed", "corpus_config": {"corpus_sha256": "same"},
                "runs": [{"repeat_index": 1, "duration_seconds": 6.0},
                         {"repeat_index": 0, "duration_seconds": 8.0}]}
        right = {"device": "cuda", "workload": "fixed", "corpus_config": {"corpus_sha256": "same"},
                 "runs": [{"repeat_index": 0, "duration_seconds": 4.0},
                          {"repeat_index": 1, "duration_seconds": 2.0}]}

        result = compare_reports(left, right)
        self.assertEqual(set(result), {"numerator_device", "denominator_device", "paired_elapsed_speedup"})
        self.assertEqual(result["numerator_device"], "cpu")
        self.assertEqual(result["denominator_device"], "cuda")
        speedup = result["paired_elapsed_speedup"]
        self.assertEqual(set(speedup), {"raw", "median", "bootstrap_ci95_mean"})
        self.assertEqual(speedup["raw"], [2.0, 3.0])
        self.assertEqual(speedup["median"], 2.5)
        self.assertEqual(set(speedup["bootstrap_ci95_mean"]), {"lower_95", "upper_95"})
        self.assertTrue(all(isinstance(value, float) for value in speedup["raw"]))
        self.assertIsInstance(speedup["median"], float)
        self.assertTrue(all(isinstance(value, float)
                            for value in speedup["bootstrap_ci95_mean"].values()))
        self.assertLessEqual(speedup["bootstrap_ci95_mean"]["lower_95"],
                             speedup["bootstrap_ci95_mean"]["upper_95"])

    def test_benchmark_summary_has_stable_empty_and_populated_schema(self):
        expected_metrics = {"raw", "median", "p95", "bootstrap_ci95_mean"}
        empty = summarize([])
        self.assertEqual(set(empty), {"repeats", "duration_seconds", "decisions_per_second"})
        self.assertEqual(empty["repeats"], 0)
        for metric in ("duration_seconds", "decisions_per_second"):
            self.assertEqual(set(empty[metric]), expected_metrics)
            self.assertEqual(empty[metric]["raw"], [])
            self.assertIsNone(empty[metric]["median"])
            self.assertEqual(set(empty[metric]["bootstrap_ci95_mean"]), {"lower_95", "upper_95"})

        populated = summarize([
            {"duration_seconds": 2.0, "decisions_per_second": 50.0},
            {"duration_seconds": 4.0, "decisions_per_second": 25.0},
        ])
        self.assertEqual(populated["repeats"], 2)
        self.assertEqual(populated["duration_seconds"]["raw"], [2.0, 4.0])
        self.assertEqual(populated["duration_seconds"]["median"], 3.0)
        self.assertEqual(populated["decisions_per_second"]["raw"], [50.0, 25.0])

    def test_record_run_timing_preserves_original_full_run_on_resume(self):
        namespace = {"Path": Path, "json": json}
        record_run_timing = notebook_helper("record_run_timing", 11, namespace)
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            original = {"resumed": False, "elapsed_seconds": 120.0}
            record_run_timing(outdir, original)
            timing_path = outdir / "run_timing.json"
            original_record = json.loads(timing_path.read_text())
            self.assertTrue(original_record["complete_full_run_timing"])

            resumed = {"resumed": True, "elapsed_seconds": 8.0}
            record_run_timing(outdir, resumed)
            self.assertEqual(json.loads(timing_path.read_text()), original_record)
            segments = [json.loads(line) for line in (outdir / "run_timing_segments.jsonl").read_text().splitlines()]
            self.assertEqual(len(segments), 1)
            self.assertFalse(segments[0]["complete_full_run_timing"])
            self.assertEqual(segments[0]["elapsed_seconds"], None)
            self.assertEqual(segments[0]["elapsed_seconds_current_invocation"], 8.0)

    def test_full_device_report_rejects_smoke_incomplete_and_non_ddp_t4x2(self):
        reports = {"cpu": {"device": "cpu", "torchrun_world_size": 1,
                           "runs": [{"world_size": 1, "smoke_mode": False}]},
                   "t4x1": {"device": "cuda", "torchrun_world_size": 1,
                            "cuda_devices": [{}], "runs": [{"world_size": 1, "smoke_mode": False}]},
                   "t4x2": {"device": "cuda", "torchrun_world_size": 2,
                            "cuda_devices": [{}, {}], "ddp_active": False,
                            "runs": [{"world_size": 2, "smoke_mode": False}]}}
        namespace = {"fixed_reports": reports, "FIXED_REPEATS": 1}
        full_device_report = notebook_helper("full_device_report", 19, namespace)
        self.assertTrue(full_device_report("cpu", "cpu", 1))
        self.assertTrue(full_device_report("t4x1", "cuda", 1))
        self.assertFalse(full_device_report("t4x2", "cuda", 2))
        self.assertFalse(full_device_report("t4x1", "cpu", 1))
        reports["t4x1"]["cuda_devices"] = []
        self.assertFalse(full_device_report("t4x1", "cuda", 1))
        reports["t4x1"]["cuda_devices"] = [{}]
        reports["t4x1"]["runs"][0]["world_size"] = 2
        self.assertFalse(full_device_report("t4x1", "cuda", 1))
        reports["cpu"]["runs"][0]["smoke_mode"] = True
        self.assertFalse(full_device_report("cpu", "cpu", 1))
        reports["cpu"]["runs"] = []
        self.assertFalse(full_device_report("cpu", "cpu", 1))

    def test_compare_fixed_runs_only_when_both_reports_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpu.json").write_text("{}")
            (root / "t4x1.json").write_text("{}")
            run = Mock()

            def write_comparison(command, **kwargs):
                Path(command[command.index("--output") + 1]).write_text('{"paired": true}')

            run.side_effect = write_comparison
            compare_fixed = notebook_helper("compare_fixed", 19, {
                "FIXED_ROOT": root, "run": run, "sys": sys, "Path": Path, "REPO_DIR": root, "json": json,
            })
            result = compare_fixed("cpu", "t4x1", "comparison.json")
            self.assertEqual(result, {"paired": True})
            run.assert_called_once()
            self.assertIsNone(compare_fixed("cpu", "t4x2", "missing.json"))
            run.assert_called_once()

    def test_same_corpus_requires_nonempty_matching_hashes(self):
        namespace = {"hash_by_device": {"cpu": "abc", "t4x1": "abc", "t4x2": "different"}}
        same_corpus = notebook_helper("same_corpus", 19, namespace)
        self.assertTrue(same_corpus("cpu", "t4x1"))
        self.assertFalse(same_corpus("cpu", "t4x2"))
        self.assertFalse(same_corpus("cpu", "missing"))

    def test_speed_helpers_read_the_validated_paired_comparison_schema(self):
        left = {"device": "cpu", "workload": "fixed", "corpus_config": {"corpus_sha256": "same"},
                "runs": [{"repeat_index": i, "duration_seconds": 4.0} for i in range(4)]}
        right = {"device": "cuda", "workload": "fixed", "corpus_config": {"corpus_sha256": "same"},
                 "runs": [{"repeat_index": i, "duration_seconds": 1.0} for i in range(4)]}
        comparison = compare_reports(left, right)
        namespace = {}
        notebook_helper("_comparison_number", 19, namespace)
        paired_speed = notebook_helper("paired_speed", 19, namespace)
        ci_lower = notebook_helper("ci_lower", 19, namespace)
        self.assertEqual(paired_speed(comparison), 4.0)
        self.assertEqual(ci_lower(comparison),
                         comparison["paired_elapsed_speedup"]["bootstrap_ci95_mean"]["lower_95"])
        self.assertTrue(math.isfinite(ci_lower(comparison)))

    def test_speed_helpers_fail_closed_for_missing_null_and_malformed_schema(self):
        namespace = {}
        notebook_helper("_comparison_number", 19, namespace)
        paired_speed = notebook_helper("paired_speed", 19, namespace)
        ci_lower = notebook_helper("ci_lower", 19, namespace)
        malformed_reports = [
            (None, None, None),
            ({}, None, None),
            ({"paired_elapsed_speedup": None}, None, None),
            ({"paired_elapsed_speedup": []}, None, None),
            ({"paired_elapsed_speedup": {"median": 1.5}}, 1.5, None),
            ({"paired_elapsed_speedup": {"bootstrap_ci95_mean": None}}, None, None),
            ({"paired_elapsed_speedup": {"bootstrap_ci95_mean": {"lower_95": None}}}, None, None),
            ({"paired_elapsed_speedup": {"median": "1.5",
                                          "bootstrap_ci95_mean": {"lower_95": "1.2"}}}, None, None),
            ({"paired_elapsed_speedup": {"median": True,
                                          "bootstrap_ci95_mean": {"lower_95": False}}}, None, None),
            ({"paired_elapsed_speedup": {"median": float("nan"),
                                          "bootstrap_ci95_mean": {"lower_95": float("inf")}}}, None, None),
        ]
        for report, expected_median, expected_lower in malformed_reports:
            with self.subTest(report=report):
                self.assertEqual(paired_speed(report), expected_median)
                self.assertEqual(ci_lower(report), expected_lower)

    def test_notebook_keeps_preregistered_timing_thresholds_unchanged(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "".join(notebook["cells"][19]["source"])
        tree = ast.parse(source)
        threshold_assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "compute_thresholds"
                    for target in node.targets)
        )
        threshold_source = ast.get_source_segment(source, threshold_assignment)
        self.assertIn('meets_speedup(comparisons["cpu_vs_t4x1"], 2.0)', threshold_source)
        self.assertIn('>= 1 / 0.70', threshold_source)
        self.assertIn('meets_speedup(comparisons["cpu_vs_t4x2"], 2.0)', threshold_source)
        self.assertIn('meets_speedup(comparisons["t4x1_vs_t4x2"], 1.20)', threshold_source)


if __name__ == "__main__":
    unittest.main()
