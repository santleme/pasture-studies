from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from validation.benchmark_torch_fixed_corpus import summarize


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


if __name__ == "__main__":
    unittest.main()
