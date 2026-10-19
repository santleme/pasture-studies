from __future__ import annotations

import json
from pathlib import Path
import unittest

from validation.benchmark_torch_fixed_corpus import compare_reports, summarize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pasture_studies_kaggle.ipynb"


class PastureNotebookHelperTests(unittest.TestCase):
    @staticmethod
    def comparison_report(*, corpus_hash="same", workload="fixed", repeats=(0, 1)):
        return {
            "device": "cpu",
            "workload": workload,
            "corpus_config": {"corpus_sha256": corpus_hash},
            "runs": [{"repeat_index": index, "duration_seconds": 1.0} for index in repeats],
        }

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

    def test_fixed_corpus_comparison_rejects_different_corpus_or_workload(self):
        left = self.comparison_report()
        with self.subTest("corpus hash"):
            right = self.comparison_report(corpus_hash="different")
            with self.assertRaisesRegex(ValueError, "same frozen corpus hash"):
                compare_reports(left, right)
        with self.subTest("workload"):
            right = self.comparison_report(workload="smoke")
            with self.assertRaisesRegex(ValueError, "same full/smoke workload"):
                compare_reports(left, right)

    def test_fixed_corpus_comparison_rejects_unmatched_repeat_indices(self):
        left = self.comparison_report(repeats=(0, 2))
        right = self.comparison_report(repeats=(0, 1))
        with self.assertRaisesRegex(ValueError, "unmatched repeat indices"):
            compare_reports(left, right)

    def test_fixed_corpus_comparison_rejects_different_repeat_counts(self):
        left = self.comparison_report(repeats=(0, 1))
        right = self.comparison_report(repeats=(0,))
        with self.assertRaisesRegex(ValueError, "same number of repeats"):
            compare_reports(left, right)

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

    def test_notebook_is_readiness_only_and_keeps_other_modes_fail_closed(self):
        notebook = json.loads(NOTEBOOK.read_text())
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn('"READINESS"', sources)
        self.assertIn('"EXPERIMENT", "FULL_VALIDATION"', sources)
        self.assertIn("hard-disabled", sources)
        self.assertNotIn("training_seed_roots", sources)


if __name__ == "__main__":
    unittest.main()
