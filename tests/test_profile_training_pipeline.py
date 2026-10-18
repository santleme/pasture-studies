from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from validation.profile_training_pipeline import profile

try:
    import torch
except ImportError:
    torch = None


class TrainingPipelineProfileTests(unittest.TestCase):
    def test_small_profile_records_raw_cpu_phases_and_explicit_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            report = profile(seed=42, ticks=2, repeats=1, output=output)
            persisted = json.loads(output.read_text())
            self.assertEqual(report["configuration"]["seed"], 42)
            self.assertEqual(report["environment"]["device"], "cpu")
            self.assertFalse(report["environment"]["cuda_tested"])
            timings = persisted["timings"]
            for phase in ("world_construction", "perception", "memory_retrieval", "feature_encoding",
                          "policy_inference", "navigation_arbitration", "physiology_reward",
                          "simulator_step_inclusive", "full_rollout", "world_serialization",
                          "json_log_serialization", "json_log_file_write",
                          "policy_checkpoint_save_npz_and_sidecar", "policy_checkpoint_load_npz_and_sidecar"):
                self.assertIn(phase, timings)
                self.assertTrue(timings[phase]["samples_seconds"])
            self.assertEqual(timings["simulator_step_inclusive"]["calls"], 2)
            self.assertEqual(timings["policy_inference"]["calls"], 10)
            self.assertEqual(timings["navigation_arbitration"]["calls"], 10)
            self.assertTrue(report["source_sha256"])
            self.assertNotIn("scalar-versus-batched CPU collection", report["coverage_and_limits"]["not_measured"])
            self.assertIn("multi-world-batched CPU collection", report["coverage_and_limits"]["comparison_note"])
            self.assertTrue(report["collector_comparison"]["parity_pass"])
            self.assertFalse(list(Path(directory).glob("*.policy-*.npz")))

    def test_profile_rejects_nonpositive_workload(self):
        with self.assertRaises(ValueError):
            profile(ticks=0)

    @unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
    def test_scalar_and_batched_collectors_share_stream_and_exact_world_state(self):
        from validation.profile_training_pipeline import benchmark_collectors

        result = benchmark_collectors(seeds=(97531,), root_seed=4421, repeats=1, model_seed=18)
        self.assertTrue(result["parity_pass"], result["paired_parity"])
        self.assertEqual(result["workload"]["ticks_per_episode"], 72)
        self.assertEqual(result["workload"]["decisions_per_repeat"], 360)
        for name in ("scalar_per_animal_inference", "batched_multiworld_inference"):
            row = result[name]
            self.assertEqual(len(row["samples_seconds"]), 1)
            self.assertGreater(row["median_seconds"], 0)
            self.assertGreater(row["p95_seconds"], 0)
            self.assertGreater(row["decisions_per_second_at_median"], 0)
        parity = result["paired_parity"][0]
        self.assertTrue(all(parity["trajectory_fields_exact"].values()))
        self.assertTrue(parity["final_world_state_exact_except_probability_telemetry"])
        self.assertTrue(parity["events_exact"])
        self.assertTrue(parity["probabilities_allclose_rtol_1e-6_atol_1e-7"])


if __name__ == "__main__":
    unittest.main()
