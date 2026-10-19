from __future__ import annotations

from validation.opt_in_training_tests import require_opt_in_training_tests

require_opt_in_training_tests()

from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:  # optional training dependency
    torch = None

from validation.benchmark_torch_fixed_corpus import (array_sha, bootstrap_interval,
    build_corpus, corpus_config, load_corpus, replay_once, save_corpus, summarize)


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TorchFixedCorpusTests(unittest.TestCase):
    def test_smoke_corpus_has_complete_episodes_all_one_masks_and_frozen_hashes(self):
        arrays, config = build_corpus(501, smoke=True)
        self.assertEqual(arrays["observations"].shape, (2, 2, 72, 5, 20))
        self.assertEqual(arrays["actions"].shape, (2, 2, 72, 5))
        self.assertEqual(arrays["rewards"].shape, (2, 2, 72, 5))
        self.assertEqual(arrays["returns_to_go"].shape, (2, 2, 72, 5))
        self.assertEqual(arrays["masks"].shape, (2, 2, 72, 5))
        self.assertTrue(np.all(arrays["masks"] == 1))
        self.assertEqual(len(set(arrays["world_ids"].reshape(-1).tolist())), 4)
        self.assertTrue(config["smoke_mode"])
        self.assertTrue(config["corpus_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            target = save_corpus(Path(directory) / "corpus.npz", arrays, config)
            loaded, loaded_config = load_corpus(target)
            self.assertEqual(loaded_config["corpus_sha256"], config["corpus_sha256"])
            for key in arrays:
                self.assertEqual(array_sha(loaded[key]), array_sha(arrays[key]))

    def test_fixed_corpus_generation_is_reproducible_for_smoke(self):
        one, one_config = build_corpus(502, smoke=True)
        two, two_config = build_corpus(502, smoke=True)
        self.assertEqual(one_config["array_sha256"], two_config["array_sha256"])
        self.assertEqual(one_config["corpus_sha256"], two_config["corpus_sha256"])
        for key in one:
            self.assertTrue(np.array_equal(one[key], two[key]))

    def test_loader_rejects_mask_or_hash_corruption(self):
        arrays, config = build_corpus(503, smoke=True)
        arrays["masks"][0, 0, 2, 0] = 0
        with tempfile.TemporaryDirectory() as directory:
            target = save_corpus(Path(directory) / "bad.npz", arrays, config)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_corpus(target)

    def test_loader_rejects_tampered_corpus_configuration(self):
        arrays, config = build_corpus(5031, smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            target = save_corpus(Path(directory) / "tamper-config.npz", arrays, config)
            sidecar = target.with_suffix(".json")
            import json
            edited = json.loads(sidecar.read_text())
            edited["root_seed"] += 1
            sidecar.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                load_corpus(target)

    def test_smoke_replay_executes_real_cpu_optimizer_updates_and_reports_work(self):
        arrays, _ = build_corpus(504, smoke=True)
        row = replay_once(arrays, seed=9001, device_name="cpu", repeat_index=0, smoke=True)
        self.assertEqual(row["updates"], 2)
        self.assertEqual(row["decisions"], 2 * 2 * 72 * 5)
        self.assertGreater(row["duration_seconds"], 0)
        self.assertGreater(row["decisions_per_second"], 0)
        self.assertTrue(all("actor_loss" in step and "gradient_norm_preclip" in step for step in row["step_rows"]))
        self.assertIn("backward", row["computation"])
        self.assertIn("Adam update", row["computation"])

    def test_summary_retains_raw_runs_and_bootstrap_intervals(self):
        rows = [{"duration_seconds": x, "decisions_per_second": 100 / x} for x in (1., 2., 3., 4., 5.)]
        result = summarize(rows)
        self.assertEqual(result["duration_seconds"]["raw"], [1., 2., 3., 4., 5.])
        self.assertEqual(result["duration_seconds"]["median"], 3)
        self.assertAlmostEqual(result["duration_seconds"]["p95"], 4.8)
        ci = bootstrap_interval([1., 2., 3., 4., 5.])
        self.assertLessEqual(ci["lower_95"], ci["upper_95"])

    def test_full_schedule_is_exactly_seventeen_full_updates_plus_tail(self):
        config = corpus_config(root_seed=5, smoke=False)
        self.assertEqual(config["partition_count"], 10)
        self.assertEqual(config["episodes_per_partition"], 278)
        self.assertEqual(config["episodes_per_partition"] // 16, 17)
        self.assertEqual(config["episodes_per_partition"] % 16, 6)


if __name__ == "__main__":
    unittest.main()
