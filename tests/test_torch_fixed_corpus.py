from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from validation.benchmark_torch_fixed_corpus import (
    SOURCE_FILES,
    array_sha,
    bootstrap_interval,
    corpus_config,
    file_sha,
    load_corpus,
    save_corpus,
    summarize,
)


def _frozen_arrays():
    # Small deterministic data fixture: no simulator rollout or policy update.
    shape = (2, 2, 72, 5)
    return {
        "observations": np.zeros(shape + (20,), dtype=np.float32),
        "actions": np.zeros(shape, dtype=np.int64),
        "rewards": np.zeros(shape, dtype=np.float32),
        "returns_to_go": np.zeros(shape, dtype=np.float32),
        "masks": np.ones(shape, dtype=np.uint8),
        "world_ids": np.asarray([[101, 102], [103, 104]], dtype=np.int64),
    }


def _frozen_config(arrays):
    config = corpus_config(root_seed=501, smoke=True)
    config["array_sha256"] = {key: array_sha(value) for key, value in arrays.items()}
    config["corpus_sha256"] = hashlib.sha256(
        json.dumps(config["array_sha256"], sort_keys=True).encode()
    ).hexdigest()
    root = Path(__file__).resolve().parents[1]
    config["source_sha256"] = {name: file_sha(root / name) for name in SOURCE_FILES}
    config["runtime"] = {"python": "fixture", "numpy": np.__version__, "torch": "fixture"}
    config["package_lock_sha256"] = file_sha(root / "requirements-training.txt")
    config["behavior_policy_sha256"] = "0" * 64
    config["environment_semantics"] = {
        "ticks_per_day": 24, "ticks_per_episode": 72,
        "animals_per_world": 5,
        "reward_source": "unchanged farm_sim.physiology.step_physiology",
    }
    config["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return config


class TorchFixedCorpusContractTests(unittest.TestCase):
    def test_frozen_arrays_round_trip_with_hash_and_shape_contracts(self):
        arrays = _frozen_arrays()
        config = _frozen_config(arrays)
        with tempfile.TemporaryDirectory() as directory:
            path = save_corpus(Path(directory) / "fixture.npz", arrays, config)
            loaded, loaded_config = load_corpus(path)
        self.assertEqual(loaded_config["corpus_sha256"], config["corpus_sha256"])
        for name in arrays:
            self.assertEqual(array_sha(loaded[name]), array_sha(arrays[name]))

    def test_summary_and_bootstrap_operate_on_frozen_metrics(self):
        result = summarize([{"duration_seconds": value,
                             "decisions_per_second": 100 / value}
                            for value in (1., 2., 3., 4., 5.)])
        self.assertEqual(result["duration_seconds"]["median"], 3)
        self.assertLessEqual(bootstrap_interval([1., 2., 3., 4., 5.])["lower_95"],
                             bootstrap_interval([1., 2., 3., 4., 5.])["upper_95"])

    def test_loader_rejects_tampered_frozen_arrays(self):
        arrays = _frozen_arrays()
        config = _frozen_config(arrays)
        with tempfile.TemporaryDirectory() as directory:
            path = save_corpus(Path(directory) / "fixture.npz", arrays, config)
            arrays["masks"][0, 0, 0, 0] = 0
            save_corpus(path, arrays, config)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
