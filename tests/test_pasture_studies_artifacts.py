from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from validation.package_torch_corpus import package_corpus, verify_package


class PastureStudiesArtifactsTests(unittest.TestCase):
    def test_dataset_package_records_and_verifies_local_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corpus.npz"
            source.write_bytes(b"test archive")
            source.with_suffix(".json").write_text("{}\n")
            arrays = {"world_ids": np.arange(2780).reshape(10, 278)}
            config = {"smoke_mode": False,
                      "split": "training-only fixed workload; never held-out; stale off-policy compute replay",
                      "partition_count": 10, "episodes_per_partition": 278,
                      "corpus_sha256": "f" * 64}
            output = root / "release"
            with patch("validation.package_torch_corpus.load_corpus", return_value=(arrays, config)):
                manifest = package_corpus(source, output, dataset_id="owner/pasture-data",
                    title="Pasture training corpus", license_name="CC0-1.0", data_version="v1")
            self.assertEqual(manifest["split"], "training-only")
            self.assertEqual(verify_package(output)["manifest_sha256"], manifest["manifest_sha256"])
            metadata = json.loads((output / "dataset-metadata.json").read_text())
            self.assertEqual(metadata["licenses"], [{"name": "CC0-1.0"}])
            (output / "pasture-studies-v1.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                verify_package(output)


if __name__ == "__main__":
    unittest.main()
