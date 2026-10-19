from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.kaggle_dataset_publish import command_for, publish
from validation.pasture_studies_dataset import generate_bundle


class KaggleDatasetPublishTests(unittest.TestCase):
    def test_builds_official_create_and_version_command_shapes(self):
        path = Path("/tmp/pasture-data")
        self.assertEqual(command_for("create", path), ["kaggle", "datasets", "create", "-p", str(path)])
        self.assertEqual(command_for("version", path, "update v2"),
                         ["kaggle", "datasets", "version", "-p", str(path), "-m", "update v2"])
        with self.assertRaisesRegex(ValueError, "version operation"):
            command_for("version", path)

    def test_default_is_verified_dry_run_and_never_calls_kaggle(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "data"
            generate_bundle(package, root_seeds=[71001], episodes_per_root=1, license_name="CC0-1.0")
            with patch("scripts.kaggle_dataset_publish.subprocess.run") as run:
                result = publish(package, operation="create")
            run.assert_not_called()
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["command"][1:3], ["datasets", "create"])


if __name__ == "__main__":
    unittest.main()
