from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.kaggle_dataset_publish import command_for, publish
from tests.fixtures.pasture_dataset import write_frozen_dataset_bundle


class KaggleDatasetPublishTests(unittest.TestCase):
    def test_builds_official_create_and_version_command_shapes(self):
        path = Path("/tmp/pasture-data")
        self.assertEqual(command_for("create", path), ["kaggle", "datasets", "create", "-p", str(path)])
        self.assertEqual(command_for("version", path, "update v2"),
                         ["kaggle", "datasets", "version", "-p", str(path), "-m", "update v2"])
        with self.assertRaisesRegex(ValueError, "version operation"):
            command_for("version", path)

    def test_default_is_verified_dry_run_using_frozen_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            package = write_frozen_dataset_bundle(Path(directory) / "fixture")
            with patch("scripts.kaggle_dataset_publish.subprocess.run") as run:
                result = publish(package, operation="create")
                version_result = publish(package, operation="version", message="v2 refresh")
            run.assert_not_called()
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["command"][1:3], ["datasets", "create"])
            self.assertTrue(version_result["dry_run"])
            self.assertEqual(version_result["command"][1:3], ["datasets", "version"])


if __name__ == "__main__":
    unittest.main()
