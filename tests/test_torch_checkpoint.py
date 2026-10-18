from __future__ import annotations
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None

class UntrustedCheckpointObject:
    pass

@unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
class TorchCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        from farm_sim.torch_training import ActorCritic
        self.torch = torch
        self.model = ActorCritic(seed=7)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.config = {"version": "test", "dtype": "float32", "episodes": 64, "batch_episodes": 16}

    def test_generation_latest_roundtrip_optimizer_cursor(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            path = save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 2, "update": 2, "episode_cursor": 32, "rng": "rngstate"},
                config=self.config, algorithm="reinforce")
            self.assertEqual((Path(tmp) / "LATEST").read_text().strip(), path.name)
            expected = {k: v.detach().clone() for k,v in self.model.state_dict().items()}
            for p in self.model.parameters(): p.data.add_(1)
            loaded = load_generation(Path(tmp) / "LATEST", model=self.model, optimizer=self.optimizer,
                                     expected_algorithm="reinforce", expected_config=self.config)
            self.assertEqual(loaded["state"]["episode_cursor"], 32)
            for k,v in self.model.state_dict().items(): self.torch.testing.assert_close(v, expected[k])

    def test_resume_adopts_complete_generation_after_interrupted_latest_update(self):
        import farm_sim.torch_checkpoint as checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint.save_generation(root, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0},
                config=self.config, algorithm="reinforce")
            original_atomic_bytes = checkpoint._atomic_bytes

            def interrupt_latest(path, content):
                if Path(path).name == "LATEST":
                    raise OSError("injected stop between generation rename and pointer commit")
                original_atomic_bytes(path, content)

            with patch.object(checkpoint, "_atomic_bytes", side_effect=interrupt_latest):
                with self.assertRaisesRegex(OSError, "injected stop"):
                    checkpoint.save_generation(root, model=self.model, optimizer=self.optimizer,
                        state={"generation": 1, "update": 1, "episode_cursor": 16},
                        config=self.config, algorithm="reinforce")

            self.assertEqual((root / "LATEST").read_text().strip(), "generation-00000000")
            loaded = checkpoint.load_generation(root / "LATEST", model=self.model, optimizer=self.optimizer,
                expected_algorithm="reinforce", expected_config=self.config)
            self.assertEqual(loaded["state"]["generation"], 1)
            self.assertEqual((root / "LATEST").read_text().strip(), "generation-00000001")

    def test_incompatible_or_invalid_orphan_does_not_advance_latest(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation

        for orphan_config, cursor in (({**self.config, "run_seed": 2}, 16), (self.config, 1)):
            with self.subTest(orphan_config=orphan_config, cursor=cursor), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                save_generation(root, model=self.model, optimizer=self.optimizer,
                    state={"generation": 0, "update": 0, "episode_cursor": 0},
                    config=self.config, algorithm="reinforce")
                save_generation(root, model=self.model, optimizer=self.optimizer,
                    state={"generation": 1, "update": 1, "episode_cursor": cursor},
                    config=orphan_config, algorithm="reinforce")
                (root / "LATEST").write_text("generation-00000000\n")
                loaded = load_generation(root / "LATEST", model=self.model, optimizer=self.optimizer,
                    expected_algorithm="reinforce", expected_config=self.config)
                self.assertEqual(loaded["state"]["generation"], 0)
                self.assertEqual((root / "LATEST").read_text().strip(), "generation-00000000")

    def test_checksum_corruption_fails_closed(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            path = save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            with (path / "training.pt").open("ab") as f: f.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_generation(path, model=self.model)


    def test_manifest_state_path_traversal_is_rejected(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            path = save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["state_file"] = "../outside.pt"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "filename"):
                load_generation(path, model=self.model)

    def test_state_symlink_and_generation_symlink_are_rejected(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_generation(root / "checkpoints", model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            original = path / "training.pt"
            external = root / "outside.pt"
            external.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_generation(path, model=self.model)
            original.unlink()
            original.write_bytes(external.read_bytes())
            alias = root / "generation-00000000"
            alias.symlink_to(path, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                load_generation(alias, model=self.model)

    def test_latest_symlink_and_non_generation_pointer_are_rejected(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            path = save_generation(root, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            latest = root / "LATEST"
            latest.unlink()
            latest.symlink_to(path / "manifest.json")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                load_generation(latest, model=self.model)
            latest.unlink()
            latest.write_text("../../outside\n")
            with self.assertRaisesRegex(ValueError, "pointer"):
                load_generation(latest, model=self.model)

    def test_nonfinite_model_or_state_refuses_checkpoint_commit(self):
        from farm_sim.torch_checkpoint import save_generation
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FloatingPointError, "model"):
                self.model.actor1.weight.data[0, 0] = float("nan")
                save_generation(tmp, model=self.model, optimizer=self.optimizer,
                    state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
        self.model.actor1.weight.data[0, 0] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FloatingPointError, "training state"):
                save_generation(tmp, model=self.model, optimizer=self.optimizer,
                    state={"generation": 0, "bad": float("nan")}, config=self.config, algorithm="reinforce")


    def test_weights_only_loader_rejects_untrusted_pickle_objects(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation, _sha256
        with tempfile.TemporaryDirectory() as tmp:
            path = save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            state_path = path / "training.pt"
            self.torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                             "state": {"generation": 0}, "untrusted": UntrustedCheckpointObject()}, state_path)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["state_sha256"] = _sha256(state_path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(Exception):
                load_generation(path, model=self.model)


    def test_execution_migration_allows_only_world_size_change_at_committed_update(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        config = {**self.config, "world_size": 2, "seed": 8}
        expected = {**config, "world_size": 1}
        with tempfile.TemporaryDirectory() as tmp:
            save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 3, "update": 3, "episode_cursor": 48,
                       "baseline_return_sum": 4.0, "baseline_return_count": 10},
                config=config, algorithm="reinforce")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_generation(tmp, model=self.model, expected_config=expected)
            loaded = load_generation(tmp, model=self.model, expected_config=expected,
                expected_algorithm="reinforce", allow_world_size_migration=True)
            self.assertEqual(loaded["execution_migration"], {
                "field": "world_size", "from": 2, "to": 1, "committed_update": 3})
            self.assertEqual(loaded["state"]["episode_cursor"], 48)
            self.assertEqual(loaded["state"]["baseline_return_count"], 10)

    def test_execution_migration_does_not_allow_other_config_changes(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        config = {**self.config, "world_size": 2}
        expected = {**config, "world_size": 1, "lr": 0.2}
        with tempfile.TemporaryDirectory() as tmp:
            save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 1, "update": 1, "episode_cursor": 16},
                config=config, algorithm="reinforce")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_generation(tmp, model=self.model, expected_config=expected,
                    allow_world_size_migration=True)

    def test_incompatible_algorithm_and_config_rejected(self):
        from farm_sim.torch_checkpoint import save_generation, load_generation
        with tempfile.TemporaryDirectory() as tmp:
            save_generation(tmp, model=self.model, optimizer=self.optimizer,
                state={"generation": 0, "update": 0, "episode_cursor": 0}, config=self.config, algorithm="reinforce")
            with self.assertRaisesRegex(ValueError, "algorithm"):
                load_generation(tmp, model=self.model, expected_algorithm="a2c")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_generation(tmp, model=self.model, expected_config={"bad": 1})

if __name__ == "__main__": unittest.main()
