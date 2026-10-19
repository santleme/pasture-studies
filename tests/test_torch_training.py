from __future__ import annotations
import importlib.util
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
import json
import numpy as np

from farm_sim.torch_training import (derive_seed, require_torch, shard_episode_indices,
    validate_equal_all_one_masks, validate_probabilities, InvalidPolicyOutput)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _ddp_gradient_parity_worker(rank, world_size, init_method, output_dir, features, actions, advantages):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from farm_sim.torch_training import ActorCritic
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    model = ActorCritic(seed=991, dtype=torch.float32)
    wrapped = DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    indices = np.array_split(np.arange(len(features)), world_size)[rank]
    x = torch.tensor(features[indices], dtype=torch.float32)
    a = torch.tensor(actions[indices], dtype=torch.long)
    adv = torch.tensor(advantages[indices], dtype=torch.float32)
    logits, _ = wrapped(x)
    loss = -(torch.log_softmax(logits, dim=-1).gather(1, a[:, None]).squeeze(1) * adv).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad.detach().cpu().numpy().copy() for p in model.parameters()]
    optimizer.step()
    if rank == 0:
        params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
        np.savez(Path(output_dir) / f"ddp-{len(features)}.npz", *gradients, **{f"param_{i}": p for i, p in enumerate(params)})
    dist.barrier()
    dist.destroy_process_group()


def _readiness_ddp_worker(rank, init_method, output_dir, resume, max_updates):
    import torch.distributed as dist
    from validation.train_torch_mvp1 import run_readiness_ddp
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        run_readiness_ddp("a2c", seed=8821, output=Path(output_dir), device="cpu",
                          resume=resume, max_updates=max_updates)
    finally:
        dist.destroy_process_group()

class TorchTrainingContractTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_order_sensitive(self):
        self.assertEqual(derive_seed(10, "world", 3), derive_seed(10, "world", 3))
        self.assertNotEqual(derive_seed(10, "world", 3), derive_seed(10, "world", 4))
        self.assertNotEqual(derive_seed(10, "a", "b"), derive_seed(10, "b", "a"))

    def test_ddp_episode_shards_and_mask_contract(self):
        full = [shard_episode_indices(0, 16, 2, rank) for rank in range(2)]
        tail = [shard_episode_indices(16, 6, 2, rank) for rank in range(2)]
        self.assertEqual([len(x) for x in full], [8, 8])
        self.assertEqual([len(x) for x in tail], [3, 3])
        self.assertEqual(sorted(full[0] + full[1]), list(range(16)))
        self.assertEqual(sorted(tail[0] + tail[1]), list(range(16, 22)))
        self.assertEqual(validate_equal_all_one_masks([np.ones((72, 5)), np.ones((72, 5))]), 360)
        with self.assertRaisesRegex(ValueError, "all-one"):
            validate_equal_all_one_masks([np.ones(3), np.array([1, 1, 0])])
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            shard_episode_indices(0, 5, 2, 0)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_ddp_gloo_gradients_and_optimizer_match_global_sample_mean(self):
        import tempfile
        import torch
        import torch.multiprocessing as mp
        from farm_sim.torch_training import ActorCritic
        rng = np.random.default_rng(812)
        for batch_size in (8, 6):  # full batch and approved equal-shard tail
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                features = rng.normal(size=(batch_size, 20)).astype(np.float32)
                actions = rng.integers(0, 6, size=batch_size, dtype=np.int64)
                advantages = rng.normal(size=batch_size).astype(np.float32)
                init_method = f"file://{Path(tmp) / 'gloo-init'}"
                mp.spawn(_ddp_gradient_parity_worker,
                         args=(2, init_method, tmp, features, actions, advantages), nprocs=2, join=True)
                result = np.load(Path(tmp) / f"ddp-{batch_size}.npz")
                model = ActorCritic(seed=991, dtype=torch.float32)
                logits, _ = model(torch.tensor(features))
                action_tensor = torch.tensor(actions, dtype=torch.long)
                advantage_tensor = torch.tensor(advantages)
                loss = -(torch.log_softmax(logits, dim=-1).gather(1, action_tensor[:, None]).squeeze(1) * advantage_tensor).mean()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for index, parameter in enumerate(model.parameters()):
                    np.testing.assert_allclose(result[f"arr_{index}"], parameter.grad.numpy(), rtol=2e-5, atol=2e-6)
                optimizer.step()
                for index, parameter in enumerate(model.parameters()):
                    np.testing.assert_allclose(result[f"param_{index}"], parameter.detach().numpy(), rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_readiness_ddp_gloo_is_bounded_resumable_and_not_experiment_eligible(self):
        import tempfile
        import torch
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mp.spawn(_readiness_ddp_worker,
                     args=(f"file://{root / 'gloo-ready-1'}", str(root), None, 1),
                     nprocs=2, join=True)
            run_dir = root / "readiness-only" / "a2c"
            pointer = run_dir / "LATEST"
            self.assertTrue(pointer.is_file())
            checkpoint = run_dir / pointer.read_text().strip()
            payload = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
            self.assertEqual(payload["state"]["update"], 1)
            self.assertEqual(payload["state"]["episode_cursor"], 16)
            self.assertTrue(payload["state"]["readiness_only"])
            self.assertFalse(payload["state"]["benchmark_eligible"])

            mp.spawn(_readiness_ddp_worker,
                     args=(f"file://{root / 'gloo-ready-2'}", str(root), "latest", None),
                     nprocs=2, join=True)
            evidence = json.loads((run_dir / "readiness-evidence.json").read_text())
            self.assertFalse(evidence["benchmark_eligible"])
            self.assertFalse(evidence["evaluation_performed"])
            self.assertFalse(evidence["locked_split_accessed"])
            self.assertEqual(len(evidence["updates"]), 2)
            for update_number, update in enumerate(evidence["updates"], start=1):
                self.assertEqual(update["update"], update_number)
                ranks = update["episode_ids_by_rank"]
                self.assertEqual([rank["rank"] for rank in ranks], [0, 1])
                self.assertEqual([rank["device"] for rank in ranks], ["cpu", "cpu"])
                self.assertEqual(sorted(ep for rank in ranks for ep in rank["episode_ids"]),
                                 list(range((update_number - 1) * 16, update_number * 16)))
                for rank in ranks:
                    self.assertTrue(rank["gradient_finite"])
                    self.assertGreater(rank["gradient_norm"], 0)
                    self.assertTrue(rank["optimizer_step_changed_model"])
                    self.assertRegex(rank["model_digest"], r"^[0-9a-f]{64}$")
                self.assertEqual(len({rank["model_digest"] for rank in ranks}), 1)
                self.assertRegex(update["checkpoint_manifest_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(update["training_state_sha256"], r"^[0-9a-f]{64}$")
                generation_path = run_dir / update["checkpoint_generation"]
                self.assertEqual(hashlib.sha256((generation_path / "manifest.json").read_bytes()).hexdigest(),
                                 update["checkpoint_manifest_sha256"])
                manifest = json.loads((generation_path / "manifest.json").read_text())
                self.assertEqual(manifest["state_sha256"], update["training_state_sha256"])
            latest = torch.load(run_dir / "generation-00000002" / "training.pt",
                                map_location="cpu", weights_only=True)
            self.assertEqual(latest["state"]["episode_cursor"], 32)

            from farm_sim.torch_checkpoint import load_generation
            from farm_sim.torch_training import ActorCritic
            model = ActorCritic(value_head=True, seed=8821)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            expected = json.loads((run_dir / "generation-00000002" / "manifest.json").read_text())["config"]
            incompatible = dict(expected, protocol="different-readiness-protocol")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_generation(run_dir, model=model, optimizer=optimizer, expected_algorithm="a2c",
                                expected_config=incompatible)

            state_file = run_dir / "generation-00000002" / "training.pt"
            original = state_file.read_bytes()
            state_file.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum|valid committed"):
                load_generation(run_dir, model=model, optimizer=optimizer, expected_algorithm="a2c",
                                expected_config=expected)

    def test_probability_validation_rejects_malformed_vectors(self):
        valid = validate_probabilities([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
        self.assertEqual(valid.shape, (6,))
        for values in ([1.0], [np.nan] * 6, [-0.1, .3, .2, .2, .2, .2],
                       [.1, .1, .1, .1, .1, .1]):
            with self.subTest(values=values), self.assertRaises(InvalidPolicyOutput):
                validate_probabilities(values)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_nan_decision_and_training_loss_fail_closed(self):
        import torch
        from farm_sim.torch_training import ActorCritic, Trajectory, calculate_loss, assert_finite_training_state
        model = ActorCritic(seed=9)
        model.probabilities = lambda x: torch.full((6,), float("nan"), device=x.device)
        with self.assertRaises(InvalidPolicyOutput):
            model.decide(np.zeros(20, dtype=np.float32))
        model = ActorCritic(seed=9)
        trajectory = Trajectory(np.zeros((1, 5, 20), np.float32), np.zeros((1, 5), np.int64),
                                np.full((1, 5), np.nan, np.float32), np.zeros((1, 5), np.float64), 1)
        with self.assertRaisesRegex(FloatingPointError, "returns"):
            calculate_loss(model, [trajectory], "reinforce")
        loss = torch.tensor(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "loss"):
            assert_finite_training_state(loss, model)
        finite_loss = torch.tensor(0.0)
        model.actor1.weight.grad = torch.zeros_like(model.actor1.weight)
        model.actor1.weight.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "gradient"):
            assert_finite_training_state(finite_loss, model, gradients=True)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_evaluation_counts_real_invalid_probability_output(self):
        import torch
        from farm_sim.torch_training import ActorCritic
        from validation.train_torch_mvp1 import deterministic_rollout
        model = ActorCritic(seed=9)
        model.probabilities = lambda x: torch.full((6,), float("nan"), device=x.device)
        result = deterministic_rollout(model, 123, root_seed=1, episode_index=0)
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["decision_attempts"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual(sum(result["action_counts"].values()), 0)

    def test_cli_help_does_not_require_torch_runtime(self):
        result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1", "--help"],
                                check=True, capture_output=True, text=True)
        for flag in ("--smoke", "--algorithm", "--device", "--seed", "--episodes",
                     "--smoke-update", "--readiness-ddp", "--exploratory-a2c",
                     "--output", "--resume", "--metrics", "--checkpoint-every", "--mode"):
            self.assertIn(flag, result.stdout)

    def test_smoke_update_requires_one_algorithm_and_explicit_device(self):
        for args, expected in (
            (["--smoke-update", "--algorithm", "both", "--device", "cpu"], "exactly one"),
            (["--smoke-update", "--algorithm", "a2c", "--device", "auto"], "explicit"),
            (["--smoke-update", "--algorithm", "a2c", "--device", "cpu", "--resume"], "fresh"),
        ):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1", *args],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_cli_denies_ordinary_training_without_independent_authorization(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-be-created"
            for args in ([], ["--episodes", "278", "--algorithm", "reinforce", "--device", "cpu"],
                         ["--mode", "run", "--episodes", "32"]):
                with self.subTest(args=args):
                    result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1",
                                             *args, "--output", str(output)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("no independently trusted experimental authorization verifier", result.stderr)
                    self.assertFalse(output.exists())

    def test_exploratory_cli_requires_fixed_budget_before_cuda_check(self):
        result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1",
            "--exploratory-a2c", "--algorithm", "a2c", "--device", "cuda",
            "--seed", "9482817", "--episodes", "32", "--eval-seeds", "5"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixed A2C/CUDA/seed/budget config", result.stderr)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_exploratory_training_lineage_is_never_formal_or_benchmark_eligible(self):
        import json
        import tempfile
        import validation.train_torch_mvp1 as runner
        runner.BATCH_EPISODES = 16
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = runner.train_algorithm("a2c", seed=9482817, episodes=16, output=root,
                device="cpu", resume=None, metrics_path=root / "metrics.jsonl",
                checkpoint_every=1, eval_count=1, exploratory=True)
            metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 1)
            self.assertTrue(metrics[0]["exploratory_only"])
            self.assertFalse(metrics[0]["formal_eligible"])
            self.assertFalse(metrics[0]["benchmark_eligible"])
            artifact = json.loads((root / "a2c" / "artifact-manifest.json").read_text())
            evaluation = json.loads((root / "a2c" / "evaluation.json").read_text())
            self.assertTrue(artifact["exploratory_only"])
            self.assertFalse(artifact["formal_eligible"])
            self.assertFalse(artifact["benchmark_eligible"])
            self.assertTrue(evaluation["exploratory_only"])
            self.assertFalse(evaluation["formal_eligible"])
            self.assertFalse(result["benchmark_eligible"])
            self.assertEqual(len(evaluation["episodes"]), 1)
            rollout = evaluation["episodes"][0]
            self.assertEqual(sum(rollout["action_counts"].values()), 72 * 5)
            self.assertEqual(len(rollout["per_animal"]), 5)
            for animal in rollout["per_animal"].values():
                self.assertEqual(animal["ticks"], 72)
                self.assertEqual(sum(animal["action_counts"].values()), 72)
        # Restore the production default even if an earlier legacy test left
        # this module-level constant modified.
        runner.BATCH_EPISODES = 16

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_smoke_update_runs_exactly_one_cpu_update_and_exports_checkpoint(self):
        import json
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reinforce"
            metrics = output / "metrics.jsonl"
            result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1",
                "--smoke-update", "--algorithm", "reinforce", "--device", "cpu",
                "--seed", "810001", "--output", str(output), "--metrics", str(metrics)],
                check=True, capture_output=True, text=True)
            checkpoint = output / "smoke" / "reinforce" / "LATEST"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(metrics.is_file())
            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["update"], 1)
            self.assertEqual(rows[0]["episodes_in_batch"], 16)
            self.assertFalse(rows[0]["benchmark_eligible"])
            self.assertFalse(rows[0]["timing"]["benchmark_eligible"])
            artifacts = output / "smoke" / "reinforce"
            artifact_manifest = json.loads((artifacts / "artifact-manifest.json").read_text())
            self.assertFalse(artifact_manifest["benchmark_eligible"])
            evaluation = json.loads((artifacts / "evaluation.json").read_text())
            self.assertFalse(evaluation["benchmark_eligible"])
            cli_summary = json.loads(result.stdout)
            self.assertFalse(cli_summary["results"][0]["benchmark_eligible"])
            self.assertTrue(list(artifacts.glob("torch-reinforce-seed-*.npz")))
            generation = checkpoint.parent / checkpoint.read_text().strip()
            saved = torch.load(generation / "training.pt", map_location="cpu", weights_only=True)
            self.assertTrue(saved["state"]["smoke_run"])

            # A normal evaluation invocation must retain the diagnostic-only
            # lineage from the checkpoint rather than promoting it.
            eval_out = output / "smoke"
            eval_result = subprocess.run([sys.executable, "-m", "validation.train_torch_mvp1",
                "--mode", "eval", "--algorithm", "reinforce", "--device", "cpu",
                "--seed", "810001", "--episodes", "16", "--output", str(eval_out),
                "--resume", str(generation)], check=True, capture_output=True, text=True)
            eval_summary = json.loads(eval_result.stdout)
            self.assertFalse(eval_summary["results"][0]["benchmark_eligible"])
            readonly = json.loads((eval_out / "reinforce" / "evaluation-readonly-seed-810001.json").read_text())
            self.assertFalse(readonly["benchmark_eligible"])

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_return_to_go_and_coupled_episode_collection(self):
        from farm_sim.torch_training import ActorCritic, collect_episode
        model = ActorCritic(seed=17)
        seed = derive_seed(99, "world", 1)
        a = collect_episode(model, seed, root_seed=99, episode_index=0)
        b = collect_episode(model, seed, root_seed=99, episode_index=0)
        self.assertEqual(a.observations.shape, (72, 5, 20))
        self.assertEqual(a.actions.shape, (72, 5))
        np.testing.assert_array_equal(a.uniforms, b.uniforms)
        np.testing.assert_array_equal(a.actions, b.actions)
        np.testing.assert_array_equal(a.rewards, b.rewards)
        np.testing.assert_allclose(a.returns_to_go()[-1], a.rewards[-1])
        self.assertEqual(a.event_counts, b.event_counts)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_float64_numpy_torch_logits_probability_and_gradient_oracle(self):
        import torch
        from farm_sim.policy import CategoricalPolicy
        from farm_sim.torch_training import ActorCritic
        seed = 23
        policy = CategoricalPolicy.initialize(seed)
        model = ActorCritic(seed=seed, dtype=torch.float64)
        rng = np.random.default_rng(301)
        golden = [np.zeros(20), *[np.eye(20)[i] for i in range(20)], rng.normal(size=20)]
        for index in range(4):
            for value in (0.0, 1.0):
                row = np.zeros(20); row[index] = value; golden.append(row)
        for feature in (4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19):
            row = np.zeros(20); row[feature] = 1.0; golden.append(row)
        depleted = np.zeros(20); depleted[4:6] = 1; depleted[8:10] = 0; golden.append(depleted)
        for x in golden:
            z = policy.logits(x)
            p = policy.probabilities(x)
            with torch.no_grad():
                tz, _ = model(torch.tensor(x, dtype=torch.float64))
                tp = torch.softmax(tz, -1).numpy()
            np.testing.assert_allclose(tz.numpy(), z, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(tp, p, rtol=1e-12, atol=1e-12)
            self.assertEqual(int(np.argmax(tz.numpy())), int(np.argmax(p)))
        x = rng.normal(size=20)
        action, advantage = 3, 0.37
        model.zero_grad(set_to_none=True)
        logits, _ = model(torch.tensor(x, dtype=torch.float64))
        loss = -torch.log_softmax(logits, -1)[action] * advantage
        loss.backward()
        probs = policy.probabilities(x)
        hidden = np.tanh(x @ policy.w1 + policy.b1)
        dlogits = probs.copy(); dlogits[action] -= 1; dlogits *= advantage
        expected = {
            "actor1.weight": (np.outer(x, (policy.w2 @ dlogits) * (1 - hidden**2))).T,
            "actor1.bias": (policy.w2 @ dlogits) * (1 - hidden**2),
            "actor2.weight": np.outer(hidden, dlogits).T,
            "actor2.bias": dlogits,
        }
        for name, param in model.named_parameters():
            np.testing.assert_allclose(param.grad.numpy(), expected[name], rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_reinforce_baseline_is_prior_batch_only_and_a2c_has_value_loss(self):
        import torch
        from farm_sim.torch_training import ActorCritic, Trajectory, calculate_loss
        model = ActorCritic(seed=4)
        obs = np.zeros((2, 5, 20), np.float32)
        actions = np.zeros((2, 5), np.int64)
        rewards = np.ones((2, 5), np.float32)
        traj = Trajectory(obs, actions, rewards, np.zeros_like(rewards), 1)
        loss, stats = calculate_loss(model, [traj], "reinforce", previous_return_sum=3, previous_return_count=2)
        self.assertEqual(stats["baseline_used"], 1.5)
        self.assertTrue(torch.isfinite(loss))
        a2c = ActorCritic(seed=4, value_head=True)
        loss, stats = calculate_loss(a2c, [traj], "a2c")
        self.assertGreaterEqual(stats["value_loss"], 0)
        self.assertIsNone(stats["baseline_used"])

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_interrupted_resume_matches_uninterrupted_training(self):
        import tempfile
        import torch
        import validation.train_torch_mvp1 as runner
        from farm_sim.torch_checkpoint import load_generation
        runner.BATCH_EPISODES = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = runner.train_algorithm("reinforce", seed=321, episodes=2, output=root / "full",
                device="cpu", resume=None, metrics_path=root / "full.jsonl", checkpoint_every=1)
            runner.train_algorithm("reinforce", seed=321, episodes=2, output=root / "split",
                device="cpu", resume=None, metrics_path=root / "split.jsonl", checkpoint_every=1, max_updates=1)
            resumed = runner.train_algorithm("reinforce", seed=321, episodes=2, output=root / "split",
                device="cpu", resume=str(root / "split" / "reinforce"), metrics_path=root / "split.jsonl", checkpoint_every=1)
            self.assertEqual(full["updates"], resumed["updates"])
            a = torch.load(root / "full" / "reinforce" / "generation-00000002" / "training.pt", map_location="cpu", weights_only=False)
            b = torch.load(root / "split" / "reinforce" / "generation-00000002" / "training.pt", map_location="cpu", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            self.assertEqual(a["state"]["baseline_return_sum"], b["state"]["baseline_return_sum"])
            self.assertEqual(a["state"]["episode_cursor"], b["state"]["episode_cursor"])

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_resume_can_migrate_world_size_at_global_update_checkpoint(self):
        import json, tempfile
        import validation.train_torch_mvp1 as runner
        runner.BATCH_EPISODES = 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = runner.train_algorithm("reinforce", seed=621, episodes=4, output=root,
                device="cpu", resume=None, metrics_path=root / "metrics.jsonl", checkpoint_every=1,
                max_updates=1)
            self.assertFalse(partial.get("complete", True))
            generation = root / "reinforce" / "generation-00000001"
            manifest_path = generation / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["world_size"] = 2  # emulate a prior T4x2 execution
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "configuration"):
                runner.train_algorithm("reinforce", seed=621, episodes=4, output=root,
                    device="cpu", resume=str(root / "reinforce"), metrics_path=root / "metrics.jsonl",
                    checkpoint_every=1)
            resumed = runner.train_algorithm("reinforce", seed=621, episodes=4, output=root,
                device="cpu", resume=str(root / "reinforce"), metrics_path=root / "metrics.jsonl",
                checkpoint_every=1, allow_execution_migration=True)
            self.assertEqual(resumed["updates"], 2)
            self.assertTrue(resumed["execution_migration"])
            self.assertFalse(resumed["benchmark_eligible"])
            with (root / "metrics.jsonl").open() as f:
                metric_rows = [json.loads(line) for line in f]
            self.assertTrue(metric_rows[-1]["execution_migration"])
            self.assertFalse(metric_rows[-1]["timing"]["benchmark_eligible"])
            evaluation = json.loads((root / "reinforce" / "evaluation.json").read_text())
            self.assertTrue(evaluation["execution_migration"])
            self.assertFalse(evaluation["benchmark_eligible"])
            checkpoint_path = root / "reinforce" / "generation-00000002" / "training.pt"
            checkpoint = runner.require_torch().load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["state"]["execution_migration"])
            latest_generation = (generation.parent / "LATEST").read_text().strip()
            latest = json.loads((generation.parent / latest_generation / "manifest.json").read_text())
            self.assertEqual(latest["config"]["world_size"], 1)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_eval_mode_is_read_only_for_training_checkpoint_and_exports(self):
        import hashlib, tempfile
        import validation.train_torch_mvp1 as runner
        runner.BATCH_EPISODES = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.train_algorithm("reinforce", seed=523, episodes=1, output=root,
                device="cpu", resume=None, metrics_path=root / "metrics.jsonl", checkpoint_every=1)
            run = root / "reinforce"
            generation = run / "generation-00000001"
            before_state = hashlib.sha256((generation / "training.pt").read_bytes()).hexdigest()
            before_export = hashlib.sha256((run / "artifact-manifest.json").read_bytes()).hexdigest()
            result = runner.train_algorithm("reinforce", seed=523, episodes=1, output=root,
                device="cpu", resume=str(generation), metrics_path=root / "metrics.jsonl",
                checkpoint_every=1, eval_count=1, eval_only=True)
            self.assertTrue(result["eval_only"])
            self.assertIsNone(result["export"])
            self.assertEqual(before_state, hashlib.sha256((generation / "training.pt").read_bytes()).hexdigest())
            self.assertEqual(before_export, hashlib.sha256((run / "artifact-manifest.json").read_bytes()).hexdigest())
            self.assertTrue(Path(result["evaluation"]).name.startswith("evaluation-readonly"))

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_npz_export_transposes_linear_weights_and_preserves_argmax(self):
        import tempfile
        import torch
        from farm_sim.torch_training import ActorCritic, export_actor_npz
        from farm_sim.policy import load_checkpoint
        model = ActorCritic(seed=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_actor_npz(model, Path(tmp) / "policy.npz", seed=10, episodes=0)
            policy = load_checkpoint(path)
            rng = np.random.default_rng(1)
            golden = [np.zeros(20), *[np.eye(20)[i] for i in range(20)], rng.normal(size=20)]
            for i in range(4):
                for value in (0.0, 1.0):
                    row = np.zeros(20); row[i] = value; golden.append(row)
            for feature in (4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19):
                row = np.zeros(20); row[feature] = 1; golden.append(row)
            depleted = np.zeros(20); depleted[4:6] = 1; golden.append(depleted)
            for x in golden:
                with torch.no_grad():
                    t_logits = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
                    t_probs = torch.softmax(torch.tensor(t_logits), -1).numpy()
                n_logits = policy.logits(x)
                n_probs = policy.probabilities(x)
                self.assertEqual(int(np.argmax(t_probs)), int(np.argmax(n_probs)))
                np.testing.assert_allclose(t_logits, n_logits, atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(t_probs, n_probs, atol=1e-6, rtol=1e-4)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch training dependency is not installed")
    def test_cuda_golden_export_parity_when_device_exists(self):
        import torch
        from farm_sim.torch_training import ActorCritic, export_actor_npz
        from farm_sim.policy import load_checkpoint
        if not torch.cuda.is_available(): self.skipTest("CUDA is not available")
        with tempfile.TemporaryDirectory() as tmp:
            model = ActorCritic(seed=44).to("cuda:0")
            path = export_actor_npz(model, Path(tmp) / "cuda-policy.npz", seed=44, episodes=0)
            policy = load_checkpoint(path)
            golden = [np.zeros(20), *[np.eye(20)[i] for i in range(20)]]
            for i in range(4):
                row = np.zeros(20); row[i] = 1; golden.append(row)
            for feature in (4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19):
                row = np.zeros(20); row[feature] = 1; golden.append(row)
            for x in golden:
                with torch.no_grad():
                    logits = model(torch.as_tensor(x, dtype=torch.float32, device="cuda:0"))[0].cpu().numpy()
                    probs = torch.softmax(torch.as_tensor(logits), -1).numpy()
                expected_probs = policy.probabilities(x)
                self.assertEqual(int(np.argmax(probs)), int(np.argmax(expected_probs)))
                np.testing.assert_allclose(probs, expected_probs, atol=1e-6, rtol=1e-4)

if __name__ == "__main__": unittest.main()
