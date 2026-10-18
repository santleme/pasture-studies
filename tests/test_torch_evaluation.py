from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # training dependency is optional for runtime-only installs
    torch = None

from farm_sim.torch_checkpoint import save_generation
from farm_sim.torch_training import ActorCritic, ACTIONS
from farm_sim.model import Action, PolicyDecision
from farm_sim.navigation import NavigationResult
from validation.evaluate_torch_mvp1 import (FAMILIES, cognition_probe,
                                             evaluate_episode, evaluate,
                                             make_scenario, verify_locked_checkpoint_binding)


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TorchEvaluationTests(unittest.TestCase):
    def test_scenarios_are_named_synthetic_and_bounds_safe(self):
        for family in FAMILIES:
            world = make_scenario(family, 12345)
            self.assertEqual(len(world.animals), 5)
            self.assertTrue(all(0 <= a.position.x < world.width and 0 <= a.position.y < world.height for a in world.animals))
            self.assertTrue(all(a.position not in world.obstacles for a in world.animals))

    def test_rollout_reports_complete_action_reward_state_and_navigation_metrics(self):
        model = ActorCritic(value_head=False, seed=101).eval()
        row = evaluate_episode(model, "starter", 752)
        self.assertEqual(row["ticks"], 72)
        self.assertEqual(sum(row["action_counts"].values()), 72 * 5)
        self.assertAlmostEqual(sum(row["action_frequencies"].values()), 1.0)
        self.assertEqual(len(row["reward_trajectory_per_tick_all_animals"]), 72)
        self.assertAlmostEqual(row["total_reward"], sum(row["reward_trajectory_per_tick_all_animals"]))
        self.assertAlmostEqual(row["total_reward"], sum(x["labeled_reward_sum"] for x in row["reward_components"].values()))
        self.assertTrue(row["navigation_reason_counts"])
        self.assertTrue(row["navigation_subreason_counts"])
        self.assertEqual(row["invalid_actions"], 0)
        self.assertEqual(set(row["final_drive_means"]), {"hunger", "thirst", "fatigue", "social_need"})
        self.assertEqual(set(row["resource_state"]), {"forage_available", "forage_total", "water_available", "water_total"})

    def test_navigation_aggregate_bucket_retains_exclusive_subreason(self):
        class RestPolicy:
            def decide(self, features):
                return PolicyDecision(Action.REST, (0, 0, 0, 1, 0, 0), "test")

        def coalesced(world, animal, action, occupied=None):
            return NavigationResult(animal.position, False, None,
                                    "no_reachable_target", "topologically_unreachable")

        with patch("farm_sim.simulation.navigate", side_effect=coalesced):
            row = evaluate_episode(RestPolicy(), "starter", 551)
        self.assertEqual(row["navigation_reason_counts"], {"no_reachable_target": 72 * 5})
        self.assertEqual(row["navigation_subreason_counts"], {"topologically_unreachable": 72 * 5})

    def test_cognition_interventions_change_only_declared_inputs_and_shams_are_identical(self):
        model = ActorCritic(value_head=False, seed=210).eval()
        result = cognition_probe(model)
        memory = result["contrasts"]["memory_success"]
        perception = result["contrasts"]["perception_available"]
        self.assertGreaterEqual(result["fixture"]["thirst"], .75)
        self.assertEqual(result["fixture"]["water_visible"], 1)
        self.assertEqual(result["fixture"]["water_available"], 1)
        self.assertLessEqual(set(memory["changed_feature_names"]), {"mem_water_success", "mem_water_blocked"})
        self.assertTrue(memory["changed_feature_names"])
        self.assertEqual(memory["direction"], "memory_success_minus_blocked")
        self.assertEqual(perception["changed_feature_names"], ["water_available"])
        self.assertEqual(perception["direction"], "water_available_feature_minus_unavailable_feature")
        for row in (memory, perception):
            self.assertTrue(row["sham"]["input_bit_identical"])
            self.assertTrue(row["sham"]["probabilities_bit_identical"])
            self.assertEqual(row["sham"]["input_sha256"], row["sham"]["repeat_input_sha256"])
            self.assertEqual(len(row["input_sha256"]["baseline"]), 64)
            self.assertEqual(set(row["baseline_probabilities"]), {a.value for a in ACTIONS})
            self.assertEqual(row["required_effect_margin_descriptive_only"], .05)
            self.assertIn("attenuation", row["ablation"])

    @staticmethod
    def checkpoint(tmp_path: Path, algorithm="a2c", seed=11001):
        from validation.preregister_torch_mvp1 import ROOT, TRAINING_SOURCE_PATHS, sha
        model = ActorCritic(value_head=algorithm == "a2c", seed=seed)
        opt = torch.optim.Adam(model.parameters())
        config = {"seed": seed, "episodes": 2,
                  "source_hashes": {name: sha(ROOT / name) for name in TRAINING_SOURCE_PATHS}}
        state = {"generation": 1, "update": 1, "episode_cursor": 2}
        return save_generation(tmp_path / algorithm, model=model, optimizer=opt,
                               state=state, config=config, algorithm=algorithm)

    def test_evaluator_pairs_three_policies_and_hashes_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            generation = self.checkpoint(tmp_path)
            output = tmp_path / "evaluation.json"
            result = evaluate(generation, algorithm="a2c", seed=11001, eval_seed=33991,
                              output=output, episodes_per_family=1)
            report = json.loads(output.read_text())
            self.assertEqual(result["episodes"], len(FAMILIES) * 3)
            self.assertEqual({r["policy"] for r in report["runs"]}, {"trained", "untrained_same_seed", "heuristic_baseline"})
            for family in FAMILIES:
                rows = [r for r in report["runs"] if r["scenario"] == family]
                self.assertEqual(len({r["world_seed"] for r in rows}), 1)
            self.assertTrue(report["training_validation_test_disjoint_check"]["pass"])
            self.assertTrue(report["source_sha256"])
            effect = report["cognition_influence"]["contrasts"]["memory_success"]["drink_probability_effect"]
            row = report["cognition_influence"]["contrasts"]["memory_success"]
            self.assertAlmostEqual(effect, row["changed_probabilities"]["drink"] - row["baseline_probabilities"]["drink"])
            self.assertEqual(len(report["paired_cognition_probe_rows"]), len(FAMILIES) * 2)

    def test_evaluator_rejects_algorithm_and_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            generation = self.checkpoint(tmp_path)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                evaluate(generation, algorithm="reinforce", seed=11001, eval_seed=77,
                         output=tmp_path / "bad-alg.json", episodes_per_family=1)
            with self.assertRaisesRegex(ValueError, "seed"):
                evaluate(generation, algorithm="a2c", seed=11002, eval_seed=77,
                         output=tmp_path / "bad-seed.json", episodes_per_family=1)

    def test_locked_binding_rejects_substituted_checkpoint_before_any_rollout(self):
        from validation.preregister_torch_mvp1 import (FAMILIES as PREREG_FAMILIES,
            TRAIN_SEEDS, build_manifest, canonical_sha, finalize_manifest)
        import hashlib

        def artifact(root, algorithm, seed, payload):
            from validation.preregister_torch_mvp1 import ROOT, TRAINING_SOURCE_PATHS, sha
            generation = root / algorithm / str(seed) / "generation-00000001"
            generation.mkdir(parents=True)
            state = generation / "training.pt"
            state.write_bytes(payload)
            manifest = {"format_version": 1, "algorithm": algorithm,
                        "config": {"seed": seed,
                                   "source_hashes": {name: sha(ROOT / name) for name in TRAINING_SOURCE_PATHS}},
                        "state_file": "training.pt",
                        "state_sha256": hashlib.sha256(payload).hexdigest(), "generation": 1}
            (generation / "manifest.json").write_text(json.dumps(manifest))
            return f"{algorithm}:{seed}={state}"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = TRAIN_SEEDS[0]
            # Actual actor checkpoint is the exact artifact selected for this arm/seed.
            selected = self.checkpoint(root / "selected", algorithm="a2c", seed=seed)
            selected_state = (selected / "training.pt").read_bytes()
            artifacts = []
            for algorithm in ("reinforce", "a2c"):
                for train_seed in TRAIN_SEEDS:
                    if algorithm == "a2c" and train_seed == seed:
                        artifacts.append(f"a2c:{seed}={selected / 'training.pt'}")
                    else:
                        artifacts.append(artifact(root / "frozen", algorithm, train_seed,
                                                  f"{algorithm}:{train_seed}".encode()))
            pending = build_manifest()
            pilot = {"schema_version": 1, "status": "MASKED_PRECISION_ONLY",
                     "manifest_sha256": pending["manifest_sha256"],
                     "pilot_seed_ids_sha256": canonical_sha(pending["seed_splits"]["precision_pilot"]),
                     "sd_untrained_returns_by_family": {family: 1.0 for family in PREREG_FAMILIES}}
            from validation.preregister_torch_mvp1 import ROOT
            analyses = [f"{name}={ROOT / name}" for name in
                        ("validation/evaluate_torch_mvp1.py", "validation/benchmark_torch_fixed_corpus.py")]
            locked = finalize_manifest(pending, pilot, checkpoint_artifacts=artifacts, analysis_artifacts=analyses)
            self.assertEqual(locked["pre_outcome_freeze"]["checkpoint_sha256"][f"a2c:{seed}"],
                             hashlib.sha256(selected_state).hexdigest())
            gen_manifest = json.loads((selected / "manifest.json").read_text())
            verify_locked_checkpoint_binding(locked, generation_path=selected,
                                             generation_manifest=gen_manifest, algorithm="a2c", seed=seed)

            wrong_sources = json.loads(json.dumps(gen_manifest))
            wrong_sources["config"]["source_hashes"]["farm_sim/torch_training.py"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "training source hashes"):
                verify_locked_checkpoint_binding(locked, generation_path=selected,
                    generation_manifest=wrong_sources, algorithm="a2c", seed=seed)

            # Preserve the state file but mutate adjacent generation metadata:
            # the lock must bind that metadata too, not only model/optimizer bytes.
            original_manifest_bytes = (selected / "manifest.json").read_bytes()
            tampered_generation_manifest = json.loads(original_manifest_bytes)
            tampered_generation_manifest["provenance_note"] = "substituted"
            (selected / "manifest.json").write_text(json.dumps(tampered_generation_manifest))
            with self.assertRaisesRegex(ValueError, "generation manifest does not match frozen"):
                verify_locked_checkpoint_binding(locked, generation_path=selected,
                    generation_manifest=tampered_generation_manifest, algorithm="a2c", seed=seed)
            (selected / "manifest.json").write_bytes(original_manifest_bytes)

            alternate = ActorCritic(value_head=True, seed=seed + 1)
            from validation.preregister_torch_mvp1 import ROOT, TRAINING_SOURCE_PATHS, sha
            substituted = save_generation(root / "substituted", model=alternate,
                optimizer=torch.optim.Adam(alternate.parameters()),
                state={"generation": 1, "update": 1, "episode_cursor": 2},
                config={"seed": seed, "episodes": 2,
                        "source_hashes": {name: sha(ROOT / name) for name in TRAINING_SOURCE_PATHS}}, algorithm="a2c")
            substituted_manifest = json.loads((substituted / "manifest.json").read_text())
            with self.assertRaisesRegex(ValueError, "does not match the frozen selected artifact"):
                verify_locked_checkpoint_binding(locked, generation_path=substituted,
                    generation_manifest=substituted_manifest, algorithm="a2c", seed=seed)

            # The public locked evaluator must reject before the first world rollout.
            manifest_path = root / "locked.json"
            manifest_path.write_text(json.dumps(locked))
            with patch("validation.evaluate_torch_mvp1.evaluate_episode") as rollout:
                with self.assertRaisesRegex(ValueError, "does not match the frozen selected artifact"):
                    evaluate(substituted, algorithm="a2c", seed=seed, eval_seed=None,
                             output=root / "should-not-exist.json", seed_manifest_path=manifest_path,
                             split_name="locked_test", episodes_per_family=1)
                rollout.assert_not_called()

            # A syntactically re-sealed manifest with a substituted analysis
            # digest must fail before the evaluator creates or runs any world.
            from validation.preregister_torch_mvp1 import canonical_sha
            bad_analysis = json.loads(json.dumps(locked))
            bad_analysis["pre_outcome_freeze"]["analysis_sha256"]["validation/evaluate_torch_mvp1.py"] = "0" * 64
            bad_analysis["manifest_sha256"] = canonical_sha({k: v for k, v in bad_analysis.items() if k != "manifest_sha256"})
            bad_analysis_path = root / "bad-analysis-lock.json"
            bad_analysis_path.write_text(json.dumps(bad_analysis))
            with patch("validation.evaluate_torch_mvp1.evaluate_episode") as rollout, \
                 patch("validation.evaluate_torch_mvp1.make_scenario") as world_factory:
                with self.assertRaisesRegex(ValueError, "analysis-code hash does not match|analysis hashes"):
                    evaluate(selected, algorithm="a2c", seed=seed, eval_seed=None,
                             output=root / "should-not-exist-analysis.json", seed_manifest_path=bad_analysis_path,
                             split_name="locked_test", episodes_per_family=1)
                rollout.assert_not_called()
                world_factory.assert_not_called()

    def test_validation_split_accepts_pending_manifest_after_analysis_edits(self):
        from validation.preregister_torch_mvp1 import build_manifest
        from validation.evaluate_torch_mvp1 import evaluation_world_schedule
        pending = build_manifest()
        with patch("validation.preregister_torch_mvp1.verify_manifest") as verify:
            rows = evaluation_world_schedule(eval_seed=None, episodes_per_family=1,
                                             seed_manifest=pending, split_name="validation")
        verify.assert_called_once_with(pending, verify_current_sources=False)
        self.assertEqual(len(rows), 30 * len(FAMILIES))


if __name__ == "__main__":
    unittest.main()
