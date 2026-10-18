import unittest
from dataclasses import replace

from farm_sim.model import Action, PolicyDecision
from validation.cognition_counterfactuals import (
    DECISION_TICK, FEATURE_INDEX, _required_world_count, _validate_checkpoint, _validate_precision_decision, _variance_components, run_episode, scenario_world, intervention_world, prepare_decision_fixture,
)


class FeatureSensitivePolicy:
    """Deterministic test policy: drink probability tracks selected cue."""
    policy_id = "feature-sensitive-test"
    def copy(self):
        return FeatureSensitivePolicy()
    def decide(self, features):
        score = 0.1 + 0.7 * features[12] - 0.7 * features[14] + 0.7 * features[9]
        score = max(0.0, min(0.95, score))
        others = [a for a in Action if a != Action.DRINK]
        probs = [0.0] * len(Action)
        probs[list(Action).index(Action.DRINK)] = score
        remainder = (1.0 - score) / len(others)
        for action in others:
            probs[list(Action).index(action)] = remainder
        chosen = Action.DRINK if score >= max(remainder, 0) else Action.WALK
        return PolicyDecision(chosen, tuple(probs), self.policy_id)


class CounterfactualHarnessTests(unittest.TestCase):
    def setUp(self):
        self.world = scenario_world("starter_seed", 31001)

    def test_checkpoint_selection_is_bound_to_training_summary_and_hash_allowlist(self):
        from pathlib import Path
        checkpoint = Path("artifacts/validation/mvp1-remediation/training-eval/encoder_generated_wider_support_current_objective/seed-11001.trained.npz")
        self.assertTrue(checkpoint.is_file())
        policy, weight_hash, sidecar_hash, provenance = _validate_checkpoint(checkpoint, {11001})
        self.assertEqual(policy.seed, 11001)
        self.assertEqual(provenance["selected_episode"], 500)
        self.assertTrue(weight_hash)
        self.assertTrue(sidecar_hash)
        self.assertTrue(provenance["training_summary_sha256"])

    def test_precision_decision_rejects_noncontiguous_or_custom_seed_provenance(self):
        import hashlib
        import json
        pilot = list(range(31001, 31061))
        base = list(range(41001, 41031))
        reserve = list(range(41101, 41131))
        manifest = {"seed_protocol": {
            "independent_precision_pilot_world_seeds": pilot,
            "locked_test_world_seeds_by_family": {"primary_starter_14x10": base},
            "predeclared_precision_reserve_world_seeds": {"primary_starter_14x10": reserve},
        }}
        pilot_hash = hashlib.sha256(json.dumps(pilot, separators=(",", ":")).encode()).hexdigest()
        checkpoint_hashes = {"/tmp/model.npz": {"training_summary_path": "artifacts/training-summary.json",
                                                   "training_summary_sha256": "abc"}}
        expected_summary = {"artifacts/training-summary.json": "abc"}
        decision = {"harness_sha256": "h", "manifest_sha256": "m",
                    "checkpoint_hashes": checkpoint_hashes, "training_summaries": expected_summary,
                    "test_sha256": "t", "pilot_seed_ids": pilot, "pilot_seed_sha256": pilot_hash,
                    "no_locked_test_outcomes_accessed": True,
                    "training_condition": "encoder_generated_wider_support_current_objective",
                    "minimum_checkpoints": 10, "required_world_seed_count_per_factor": 32,
                    "selected_locked_world_seeds": {"starter_seed": base + reserve[:2]},
                    "precision_decision": "PLANNED_N_MET_TARGET"}
        self.assertTrue(_validate_precision_decision(decision, manifest, manifest_hash="m",
                                                     harness_hash="h", test_hash="t",
                                                     checkpoint_hashes=checkpoint_hashes,
                                                     selected_seeds=base + reserve[:2]))
        with self.assertRaises(ValueError):
            _validate_precision_decision(decision, manifest, manifest_hash="m",
                                         harness_hash="h", test_hash="t",
                                         checkpoint_hashes=checkpoint_hashes,
                                         selected_seeds=base + reserve[:1] + reserve[2:3])

    def test_unverified_custom_cells_can_never_pass_causal_gate(self):
        cell = {"training_seed": 11001, "world_seed": 31001, "environment_family": "starter_seed",
                "memory": {"success": {"p_drink": .8}, "blocked": {"p_drink": .1},
                           "success_ablation": {"p_drink": .5}, "blocked_ablation": {"p_drink": .5}},
                "perception": {"available": {"p_drink": .8}, "masked": {"p_drink": .1},
                               "available_ablation": {"p_drink": .5}, "masked_ablation": {"p_drink": .5}}}
        result = __import__("validation.cognition_counterfactuals", fromlist=["paired_stats"]).paired_stats([cell])
        self.assertEqual(result["starter_seed"]["memory"]["causal_gate"], "EXPLORATORY_ONLY_NO_PASS")
        self.assertEqual(result["starter_seed"]["perception"]["causal_gate"], "EXPLORATORY_ONLY_NO_PASS")

    def test_frozen_fixture_is_tick_12_with_age_12_memory_and_available_water(self):
        first = prepare_decision_fixture(self.world, FeatureSensitivePolicy())
        second = prepare_decision_fixture(self.world, FeatureSensitivePolicy())
        self.assertEqual(first, second)
        self.assertEqual(first.tick, DECISION_TICK)
        cow = next(a for a in first.animals if a.animal_id == "cow-1")
        self.assertEqual(len(cow.memory.events), 1)
        self.assertEqual(DECISION_TICK - cow.memory.events[0].tick, 12)
        self.assertEqual(cow.memory.events[0].kind, "water_access")
        self.assertTrue(cow.memory.events[0].success)
        self.assertTrue(any(p.position == cow.position and p.available for p in first.water))

    def test_memory_intervention_is_faithful_and_sham_replays_exactly(self):
        success = run_episode(FeatureSensitivePolicy(), self.world,
                              factor="memory", condition="success", horizon=3)
        blocked = run_episode(FeatureSensitivePolicy(), self.world,
                              factor="memory", condition="blocked", horizon=3)
        sham = run_episode(FeatureSensitivePolicy(), self.world,
                           factor="memory", condition="sham_success", horizon=3)
        self.assertTrue(success["intervention_fidelity"])
        self.assertTrue(blocked["intervention_fidelity"])
        self.assertEqual(success["first_features"][12], 1.0)
        self.assertEqual(success["first_features"][14], 0.0)
        self.assertEqual(blocked["first_features"][12], 0.0)
        self.assertEqual(blocked["first_features"][14], 1.0)
        self.assertEqual(blocked["intervention_changed_feature_indices"], [12, 14])
        self.assertEqual(blocked["memory_event_mutation_count"], 1)
        self.assertEqual(success["p_drink"], sham["p_drink"])
        self.assertEqual(success["target_trajectory"], sham["target_trajectory"])
        self.assertEqual(success["reward_curve_all_animals"], sham["reward_curve_all_animals"])
        self.assertTrue(all(success[k] == sham[k] for k in success if k != "condition"))
        self.assertNotEqual(success["p_drink"], blocked["p_drink"])
        self.assertEqual(len(success["all_animal_trajectories"]), 5)
        self.assertEqual(len(success["reward_curve_all_animals"]), 3)

    def test_memory_feature_off_ablation_attenuates_cue_to_zero(self):
        success = run_episode(FeatureSensitivePolicy(), self.world,
                              factor="memory", condition="success", ablation="memory", horizon=2)
        blocked = run_episode(FeatureSensitivePolicy(), self.world,
                              factor="memory", condition="blocked", ablation="memory", horizon=2)
        self.assertTrue(success["intervention_fidelity"])
        self.assertEqual(success["first_features"][12:17], [0.0] * 5)
        self.assertEqual(success["first_probabilities"], blocked["first_probabilities"])

    def test_perception_mask_changes_only_water_available_input(self):
        available = run_episode(FeatureSensitivePolicy(), self.world,
                                factor="perception", condition="available", horizon=3)
        masked = run_episode(FeatureSensitivePolicy(), self.world,
                             factor="perception", condition="masked", horizon=3)
        sham = run_episode(FeatureSensitivePolicy(), self.world,
                           factor="perception", condition="sham_available", horizon=3)
        changed = [i for i, (a, b) in enumerate(zip(available["first_features"], masked["first_features"])) if a != b]
        self.assertTrue(available["intervention_fidelity"])
        self.assertTrue(masked["intervention_fidelity"])
        self.assertEqual(changed, [FEATURE_INDEX["water_available"]])
        self.assertEqual(available["first_features"][9], 1.0)
        self.assertEqual(masked["first_features"][9], 0.0)
        self.assertEqual(available["target_trajectory"], sham["target_trajectory"])
        self.assertEqual(available["reward_curve_all_animals"], sham["reward_curve_all_animals"])
        self.assertTrue(all(available[k] == sham[k] for k in available if k != "condition"))
        self.assertNotEqual(available["p_drink"], masked["p_drink"])

    def test_perception_feature_off_ablation_removes_intervention_contrast(self):
        available = run_episode(FeatureSensitivePolicy(), self.world,
                                factor="perception", condition="available", ablation="perception", horizon=2)
        masked = run_episode(FeatureSensitivePolicy(), self.world,
                             factor="perception", condition="masked", ablation="perception", horizon=2)
        self.assertEqual(available["first_features"][9], 0.0)
        self.assertEqual(available["first_probabilities"], masked["first_probabilities"])

    def test_precision_pilot_planner_uses_only_variance_and_reserve_ceiling(self):
        zero = {"variance_checkpoint": 0.0, "variance_world": 0.0,
                "variance_checkpoint_by_world_residual": 0.0}
        plan = _required_world_count(zero)
        self.assertEqual(plan["required_world_seeds"], 30)
        self.assertTrue(plan["precision_attainable_with_approved_reserves"])
        high = {"variance_checkpoint": 0.0, "variance_world": 1.0,
                "variance_checkpoint_by_world_residual": 0.0}
        capped = _required_world_count(high)
        self.assertEqual(capped["required_world_seeds"], 60)
        self.assertFalse(capped["precision_attainable_with_approved_reserves"])
        # A deterministic additive matrix has no residual component.
        matrix = [[r + c for c in range(4)] for r in range(3)]
        components = _variance_components(matrix)
        self.assertEqual(components["variance_checkpoint_by_world_residual"], 0.0)

    def test_memory_world_arms_change_only_latest_success_bit(self):
        success = intervention_world(self.world, "memory", "success")
        blocked = intervention_world(self.world, "memory", "blocked")
        for a, b in zip(success.animals, blocked.animals):
            if a.animal_id == "cow-1":
                self.assertEqual(len(a.memory.events), len(b.memory.events))
                self.assertEqual(a.memory.events[-1].tick, b.memory.events[-1].tick)
                self.assertEqual(a.memory.events[-1].kind, b.memory.events[-1].kind)
                self.assertEqual(a.memory.events[-1].position, b.memory.events[-1].position)
                self.assertNotEqual(a.memory.events[-1].success, b.memory.events[-1].success)
            else:
                self.assertEqual(a, b)
        self.assertEqual(success.water, blocked.water)
        self.assertEqual(success.forage, blocked.forage)
        self.assertEqual(success.obstacles, blocked.obstacles)


if __name__ == "__main__":
    unittest.main()
