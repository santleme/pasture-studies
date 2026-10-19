from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validation import compare_exploratory_checkpoints as compare


class ExploratoryCheckpointComparisonTests(unittest.TestCase):
    def _manifest(self, root: Path, *, episodes: int = 512, formal: bool = False):
        conditions = {}
        for name in compare.CONDITIONS:
            rows = []
            for seed in (101, 202, 303):
                generation = root / name / str(seed) / "generation-00000032"
                generation.mkdir(parents=True)
                (generation / "manifest.json").write_text(json.dumps({
                    "algorithm": "a2c",
                    "config": {"algorithm": "a2c", "seed": seed, "episodes": episodes,
                               "exploratory_only": True, "formal_eligible": False,
                               "locked_outcomes_accessed": False},
                }))
                rows.append({"generation": str(generation.relative_to(root)), "algorithm": "a2c", "seed": seed})
            conditions[name] = rows
        path = root / "checkpoints.json"
        path.write_text(json.dumps({"schema_version": 1, "mode": "exploratory_comparison",
                                    "formal_eligible": False, "locked_test_access": False,
                                    "conditions": conditions}))
        return path

    def test_manifest_requires_complete_matching_nonformal_a2c_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            with patch.object(compare, "ActorCritic", return_value=object()), \
                    patch.object(compare, "require_torch", return_value=type("Torch", (), {"float32": "float32"})()), \
                    patch.object(compare, "load_generation", return_value={"state": {
                        "episode_cursor": 512, "exploratory_only": True,
                        "formal_eligible": False, "locked_outcomes_accessed": False}}):
                groups, _ = compare.load_checkpoint_manifest(manifest)
            self.assertEqual(tuple(groups), compare.CONDITIONS)
            self.assertEqual({tuple(row["seed"] for row in group) for group in groups.values()},
                             {(101, 202, 303)})

    def test_manifest_rejects_formal_locked_and_non_512_checkpoint_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, episodes=256)
            with patch.object(compare, "ActorCritic", return_value=object()), \
                    patch.object(compare, "require_torch", return_value=type("Torch", (), {"float32": "float32"})()), \
                    patch.object(compare, "load_generation", return_value={"state": {
                        "episode_cursor": 512, "exploratory_only": True, "formal_eligible": False}}):
                with self.assertRaisesRegex(ValueError, "512-episode"):
                    compare.load_checkpoint_manifest(manifest)

            doc = json.loads(manifest.read_text())
            doc["formal_eligible"] = True
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "exploratory-only"):
                compare.load_checkpoint_manifest(manifest)

    def test_policy_aggregation_and_svg_rank_are_descriptive_and_deterministic(self):
        rows = []
        for name, reward in (("trained", 2.0), ("untrained_same_seed", 1.0),
                             ("heuristic_baseline", 0.5)):
            rows.append({"policy": name, "total_reward": reward,
                          "mean_policy_entropy_nats": .4,
                          "entropy_of_mean_action_distribution_nats": .6,
                          "action_frequencies": {"rest": 1.0},
                          "final_drive_means": {"hunger": .2},
                          "resource_state": {"forage_available": 1, "forage_total": 2},
                          "reward_components": {"forage_success": {"count": 2}},
                          "invalid_actions": 0})
        result = compare._aggregate_policy_rows([{"runs": rows}])
        self.assertEqual(result["trained"]["mean_original_reward_per_episode"], 2.0)
        self.assertEqual(result["trained"]["mean_action_frequencies"], {"rest": 1.0})
        self.assertEqual(result["trained"]["reward_component_counts"], {"forage_success": 2})
        plot = compare._svg_plot([{"condition": "A", "mean_original_reward_per_episode": 2.0,
                                   "trained_episode_count": 3}])
        self.assertIn("original simulator reward", plot)
        self.assertIn("descriptive only", plot)
        self.assertIn("n=3", plot)


if __name__ == "__main__":
    unittest.main()
