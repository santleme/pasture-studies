from __future__ import annotations

import unittest

from validation.summarize_neural_benchmark import FAMILIES, POLICIES, summarize


def _row(family, seed, policy, reward):
    return {
        "scenario": family,
        "world_seed": seed,
        "policy": policy,
        "total_reward": reward,
        "ticks": 2,
        "action_counts": {"DRINK": 5, "WALK": 5},
        "action_frequencies": {"DRINK": .5, "WALK": .5},
        "mean_policy_entropy_nats": .4,
        "invalid_actions": 0,
    }


def _report():
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for seed in (101, 102):
            for policy_index, policy in enumerate(POLICIES):
                rows.append(_row(family, seed, policy, 10 * family_index + seed % 100 + policy_index))
    return {"seed_split": {"evaluation_split": "validation", "sha256": "f" * 64},
            "generation_manifest_sha256": "a" * 64, "runs": rows}


class NeuralBenchmarkSummaryTests(unittest.TestCase):
    def test_summarizes_paired_baselines_behavior_and_novel_family_gap(self):
        result = summarize(_report())
        self.assertEqual(result["evaluation_split"], "validation")
        self.assertEqual(result["families"]["starter"]["policies"]["trained"]["episodes"], 2)
        self.assertEqual(result["families"]["starter"]["paired_reward_deltas"][
            "trained_minus_heuristic_baseline"]["paired_worlds"], 2)
        self.assertAlmostEqual(result["families"]["starter"]["paired_reward_deltas"][
            "trained_minus_heuristic_baseline"]["mean_total_reward_delta"], -1)
        self.assertEqual(result["families"]["starter"]["policies"]["trained"]["invalid_action_rate"], 0)
        self.assertAlmostEqual(result["generalization"]["trained_novel_minus_starter_reward"], 20)

    def test_rejects_locked_test_and_incomplete_or_duplicate_pairs(self):
        report = _report()
        report["seed_split"]["evaluation_split"] = "locked_test"
        with self.assertRaisesRegex(ValueError, "locked_test"):
            summarize(report)

        report = _report()
        report["runs"].pop()
        with self.assertRaisesRegex(ValueError, "missing policies"):
            summarize(report)

        report = _report()
        report["runs"].append(report["runs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            summarize(report)


if __name__ == "__main__":
    unittest.main()
