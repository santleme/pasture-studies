from __future__ import annotations

import unittest

import numpy as np

from validation.torch_locked_statistics import (
    centered_bootstrap_p_value,
    crossed_bootstrap_estimate,
    crossed_bootstrap_means,
    holm_adjust,
    holm_inverted_lower_bound,
)


class TorchLockedStatisticsTests(unittest.TestCase):
    def setUp(self):
        # Synthetic paired checkpoint × world effects with a known positive
        # direction. Values are not observations from the cattle simulator.
        self.positive = np.array([
            [0.6, 0.8, 0.7, 0.9],
            [0.4, 0.7, 0.8, 0.6],
            [0.7, 0.9, 0.6, 0.8],
        ])

    def test_crossed_estimate_reports_mean_and_reproducible_interval(self):
        first = crossed_bootstrap_estimate(self.positive, resamples=800, seed=41)
        second = crossed_bootstrap_estimate(self.positive, resamples=800, seed=41)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, self.positive.mean())
        self.assertLess(first.lower, first.estimate)
        self.assertGreater(first.upper, first.estimate)

    def test_crossed_bootstrap_is_reproducible_and_directional(self):
        a = crossed_bootstrap_means(self.positive, resamples=500, seed=19)
        b = crossed_bootstrap_means(self.positive, resamples=500, seed=19)
        np.testing.assert_array_equal(a, b)
        self.assertLess(centered_bootstrap_p_value(self.positive, 0.0, resamples=2000, seed=2), .01)
        self.assertGreater(centered_bootstrap_p_value(-self.positive, 0.0, resamples=2000, seed=2), .99)

    def test_permutation_and_balanced_duplication_preserve_point_estimate(self):
        expected = self.positive.mean()
        self.assertAlmostEqual(self.positive[::-1, :].mean(), expected)
        self.assertAlmostEqual(self.positive[:, [2, 0, 3, 1]].mean(), expected)
        doubled = np.tile(self.positive, (2, 2))
        self.assertAlmostEqual(doubled.mean(), expected)
        # Bootstrap uncertainty depends on matching cell-level outcomes before
        # aggregation. Permuting one policy's world pairing preserves its mean
        # but changes the crossed-bootstrap distribution of the contrast.
        baseline = np.array([[0, 2, 1, 4], [3, 1, 5, 0], [2, 6, 1, 3.]], dtype=float)
        candidate = baseline + self.positive
        paired = candidate - baseline
        self.assertAlmostEqual(paired.mean(), expected)
        wrong_pairs = candidate - baseline[:, ::-1]
        paired_sd = crossed_bootstrap_means(paired, resamples=1500, seed=55).std()
        wrong_sd = crossed_bootstrap_means(wrong_pairs, resamples=1500, seed=55).std()
        self.assertNotAlmostEqual(paired_sd, wrong_sd, places=2)

    def test_one_cluster_axes_and_constant_matrices_are_explicit(self):
        one_axis = np.array([[1.0, 2.0, 3.0]])
        result = crossed_bootstrap_estimate(one_axis, resamples=300, seed=4)
        self.assertEqual(result.estimate, 2.0)
        self.assertLess(result.lower, result.upper)
        constant = np.full((2, 3), 0.25)
        self.assertEqual(centered_bootstrap_p_value(constant, .2, resamples=100, seed=0), 1 / 101)
        self.assertEqual(centered_bootstrap_p_value(constant, .25, resamples=100, seed=0), 1.0)
        self.assertAlmostEqual(holm_inverted_lower_bound(constant, [.4, .7], resamples=100), .25)

    def test_constant_bound_does_not_bypass_holm_resolution(self):
        constant = [[.25] * 8 for _ in range(4)]
        # At B=10 the target's minimum attainable p is 1/11; companions
        # prevent this from crossing the three-test Holm threshold.
        self.assertIsNone(holm_inverted_lower_bound(constant, [.8, .9], resamples=10))

    def test_holm_step_down_matches_known_values_and_is_monotone(self):
        raw = np.array([.01, .04, .03, .8])
        adjusted = holm_adjust(raw)
        np.testing.assert_allclose(adjusted, [.04, .09, .09, .8])
        self.assertTrue(np.all(adjusted >= raw))
        self.assertTrue(np.all(holm_adjust([.01, .04, .03, .8]) <= holm_adjust([.02, .05, .04, .9])))
        self.assertTrue(np.all(np.diff(holm_adjust([.01, .02, .03, .04])) >= 0))
        baseline = holm_adjust([.01, .03, .2])
        changed = holm_adjust([.01, .08, .2])
        self.assertGreaterEqual(changed[1], baseline[1])

    def test_holm_inverted_bound_uses_fixed_companion_null_p_values(self):
        bound_a = holm_inverted_lower_bound(self.positive, [.2, .3], resamples=2500, target_seed=13)
        bound_b = holm_inverted_lower_bound(self.positive, [.2, .3], resamples=2500, target_seed=13)
        self.assertEqual(bound_a, bound_b)
        self.assertLess(bound_a, self.positive.mean())
        self.assertLess(bound_a, self.positive.mean())

    def test_rejects_malformed_nonfinite_empty_and_invalid_parameters(self):
        invalid = [[], np.array([]), [1, 2], [[1, 2], [3]], [[1, np.nan]], [[np.inf]], [["x"]]]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                crossed_bootstrap_means(values, resamples=10, seed=1)
        for kwargs in ({"resamples": 0}, {"resamples": 2.5}, {"seed": 1.2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                crossed_bootstrap_means(self.positive, **kwargs)
        with self.assertRaises(ValueError):
            crossed_bootstrap_estimate(self.positive, confidence_level=1.0)
        with self.assertRaises(ValueError):
            centered_bootstrap_p_value(self.positive, float("nan"))
        for bad in ([float("nan")], [-.1], [1.1]):
            with self.subTest(other=bad), self.assertRaises(ValueError):
                holm_inverted_lower_bound(self.positive, bad, resamples=50)
        for bad in ([], [0.2, float("nan")], [0.1, 1.1], [[.1]]):
            with self.subTest(p_values=bad), self.assertRaises(ValueError):
                holm_adjust(bad)


if __name__ == "__main__":
    unittest.main()
