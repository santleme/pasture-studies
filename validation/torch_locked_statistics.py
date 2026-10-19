"""Outcome-agnostic statistics for preregistered paired locked analyses.

Inputs are paired episode-cell effects arranged as checkpoint × world.  The
crossed bootstrap resamples those two cluster axes independently, preserving
their crossed pairing; it never treats repeated cells as independent units.

This module contains no experiment access, policy evaluation, thresholds, or
scientific interpretation.  It is a numerical utility only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


def _matrix(values: object) -> np.ndarray:
    """Return a validated finite, non-empty 2-D float64 cell matrix."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("values must be a rectangular numeric matrix") from exc
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("values must be a non-empty checkpoint-by-world matrix")
    if not np.isfinite(result).all():
        raise ValueError("values must contain only finite numbers")
    return result


def _validate_resamples(resamples: int) -> int:
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    return int(resamples)


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return value


def crossed_bootstrap_means(
    values: object, *, resamples: int = 10_000, seed: int = 0
) -> np.ndarray:
    """Sample crossed means by independent checkpoint/world cluster resampling.

    One sampled checkpoint index vector and one sampled world index vector are
    drawn per replicate. Their Cartesian product is averaged, so cell pairing
    and the crossed design are retained. A one-cluster axis is valid and is
    repeatedly selected as the only available cluster.
    """
    cells = _matrix(values)
    count = _validate_resamples(resamples)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    n_checkpoints, n_worlds = cells.shape
    output = np.empty(count, dtype=np.float64)
    for i in range(count):
        checkpoints = rng.integers(0, n_checkpoints, size=n_checkpoints)
        worlds = rng.integers(0, n_worlds, size=n_worlds)
        output[i] = cells[np.ix_(checkpoints, worlds)].mean()
    return output


def crossed_bootstrap_estimate(
    values: object,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> BootstrapEstimate:
    """Estimate the paired-cell mean and its two-sided percentile interval."""
    cells = _matrix(values)
    confidence = _validate_probability(confidence_level, "confidence_level")
    samples = crossed_bootstrap_means(cells, resamples=resamples, seed=seed)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail], method="linear")
    return BootstrapEstimate(float(cells.mean()), float(low), float(high), confidence, len(samples))


def centered_bootstrap_p_value(
    values: object,
    null_margin: float = 0.0,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """One-sided centered-bootstrap p-value for H0: mean <= ``null_margin``.

    Bootstrap effects are centered at the observed mean and shifted to the
    null boundary. The plus-one correction avoids zero Monte Carlo p-values.
    Degenerate inputs are handled explicitly: a constant matrix strictly
    above the margin gets 1/(B+1), otherwise its p-value is 1.
    """
    cells = _matrix(values)
    margin = float(null_margin)
    if not math.isfinite(margin):
        raise ValueError("null_margin must be finite")
    samples = crossed_bootstrap_means(cells, resamples=resamples, seed=seed)
    observed = float(cells.mean())
    if np.all(cells == cells.flat[0]):
        return 1.0 / (len(samples) + 1) if observed > margin else 1.0
    # Centering each bootstrap estimate at the observed mean and shifting to
    # the null makes this a bootstrap distribution under the stated boundary.
    null_samples = samples - observed + margin
    exceedances = int(np.count_nonzero(null_samples >= observed))
    return float((exceedances + 1) / (len(samples) + 1))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the original order."""
    try:
        p = np.asarray(list(p_values), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("p_values must be a non-empty numeric sequence") from exc
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a non-empty finite sequence in [0, 1]")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    m = p.size
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def holm_inverted_lower_bound(
    target_values: object,
    other_null_p_values: Sequence[float],
    *,
    target_seed: int = 0,
    resamples: int = 10_000,
    alpha: float = 0.05,
    iterations: int = 60,
) -> float | None:
    """Invert a Holm-adjusted one-sided test to obtain a lower bound.

    ``other_null_p_values`` are the registered companion tests' p-values at
    their fixed null margins. Only the target test's null margin is varied.
    At each candidate margin its centered-bootstrap p-value is recomputed from
    one reproducible crossed-bootstrap sample, Holm is recalculated over the
    complete family, and the target adjusted p-value is compared with alpha.

    ``None`` means even the smallest attainable target p-value cannot pass
    Holm with the supplied companion p-values; in that case no finite lower
    confidence bound is supported. This also handles constant matrices and
    low-resample Monte Carlo resolution without bypassing multiplicity.
    """
    cells = _matrix(target_values)
    if not isinstance(other_null_p_values, Sequence) or isinstance(other_null_p_values, (str, bytes)):
        raise ValueError("other_null_p_values must be a finite sequence")
    try:
        others = [float(x) for x in other_null_p_values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("other_null_p_values must be numeric") from exc
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in others):
        raise ValueError("other_null_p_values must lie in [0, 1]")
    if isinstance(target_seed, bool) or not isinstance(target_seed, (int, np.integer)):
        raise ValueError("target_seed must be an integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    alpha = _validate_probability(alpha, "alpha")
    count = _validate_resamples(resamples)
    observed = float(cells.mean())
    samples = crossed_bootstrap_means(cells, resamples=count, seed=int(target_seed))
    centered = samples - observed

    def adjusted_target_at(margin: float) -> float:
        p_target = (np.count_nonzero(centered >= observed - margin) + 1) / (count + 1)
        adjusted = holm_adjust([*others, float(p_target)])
        return float(adjusted[-1])

    # Find a finite bracket with adjusted p <= alpha on the left and > alpha
    # on the right. Doubling allows a bound beyond observed sample support.
    span = max(float(np.ptp(cells)), abs(observed), 1.0)
    low, high = observed - span, observed + span
    for _ in range(128):
        if adjusted_target_at(low) <= alpha:
            break
        span *= 2.0
        low = observed - span
    else:
        return None
    span = max(float(np.ptp(cells)), abs(observed), 1.0)
    high = observed + span
    for _ in range(128):
        if adjusted_target_at(high) > alpha:
            break
        span *= 2.0
        high = observed + span
    else:
        raise ArithmeticError("could not bracket Holm-inverted lower bound")

    for _ in range(iterations):
        mid = (low + high) / 2.0
        if adjusted_target_at(mid) <= alpha:
            low = mid
        else:
            high = mid
    return float(low)
