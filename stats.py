"""Resampling statistics used by the detector.

All tests here are *paired*: every benchmark-styled prompt has a twin
user-styled prompt asking the same question. Under the null hypothesis that
prompt style does not matter, the two responses in a pair are exchangeable,
so we can swap them at random to build the null distribution of any
statistic. No distributional assumptions, and the p-values are exact up to
Monte Carlo error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    statistic: float
    p_value: float
    n_permutations: int
    null_mean: float
    null_sd: float


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> PermutationResult:
    """Two-sided paired permutation test for the mean difference ``a - b``.

    ``a`` and ``b`` are 1-D arrays of the same length (one entry per pair).
    Each permutation flips the sign of a random subset of pair differences.
    The p-value includes the observed statistic in the count (the usual
    ``(k + 1) / (n + 1)`` estimator), so it is never exactly zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    d = a - b
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.shape[0]))
    null = (signs * d[None, :]).mean(axis=1)
    k = int(np.sum(np.abs(null) >= abs(obs) - 1e-12))
    return PermutationResult(obs, (k + 1) / (n_perm + 1), n_perm, float(null.mean()), float(null.std()))


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def _energy_from_matrix(D: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> float:
    return float(2 * D[np.ix_(xi, yi)].mean() - D[np.ix_(xi, xi)].mean() - D[np.ix_(yi, yi)].mean())


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance between two samples of feature vectors (rows)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    D = _pairwise_distances(np.concatenate([x, y], axis=0))
    return _energy_from_matrix(D, np.arange(len(x)), len(x) + np.arange(len(y)))


def paired_energy_test(x: np.ndarray, y: np.ndarray, n_perm: int = 2_000, seed: int = 0) -> PermutationResult:
    """Paired permutation test on the energy distance between two feature samples.

    ``x[i]`` and ``y[i]`` are feature vectors for the two responses in pair
    ``i``. Each permutation swaps ``x[i]`` and ``y[i]`` for a random subset of
    pairs. This detects *any* shift in the joint distribution of the
    features, including changes the mean-gap tests would miss.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be 2-D arrays of equal shape")
    n = x.shape[0]
    D = _pairwise_distances(np.concatenate([x, y], axis=0))
    base = np.arange(n)
    obs = _energy_from_matrix(D, base, n + base)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        xi = np.where(swap, n + base, base)
        yi = np.where(swap, base, n + base)
        null[i] = _energy_from_matrix(D, xi, yi)
    k = int(np.sum(null >= obs - 1e-12))
    return PermutationResult(float(obs), (k + 1) / (n_perm + 1), n_perm, float(null.mean()), float(null.std()))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2_000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a mean. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)
