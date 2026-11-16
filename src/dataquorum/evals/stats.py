"""Statistics for eval results (spec S10). Pure Python + NumPy; verified against SciPy and
statsmodels in tests/test_stats.py.

- `wilson_interval`: confidence interval for a pass rate. Better than the normal
  approximation for small n and rates near 0 or 1 (our 55/55 baseline).
- `bootstrap_ci`: percentile bootstrap for any statistic (mean cost, median latency).
- `pass_at_k`: unbiased pass@k from n attempts with c successes (Chen et al., 2021):
  1 - C(n-c, k) / C(n, k).
- `mcnemar`: paired test for "did configuration B change which tasks pass?". Uses the exact
  binomial test on the discordant pairs, which is right for the small counts we get.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = _z(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, data.size), replace=True)
    stats = np.apply_along_axis(statistic, 1, samples)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples (from n tries, c correct) is correct."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass(frozen=True)
class McNemar:
    only_a: int  # tasks correct in A but not B
    only_b: int  # tasks correct in B but not A
    both: int
    neither: int
    p_value: float  # two-sided exact binomial test on the discordant pairs


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> McNemar:
    """Paired comparison over the task ids both runs share."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the runs share no task ids")
    only_a = sum(a[t] and not b[t] for t in shared)
    only_b = sum(b[t] and not a[t] for t in shared)
    both = sum(a[t] and b[t] for t in shared)
    neither = len(shared) - only_a - only_b - both
    return McNemar(only_a, only_b, both, neither, _binomial_two_sided(only_a, only_a + only_b))


def _binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test with p = 0.5 (probabilities as extreme as observed)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2**n for i in range(n + 1)]
    observed = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= observed * (1 + 1e-9))))


# ------------------------------------------------------------------ comparing several methods
@dataclass(frozen=True)
class Friedman:
    methods: list[str]
    average_ranks: list[float]  # 1 = best
    statistic: float
    p_value: float
    critical_difference: float  # Nemenyi, alpha = 0.05: ranks further apart than this differ


# Nemenyi q_0.05 (studentized range / sqrt 2) for k = 2..6 methods (Demsar, 2006).
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def rank_rows(scores: Sequence[Sequence[float]], higher_is_better: Sequence[bool]) -> np.ndarray:
    """Rank methods within each row (task); 1 = best, ties share the average rank."""
    ranks = []
    for row, higher in zip(scores, higher_is_better, strict=True):
        values = -np.asarray(row, dtype=float) if higher else np.asarray(row, dtype=float)
        order = values.argsort(kind="stable")
        r = np.empty(len(values))
        r[order] = np.arange(1, len(values) + 1)
        for v in np.unique(values):  # average ties
            r[values == v] = r[values == v].mean()
        ranks.append(r)
    return np.asarray(ranks)


def friedman(
    methods: Sequence[str], scores: Sequence[Sequence[float]], higher_is_better: Sequence[bool]
) -> Friedman:
    """Friedman test over tasks (rows) x methods (columns), as in Demsar (2006)."""
    ranks = rank_rows(scores, higher_is_better)
    n, k = ranks.shape
    mean_ranks = ranks.mean(axis=0)
    statistic = 12 * n / (k * (k + 1)) * float(np.sum(mean_ranks**2)) - 3 * n * (k + 1)
    # Tie correction (as scipy does).
    ties = sum(
        float(np.sum(c**3 - c)) for c in (np.unique(r, return_counts=True)[1] for r in ranks)
    )
    statistic /= 1 - ties / (n * k * (k * k - 1))
    q = _NEMENYI_Q05.get(k, float("nan"))
    return Friedman(
        methods=list(methods),
        average_ranks=[float(r) for r in mean_ranks],
        statistic=float(statistic),
        p_value=chi2_sf(float(statistic), k - 1),
        critical_difference=q * math.sqrt(k * (k + 1) / (6 * n)),
    )


def chi2_sf(x: float, df: int) -> float:
    """P(X >= x) for a chi-square variable: the regularised upper incomplete gamma Q(df/2, x/2)."""
    if x <= 0:
        return 1.0
    a, z = df / 2, x / 2
    if z < a + 1:  # series for the lower function, then complement
        term = total = 1 / a
        n = a
        for _ in range(500):
            n += 1
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return float(max(0.0, 1 - total * math.exp(-z + a * math.log(z) - math.lgamma(a))))
    b, c, d = z + 1 - a, 1 / 1e-300, 1 / (z + 1 - a)  # continued fraction (Lentz)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return float(math.exp(-z + a * math.log(z) - math.lgamma(a)) * h)
