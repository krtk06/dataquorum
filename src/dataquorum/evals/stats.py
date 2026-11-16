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
