"""stats.py checked against SciPy and statsmodels (dev dependencies)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.proportion import proportion_confint

from dataquorum.evals.stats import bootstrap_ci, mcnemar, pass_at_k, wilson_interval


@pytest.mark.parametrize(("k", "n"), [(55, 55), (0, 10), (7, 10), (1, 3), (40, 55), (499, 1000)])
@pytest.mark.parametrize("conf", [0.9, 0.95, 0.99])
def test_wilson_matches_statsmodels(k: int, n: int, conf: float) -> None:
    lo, hi = proportion_confint(k, n, alpha=1 - conf, method="wilson")
    assert wilson_interval(k, n, conf) == pytest.approx((lo, hi), abs=1e-12)


def test_wilson_baseline_week6() -> None:
    lo, hi = wilson_interval(55, 55)
    assert round(lo, 3) == 0.935 and hi == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "n"), [(0, 4), (1, 6), (5, 7), (3, 12), (9, 9), (10, 25)])
def test_mcnemar_exact_matches_scipy_and_statsmodels(k: int, n: int) -> None:
    a = {f"t{i}": i < k for i in range(n)} | {"same1": True, "same2": False}
    b = {f"t{i}": i >= k for i in range(n)} | {"same1": True, "same2": False}
    result = mcnemar(a, b)
    assert (result.only_a, result.only_b, result.both, result.neither) == (k, n - k, 1, 1)
    assert result.p_value == pytest.approx(sps.binomtest(k, n, 0.5).pvalue, abs=1e-12)
    table = [[1, k], [n - k, 1]]
    assert result.p_value == pytest.approx(sm_mcnemar(table, exact=True).pvalue, abs=1e-12)


def test_mcnemar_needs_shared_tasks() -> None:
    with pytest.raises(ValueError):
        mcnemar({"a": True}, {"b": True})


@pytest.mark.parametrize(
    ("n", "c", "k"), [(10, 3, 1), (10, 3, 5), (5, 5, 2), (5, 0, 3), (20, 1, 10)]
)
def test_pass_at_k_matches_definition(n: int, c: int, k: int) -> None:
    # Probability that a random k-subset of the n tries contains at least one success.
    expected = 1 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0
    assert pass_at_k(n, c, k) == pytest.approx(expected)
    if k == 1:
        assert pass_at_k(n, c, k) == pytest.approx(c / n)


def test_bootstrap_ci_brackets_the_mean_and_matches_scipy() -> None:
    values = np.random.default_rng(1).exponential(scale=0.002, size=55)
    lo, hi = bootstrap_ci(values, resamples=20_000, seed=3)
    assert lo < values.mean() < hi
    ref = sps.bootstrap((values,), np.mean, n_resamples=20_000, method="percentile", random_state=3)
    assert lo == pytest.approx(ref.confidence_interval.low, rel=0.05)
    assert hi == pytest.approx(ref.confidence_interval.high, rel=0.05)


@pytest.mark.parametrize("df", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("x", [0.1, 1.0, 2.5, 7.0, 15.0, 40.0])
def test_chi2_sf_matches_scipy(x: float, df: int) -> None:
    from dataquorum.evals.stats import chi2_sf

    assert chi2_sf(x, df) == pytest.approx(sps.chi2.sf(x, df), rel=1e-9, abs=1e-14)


def test_friedman_matches_scipy_with_and_without_ties() -> None:
    from dataquorum.evals.stats import friedman

    rng = np.random.default_rng(0)
    scores = rng.random((8, 3)).round(1)  # rounding creates ties
    result = friedman(["a", "b", "c"], scores.tolist(), [True] * 8)
    ref = sps.friedmanchisquare(*(-scores).T)  # scipy ranks ascending, so negate
    assert result.statistic == pytest.approx(ref.statistic)
    assert result.p_value == pytest.approx(ref.pvalue)
    assert sum(result.average_ranks) == pytest.approx(6.0)  # 1 + 2 + 3


def test_friedman_respects_metric_direction() -> None:
    from dataquorum.evals.stats import friedman

    # a is best everywhere: highest AUC on task 1, lowest RMSE on task 2.
    result = friedman(["a", "b", "c"], [[0.9, 0.8, 0.7], [1.0, 2.0, 3.0]], [True, False])
    assert result.average_ranks == [1.0, 2.0, 3.0]
    assert result.critical_difference == pytest.approx(2.343 * math.sqrt(12 / 12))
