"""Statistical inference helpers for market comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    mean_difference: float
    confidence_interval_95: tuple[float, float]
    test_statistic: float
    p_value: float
    test_name: str


def compare_market_prices(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    *,
    paired: bool = False,
) -> ComparisonResult:
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Each market requires at least three finite observations")
    if paired:
        if len(a) != len(b):
            raise ValueError("Paired samples must have equal lengths")
        differences = a - b
        statistic, p_value = stats.ttest_rel(a, b)
        standard_error = stats.sem(differences)
        interval = stats.t.interval(
            0.95, len(differences) - 1, loc=differences.mean(), scale=standard_error
        )
        name = "paired_t_test"
    else:
        statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
        difference = a.mean() - b.mean()
        standard_error = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        degrees = min(len(a), len(b)) - 1
        margin = stats.t.ppf(0.975, degrees) * standard_error
        interval = (difference - margin, difference + margin)
        differences = np.array([difference])
        name = "welch_t_test"
    return ComparisonResult(
        mean_difference=float(a.mean() - b.mean()),
        confidence_interval_95=(float(interval[0]), float(interval[1])),
        test_statistic=float(statistic),
        p_value=float(p_value),
        test_name=name,
    )
