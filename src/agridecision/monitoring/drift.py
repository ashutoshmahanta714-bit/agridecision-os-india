"""Population Stability Index and distribution diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    expected = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    observed = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(expected) or not len(observed):
        return float("nan")
    internal = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1])
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    expected_counts, _ = np.histogram(expected, bins=edges)
    observed_counts, _ = np.histogram(observed, bins=edges)
    expected_share = np.clip(expected_counts / expected_counts.sum(), epsilon, None)
    observed_share = np.clip(observed_counts / observed_counts.sum(), epsilon, None)
    return float(
        np.sum((observed_share - expected_share) * np.log(observed_share / expected_share))
    )


def numeric_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    warning_threshold: float = 0.10,
    alert_threshold: float = 0.25,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in reference or column not in current:
            continue
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur = pd.to_numeric(current[column], errors="coerce").dropna()
        if not len(ref) or not len(cur):
            continue
        psi = population_stability_index(ref, cur)
        ks = ks_2samp(ref, cur)
        status = (
            "alert"
            if psi >= alert_threshold
            else "warning"
            if psi >= warning_threshold
            else "stable"
        )
        rows.append(
            {
                "feature": column,
                "reference_mean": float(ref.mean()),
                "current_mean": float(cur.mean()),
                "psi": psi,
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
