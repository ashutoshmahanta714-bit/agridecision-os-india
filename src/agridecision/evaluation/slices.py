"""Evaluate performance across markets instead of hiding weak subgroups."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error


def regression_slice_report(
    predictions: pd.DataFrame,
    *,
    group_column: str = "market",
    minimum_rows: int = 10,
) -> pd.DataFrame:
    required = {group_column, "actual_price", "predicted_price", "baseline_price"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Slice report is missing: {missing}")
    rows = []
    for group, part in predictions.groupby(group_column, observed=True):
        if len(part) < minimum_rows:
            continue
        model_mae = mean_absolute_error(part["actual_price"], part["predicted_price"])
        baseline_mae = mean_absolute_error(part["actual_price"], part["baseline_price"])
        rows.append(
            {
                group_column: group,
                "rows": len(part),
                "model_mae": model_mae,
                "baseline_mae": baseline_mae,
                "model_beats_baseline": model_mae < baseline_mae,
            }
        )
    return pd.DataFrame(rows).sort_values("model_mae", ascending=False).reset_index(drop=True)
