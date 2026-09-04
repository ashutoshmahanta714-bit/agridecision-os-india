"""Join delayed outcomes and calculate production forecast performance."""

from __future__ import annotations

import pandas as pd

from agridecision.evaluation.metrics import classification_metrics, regression_metrics


def evaluate_prediction_log(
    prediction_log: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    keys: list[str] | None = None,
) -> dict:
    keys = keys or ["market", "commodity", "target_date"]
    joined = prediction_log.merge(outcomes, on=keys, how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("No prediction rows matched realised outcomes")
    result = {
        "matched_predictions": len(joined),
        "forecast": regression_metrics(
            joined["actual_price"].to_numpy(), joined["predicted_price"].to_numpy()
        ),
    }
    if {"actual_shock", "shock_probability"}.issubset(joined.columns):
        result["risk"] = classification_metrics(
            joined["actual_shock"].to_numpy(), joined["shock_probability"].to_numpy()
        )
    return result
