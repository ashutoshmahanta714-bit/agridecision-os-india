"""Convert forecasts into transparent market recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    minimum, maximum = series.min(), series.max()
    if pd.isna(minimum) or pd.isna(maximum) or np.isclose(maximum, minimum):
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def rank_markets(
    forecasts: pd.DataFrame,
    *,
    transport_cost_per_km_quintal: float = 1.2,
    risk_penalty: float = 250.0,
    anomaly_penalty: float = 80.0,
) -> pd.DataFrame:
    """Rank markets using expected net price and explicit risk penalties.

    Required columns are ``market``, ``predicted_price``, and ``distance_km``.
    Probabilistic risk and anomaly scores are optional but recommended.
    """

    required = {"market", "predicted_price", "distance_km"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Market ranking input is missing: {missing}")
    result = forecasts.copy()
    result["shock_probability"] = result.get("shock_probability", 0.0)
    result["anomaly_score"] = result.get("anomaly_score", 0.0)
    result["transport_cost"] = result["distance_km"] * transport_cost_per_km_quintal
    result["expected_net_price"] = result["predicted_price"] - result["transport_cost"]
    result["decision_utility"] = (
        result["expected_net_price"]
        - risk_penalty * result["shock_probability"]
        - anomaly_penalty * _minmax(result["anomaly_score"])
    )
    result["recommendation_rank"] = (
        result["decision_utility"].rank(method="dense", ascending=False).astype(int)
    )
    result.sort_values(["recommendation_rank", "market"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
