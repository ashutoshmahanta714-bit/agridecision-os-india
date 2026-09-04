"""Price forecasting model and interpretable baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from agridecision.models.common import make_preprocessor


def make_forecast_model(
    categorical_columns: list[str],
    numeric_columns: list[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = make_preprocessor(categorical_columns, numeric_columns, scale_numeric=False)
    estimator = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.2,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def seasonal_naive_prediction(frame: pd.DataFrame, *, horizon_days: int = 7) -> np.ndarray:
    lag_column = f"price_lag_{horizon_days}"
    if lag_column in frame:
        return frame[lag_column].to_numpy(dtype=float)
    if "current_modal_price" in frame:
        return frame["current_modal_price"].to_numpy(dtype=float)
    raise ValueError("No compatible baseline feature is available")


def recent_median_prediction(frame: pd.DataFrame, *, window_days: int = 7) -> np.ndarray:
    column = f"price_roll_median_{window_days}"
    if column not in frame:
        raise ValueError(f"Missing recent-median feature: {column}")
    return frame[column].to_numpy(dtype=float)
