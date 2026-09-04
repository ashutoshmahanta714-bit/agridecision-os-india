"""Time-series features that use only information available at prediction time."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["state", "district", "market", "commodity", "variety"]
CATEGORICAL_FEATURES = ["state", "district", "market", "commodity", "variety", "grade"]
OPTIONAL_NUMERIC_FEATURES = [
    "arrival_quantity",
    "rainfall_mm",
    "temp_min_c",
    "temp_max_c",
    "latitude",
    "longitude",
]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date = pd.to_datetime(result["arrival_date"])
    result["year"] = date.dt.year
    result["month"] = date.dt.month
    result["day_of_month"] = date.dt.day
    result["day_of_week"] = date.dt.dayofweek
    result["week_of_year"] = date.dt.isocalendar().week.astype(int)
    result["quarter"] = date.dt.quarter
    result["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype(int)
    result["day_of_year_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    return result


def add_lagged_features(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = (1, 7, 14, 28),
    rolling_windows: Iterable[int] = (7, 14, 28),
) -> pd.DataFrame:
    """Create group-specific lags and shifted rolling statistics.

    Every rolling window is shifted by one day; today's target value is never
    included in a historical aggregate.
    """

    result = frame.sort_values(GROUP_COLUMNS + ["arrival_date"]).copy()
    grouped_price = result.groupby(GROUP_COLUMNS, observed=True)["modal_price"]

    for lag in lags:
        result[f"price_lag_{lag}"] = grouped_price.shift(lag)

    for window in rolling_windows:
        min_periods = max(2, window // 2)
        result[f"price_roll_mean_{window}"] = grouped_price.transform(
            lambda series, window=window, min_periods=min_periods: (
                series.shift(1).rolling(window, min_periods=min_periods).mean()
            )
        )
        result[f"price_roll_std_{window}"] = grouped_price.transform(
            lambda series, window=window, min_periods=min_periods: (
                series.shift(1).rolling(window, min_periods=min_periods).std()
            )
        )
        result[f"price_roll_median_{window}"] = grouped_price.transform(
            lambda series, window=window, min_periods=min_periods: (
                series.shift(1).rolling(window, min_periods=min_periods).median()
            )
        )

    if "arrival_quantity" in result:
        grouped_arrivals = result.groupby(GROUP_COLUMNS, observed=True)["arrival_quantity"]
        result["arrival_lag_1"] = grouped_arrivals.shift(1)
        result["arrival_roll_mean_7"] = grouped_arrivals.transform(
            lambda series: series.shift(1).rolling(7, min_periods=3).mean()
        )

    result["current_modal_price"] = result["modal_price"]
    result["price_spread"] = result["max_price"] - result["min_price"]
    result["modal_position"] = (result["modal_price"] - result["min_price"]) / result[
        "price_spread"
    ].replace(0, np.nan)
    return result


def add_exact_horizon_target(frame: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Attach the price exactly ``horizon_days`` later for the same market.

    Using a date-based self-join avoids the common mistake of treating the
    seventh available observation as seven calendar days.
    """

    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    future = frame[GROUP_COLUMNS + ["arrival_date", "modal_price"]].copy()
    future["arrival_date"] = future["arrival_date"] - pd.Timedelta(days=horizon_days)
    future.rename(columns={"modal_price": "target_modal_price"}, inplace=True)
    return frame.merge(
        future, how="left", on=GROUP_COLUMNS + ["arrival_date"], validate="one_to_one"
    )


def build_supervised_frame(
    frame: pd.DataFrame,
    *,
    horizon_days: int = 7,
    shock_threshold: float = 0.15,
    lags: Iterable[int] = (1, 7, 14, 28),
    rolling_windows: Iterable[int] = (7, 14, 28),
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build regression and classification targets with leakage-safe features."""

    result = add_calendar_features(frame)
    result = add_lagged_features(result, lags=lags, rolling_windows=rolling_windows)
    result = add_exact_horizon_target(result, horizon_days=horizon_days)
    result["target_return"] = result["target_modal_price"] / result["current_modal_price"] - 1.0
    result["target_price_shock"] = (result["target_return"] >= shock_threshold).astype("Int64")
    if drop_incomplete:
        required = ["target_modal_price", "price_lag_1", "price_lag_7"]
        result = result.dropna(subset=required).copy()
    result.sort_values(["arrival_date", *GROUP_COLUMNS], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def select_model_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    excluded = {
        *GROUP_COLUMNS,
        "grade",
        "arrival_date",
        "min_price",
        "max_price",
        "modal_price",
        "target_modal_price",
        "target_return",
        "target_price_shock",
        "is_synthetic",
    }
    numeric = [
        column
        for column in frame.select_dtypes(include=["number", "boolean"]).columns
        if column not in excluded
    ]
    return categorical, numeric
