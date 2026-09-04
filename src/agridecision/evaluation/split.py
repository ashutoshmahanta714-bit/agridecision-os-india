"""Chronological splitting utilities."""

from __future__ import annotations

import pandas as pd


def temporal_holdout(
    frame: pd.DataFrame,
    *,
    date_column: str = "arrival_date",
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    dates = pd.Series(pd.to_datetime(frame[date_column]).dropna().unique()).sort_values()
    if len(dates) < 10:
        raise ValueError("At least 10 unique dates are required for temporal evaluation")
    split_position = min(len(dates) - 1, max(1, int(len(dates) * (1 - test_fraction))))
    cutoff = pd.Timestamp(dates.iloc[split_position])
    train = frame.loc[pd.to_datetime(frame[date_column]) < cutoff].copy()
    test = frame.loc[pd.to_datetime(frame[date_column]) >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set")
    return train, test, cutoff


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    date_column: str = "arrival_date",
    minimum_train_days: int = 180,
    validation_days: int = 30,
    step_days: int = 30,
):
    dates = pd.to_datetime(frame[date_column])
    start = dates.min() + pd.Timedelta(days=minimum_train_days)
    final = dates.max()
    cutoff = start
    while cutoff + pd.Timedelta(days=validation_days) <= final:
        train_index = frame.index[dates < cutoff]
        validation_index = frame.index[
            (dates >= cutoff) & (dates < cutoff + pd.Timedelta(days=validation_days))
        ]
        if len(train_index) and len(validation_index):
            yield train_index, validation_index
        cutoff += pd.Timedelta(days=step_days)
