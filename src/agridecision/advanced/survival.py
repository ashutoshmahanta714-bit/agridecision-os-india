"""Price-shock episode duration and Kaplan-Meier analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_shock_episodes(
    frame: pd.DataFrame,
    *,
    shock_column: str = "is_shock",
    group_column: str = "market",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market, group in frame.sort_values([group_column, "arrival_date"]).groupby(group_column):
        active_start: pd.Timestamp | None = None
        previous_date: pd.Timestamp | None = None
        for record in group[["arrival_date", shock_column]].itertuples(index=False):
            date = pd.Timestamp(record[0])
            active = bool(record[1])
            if active and active_start is None:
                active_start = date
            if not active and active_start is not None:
                end = previous_date or date
                rows.append(
                    {
                        "market": market,
                        "start_date": active_start,
                        "end_date": end,
                        "duration_days": (end - active_start).days + 1,
                        "event_observed": 1,
                    }
                )
                active_start = None
            previous_date = date
        if active_start is not None and previous_date is not None:
            rows.append(
                {
                    "market": market,
                    "start_date": active_start,
                    "end_date": previous_date,
                    "duration_days": (previous_date - active_start).days + 1,
                    "event_observed": 0,
                }
            )
    return pd.DataFrame(rows)


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    if len(durations) != len(events) or not len(durations):
        raise ValueError("durations and events must have the same non-zero length")
    if np.any(durations <= 0) or not set(np.unique(events)).issubset({0, 1}):
        raise ValueError("durations must be positive and events must be binary")

    survival = 1.0
    rows = [
        {
            "time": 0.0,
            "at_risk": len(durations),
            "events": 0,
            "censored": 0,
            "survival_probability": 1.0,
        }
    ]
    for time in np.sort(np.unique(durations)):
        at_risk = int(np.sum(durations >= time))
        observed = int(np.sum((durations == time) & (events == 1)))
        censored = int(np.sum((durations == time) & (events == 0)))
        if at_risk and observed:
            survival *= 1 - observed / at_risk
        rows.append(
            {
                "time": float(time),
                "at_risk": at_risk,
                "events": observed,
                "censored": censored,
                "survival_probability": survival,
            }
        )
    return pd.DataFrame(rows)
