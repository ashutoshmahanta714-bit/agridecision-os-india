"""Training-readiness checks kept separate from row-level validity."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ReadinessReport:
    rows: int
    unique_dates: int
    markets: int
    date_min: str | None
    date_max: str | None
    markets_with_minimum_history: int
    minimum_dates_per_market: int
    ready_for_forecasting: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_training_readiness(
    frame: pd.DataFrame,
    *,
    minimum_dates_per_market: int = 90,
    minimum_eligible_markets: int = 2,
) -> tuple[pd.DataFrame, ReadinessReport]:
    coverage = (
        frame.groupby(["state", "district", "market", "commodity"], observed=True)
        .agg(
            rows=("arrival_date", "size"),
            unique_dates=("arrival_date", "nunique"),
            first_date=("arrival_date", "min"),
            last_date=("arrival_date", "max"),
        )
        .reset_index()
    )
    coverage["eligible"] = coverage["unique_dates"] >= minimum_dates_per_market
    eligible = int(coverage["eligible"].sum())
    reasons: list[str] = []
    if frame["arrival_date"].nunique() < minimum_dates_per_market:
        reasons.append("dataset_date_coverage_too_short")
    if eligible < minimum_eligible_markets:
        reasons.append("too_few_markets_with_sufficient_history")
    report = ReadinessReport(
        rows=len(frame),
        unique_dates=int(frame["arrival_date"].nunique()),
        markets=int(frame["market"].nunique()),
        date_min=str(frame["arrival_date"].min().date()) if len(frame) else None,
        date_max=str(frame["arrival_date"].max().date()) if len(frame) else None,
        markets_with_minimum_history=eligible,
        minimum_dates_per_market=minimum_dates_per_market,
        ready_for_forecasting=not reasons,
        reasons=tuple(reasons),
    )
    return coverage, report
