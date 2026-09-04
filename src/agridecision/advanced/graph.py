"""Correlation and lead-lag networks between mandi price movements."""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_return_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prices = frame.pivot_table(
        index="arrival_date", columns="market", values="modal_price", aggfunc="median"
    ).sort_index()
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def correlation_edges(
    frame: pd.DataFrame,
    *,
    minimum_absolute_correlation: float = 0.55,
    minimum_shared_days: int = 30,
) -> pd.DataFrame:
    returns = market_return_matrix(frame)
    rows: list[dict[str, object]] = []
    markets = returns.columns.tolist()
    for left_index, left in enumerate(markets):
        for right in markets[left_index + 1 :]:
            pair = returns[[left, right]].dropna()
            if len(pair) < minimum_shared_days:
                continue
            correlation = float(pair[left].corr(pair[right]))
            if abs(correlation) >= minimum_absolute_correlation:
                rows.append(
                    {
                        "source_market": left,
                        "target_market": right,
                        "correlation": correlation,
                        "absolute_weight": abs(correlation),
                        "shared_days": len(pair),
                    }
                )
    return pd.DataFrame(rows)


def market_centrality(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["market", "weighted_degree", "connected_markets"])
    left = edges.groupby("source_market").agg(
        weighted_degree=("absolute_weight", "sum"), connected_markets=("target_market", "nunique")
    )
    right = edges.groupby("target_market").agg(
        weighted_degree=("absolute_weight", "sum"), connected_markets=("source_market", "nunique")
    )
    combined = left.add(right, fill_value=0).reset_index().rename(columns={"index": "market"})
    combined.rename(columns={combined.columns[0]: "market"}, inplace=True)
    return combined.sort_values("weighted_degree", ascending=False).reset_index(drop=True)


def lead_lag_edges(
    frame: pd.DataFrame, *, lag_days: int = 1, minimum_correlation: float = 0.25
) -> pd.DataFrame:
    returns = market_return_matrix(frame)
    rows: list[dict[str, object]] = []
    for source in returns.columns:
        source_lagged = returns[source].shift(lag_days)
        for target in returns.columns:
            if source == target:
                continue
            pair = pd.concat([source_lagged, returns[target]], axis=1).dropna()
            if len(pair) < 30:
                continue
            correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if correlation >= minimum_correlation:
                rows.append(
                    {
                        "source_market": source,
                        "target_market": target,
                        "lag_days": lag_days,
                        "correlation": correlation,
                    }
                )
    return (
        pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
