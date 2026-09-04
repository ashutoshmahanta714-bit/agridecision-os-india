"""Unsupervised market segmentation with interpretable profiles."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_market_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["daily_return"] = working.groupby("market", observed=True)["modal_price"].pct_change()
    profile = working.groupby(["state", "market"], observed=True).agg(
        mean_price=("modal_price", "mean"),
        price_volatility=("daily_return", "std"),
        observation_count=("modal_price", "size"),
    )
    if "arrival_quantity" in working:
        profile["mean_arrivals"] = working.groupby(["state", "market"], observed=True)[
            "arrival_quantity"
        ].mean()
    return profile.reset_index().dropna()


def segment_markets(
    profiles: pd.DataFrame,
    *,
    clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Pipeline]:
    numeric = profiles.select_dtypes(include="number").columns.tolist()
    numeric.remove("observation_count") if "observation_count" in numeric else None
    if len(profiles) < clusters:
        raise ValueError("Number of profiles must be at least the number of clusters")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("cluster", KMeans(n_clusters=clusters, n_init=20, random_state=random_state)),
        ]
    )
    result = profiles.copy()
    result["cluster"] = pipeline.fit_predict(result[numeric])
    return result, pipeline
