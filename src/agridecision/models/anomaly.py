"""Unsupervised anomaly scoring for market observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ANOMALY_FEATURES = [
    "current_modal_price",
    "price_spread",
    "price_lag_1",
    "price_lag_7",
    "price_roll_mean_7",
    "price_roll_std_7",
    "arrival_quantity",
    "rainfall_mm",
]


def available_anomaly_features(frame: pd.DataFrame) -> list[str]:
    return [column for column in ANOMALY_FEATURES if column in frame]


def make_anomaly_model(*, contamination: float = 0.03, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def anomaly_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    # Negation makes larger values more anomalous and easier to explain.
    return -model.decision_function(features)
