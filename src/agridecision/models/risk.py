"""Calibratable price-shock probability model."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agridecision.models.common import make_preprocessor


def make_risk_model(
    categorical_columns: list[str],
    numeric_columns: list[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = make_preprocessor(categorical_columns, numeric_columns, scale_numeric=True)
    estimator = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
