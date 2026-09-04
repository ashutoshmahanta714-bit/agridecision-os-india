"""Shared preprocessing for mixed tabular data."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    return ColumnTransformer(
        [("categorical", categorical, categorical_columns), ("numeric", numeric, numeric_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
