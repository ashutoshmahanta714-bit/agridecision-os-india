"""Canonical schema for mandi price observations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "arrival_date",
    "min_price",
    "max_price",
    "modal_price",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "arrival_quantity",
    "rainfall_mm",
    "temp_min_c",
    "temp_max_c",
    "latitude",
    "longitude",
]

ALIASES = {
    "date": "arrival_date",
    "arrival_dt": "arrival_date",
    "market_name": "market",
    "mandi": "market",
    "min_price_rs_quintal": "min_price",
    "maximum_price": "max_price",
    "max_price_rs_quintal": "max_price",
    "minimum_price": "min_price",
    "modal_price_rs_quintal": "modal_price",
    "model_price": "modal_price",
    "arrivals": "arrival_quantity",
    "arrival_qty": "arrival_quantity",
}


def _snake_case(value: Any) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def standardise_mandi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical names and data types.

    The function intentionally does not silently drop invalid rows; validation
    is handled separately so rejected observations remain auditable.
    """

    result = frame.copy()
    normalised = [_snake_case(column) for column in result.columns]
    result.columns = [ALIASES.get(column, column) for column in normalised]

    duplicated_names = result.columns[result.columns.duplicated()].tolist()
    if duplicated_names:
        raise ValueError(f"Columns collapse to duplicate canonical names: {duplicated_names}")

    missing = sorted(set(REQUIRED_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"Missing required mandi columns: {missing}")

    text_columns = ["state", "district", "market", "commodity", "variety", "grade"]
    for column in text_columns:
        if column in result:
            result[column] = result[column].astype("string").str.strip()
            result[column] = result[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    result["arrival_date"] = pd.to_datetime(
        result["arrival_date"], errors="coerce", format="mixed", dayfirst=True
    ).dt.normalize()

    numeric_columns = ["min_price", "max_price", "modal_price", *OPTIONAL_NUMERIC_COLUMNS]
    for column in numeric_columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "grade" not in result:
        result["grade"] = pd.Series(pd.NA, index=result.index, dtype="string")

    ordered = REQUIRED_COLUMNS + [
        column for column in result.columns if column not in REQUIRED_COLUMNS
    ]
    result = result.loc[:, ordered]
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
