"""Small reproducible analytical store for local development."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def write_mandi_observations(frame: pd.DataFrame, database_path: str | Path) -> None:
    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialisable = frame.copy()
    serialisable["arrival_date"] = pd.to_datetime(serialisable["arrival_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    with sqlite3.connect(destination) as connection:
        serialisable.to_sql("mandi_prices", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_mandi_date "
            "ON mandi_prices(state, district, market, commodity, arrival_date)"
        )


def query_frame(database_path: str | Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(sql, connection)
