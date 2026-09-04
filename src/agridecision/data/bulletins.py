"""Load labelled agricultural bulletins for the NLP laboratory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_bulletin_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"published_date", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Bulletin file is missing: {missing}")
    result = frame.copy()
    result["published_date"] = pd.to_datetime(result["published_date"], errors="coerce")
    result["text"] = result["text"].astype("string").str.strip()
    result = result.dropna(subset=["published_date", "text"])
    return result.reset_index(drop=True)
