"""Load portal downloads from CSV or ZIP without changing source files."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from agridecision.data.schema import standardise_mandi_frame


def _read_csv(path_or_buffer: object) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path_or_buffer, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)  # type: ignore[attr-defined]
    raise ValueError("Could not decode the CSV file") from last_error


def load_mandi_file(path: str | Path) -> pd.DataFrame:
    """Read a Data.gov.in CSV/ZIP download and apply the canonical schema."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Mandi data file not found: {source}")

    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected one CSV inside ZIP, found {len(csv_names)}")
            with archive.open(csv_names[0]) as handle:
                frame = _read_csv(handle)
    else:
        frame = _read_csv(source)

    return standardise_mandi_frame(frame)
