"""Transparent data-quality rules and quarantine reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

KEY_COLUMNS = ["state", "district", "market", "commodity", "variety", "arrival_date"]


@dataclass(frozen=True)
class QualityReport:
    rows_received: int
    rows_accepted: int
    rows_quarantined: int
    duplicate_rows_removed: int
    missing_required_rows: int
    non_positive_price_rows: int
    inconsistent_price_rows: int
    future_date_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_mandi_data(
    frame: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """Split standardised observations into accepted and quarantined rows."""

    today = (today or pd.Timestamp.now()).normalize()
    working = frame.copy()

    missing = working[KEY_COLUMNS + ["modal_price"]].isna().any(axis=1)
    non_positive = (working[["min_price", "max_price", "modal_price"]] <= 0).any(axis=1)
    inconsistent = ~(
        (working["min_price"] <= working["modal_price"])
        & (working["modal_price"] <= working["max_price"])
    )
    future_date = working["arrival_date"] > today

    reason = pd.Series("", index=working.index, dtype="string")
    checks = {
        "missing_required": missing,
        "non_positive_price": non_positive,
        "inconsistent_price_order": inconsistent,
        "future_date": future_date,
    }
    for label, mask in checks.items():
        reason.loc[mask] = reason.loc[mask].where(reason.loc[mask] == "", reason.loc[mask] + ";")
        reason.loc[mask] = reason.loc[mask] + label

    invalid = reason != ""
    quarantine = working.loc[invalid].copy()
    quarantine["quality_failure"] = reason.loc[invalid]

    accepted = working.loc[~invalid].copy()
    before_dedup = len(accepted)
    accepted = accepted.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    duplicates_removed = before_dedup - len(accepted)
    accepted.sort_values(KEY_COLUMNS, inplace=True)
    accepted.reset_index(drop=True, inplace=True)
    quarantine.reset_index(drop=True, inplace=True)

    report = QualityReport(
        rows_received=len(working),
        rows_accepted=len(accepted),
        rows_quarantined=len(quarantine),
        duplicate_rows_removed=duplicates_removed,
        missing_required_rows=int(missing.sum()),
        non_positive_price_rows=int(non_positive.sum()),
        inconsistent_price_rows=int(inconsistent.sum()),
        future_date_rows=int(future_date.sum()),
    )
    return accepted, quarantine, report
