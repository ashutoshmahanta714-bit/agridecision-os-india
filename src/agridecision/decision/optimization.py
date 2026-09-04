"""Allocate a fixed quantity across markets under capacity and risk constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class AllocationResult:
    allocations: pd.DataFrame
    expected_total_value: float
    success: bool
    message: str


def optimise_market_allocation(
    markets: pd.DataFrame,
    *,
    available_quantity: float,
    maximum_weight_per_market: float = 0.60,
    risk_penalty_per_probability_unit: float = 250.0,
) -> AllocationResult:
    required = {"market", "expected_net_price", "shock_probability"}
    missing = sorted(required - set(markets.columns))
    if missing:
        raise ValueError(f"Allocation input is missing: {missing}")
    if available_quantity <= 0:
        raise ValueError("available_quantity must be positive")

    utility = markets["expected_net_price"].to_numpy(
        dtype=float
    ) - risk_penalty_per_probability_unit * markets["shock_probability"].to_numpy(dtype=float)
    if "capacity" in markets:
        upper = np.minimum(
            markets["capacity"].to_numpy(dtype=float),
            available_quantity * maximum_weight_per_market,
        )
    else:
        upper = np.full(len(markets), available_quantity * maximum_weight_per_market)

    result = linprog(
        c=-utility,
        A_eq=np.ones((1, len(markets))),
        b_eq=np.array([available_quantity]),
        bounds=[(0.0, float(value)) for value in upper],
        method="highs",
    )
    allocation = markets[["market", "expected_net_price", "shock_probability"]].copy()
    allocation["allocated_quantity"] = result.x if result.success else 0.0
    allocation["expected_value"] = allocation["allocated_quantity"] * utility
    allocation = allocation.loc[allocation["allocated_quantity"] > 1e-8].reset_index(drop=True)
    return AllocationResult(
        allocations=allocation,
        expected_total_value=float(-result.fun) if result.success else 0.0,
        success=bool(result.success),
        message=str(result.message),
    )
