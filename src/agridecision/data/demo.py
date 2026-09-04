"""Deterministic synthetic data used only for tests and product demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKETS = [
    ("Maharashtra", "Nashik", "Lasalgaon", 20.15, 74.24, 250.0),
    ("Maharashtra", "Pune", "Pune", 18.52, 73.86, 180.0),
    ("Telangana", "Hyderabad", "Bowenpally", 17.47, 78.47, 140.0),
    ("Telangana", "Ranga Reddy", "Gudimalkapur", 17.38, 78.43, 130.0),
    ("Odisha", "Khordha", "Bhubaneswar", 20.30, 85.82, 115.0),
    ("Odisha", "Cuttack", "Cuttack", 20.46, 85.88, 110.0),
    ("Karnataka", "Bengaluru", "Binny Mill", 12.97, 77.58, 155.0),
    ("Karnataka", "Belagavi", "Belagavi", 15.85, 74.50, 170.0),
]


def generate_demo_mandi_data(
    *,
    days: int = 480,
    end_date: str | pd.Timestamp = "2026-08-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Create realistic-looking but explicitly synthetic daily onion observations."""

    if days < 90:
        raise ValueError("days must be at least 90 for lagged backtesting")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp(end_date), periods=days, freq="D")
    rows: list[dict[str, object]] = []

    for market_index, (state, district, market, lat, lon, base_arrivals) in enumerate(MARKETS):
        state_effect = market_index * 55.0
        previous_price = 1800.0 + state_effect
        market_noise = rng.normal(0, 30, size=days)
        # Regularly spaced supply disruptions guarantee representation across
        # chronological train/test periods. Their timing differs by market.
        shock_days = set(range(42 + (market_index * 7) % 35, days - 8, 58))
        shock_effect = 0.0

        for day_index, date in enumerate(dates):
            annual = 260 * np.sin(2 * np.pi * day_index / 365 + market_index / 4)
            weekly = 45 * np.sin(2 * np.pi * day_index / 7)
            monsoon = 1.0 if date.month in (6, 7, 8, 9) else 0.0
            days_to_disruption = min(
                (shock_day - day_index for shock_day in shock_days if shock_day >= day_index),
                default=999,
            )
            early_warning = 1.0 if 1 <= days_to_disruption <= 7 else 0.0
            rainfall = max(
                0.0,
                rng.gamma(1.2, 5.0) * monsoon
                + rng.normal(0, 0.8)
                + early_warning * rng.uniform(10, 20),
            )
            temp_max = 30 + 5 * np.sin(2 * np.pi * (day_index + 70) / 365) + rng.normal(0, 1.2)
            arrivals = max(
                15.0,
                base_arrivals
                + 35 * np.sin(2 * np.pi * (day_index + 25) / 180)
                - 1.4 * rainfall
                - early_warning * rng.uniform(35, 60)
                + rng.normal(0, 16),
            )

            if day_index in shock_days:
                shock_effect += rng.uniform(600, 950)
            shock_effect *= 0.88
            supply_pressure = -1.7 * (arrivals - base_arrivals)
            target_level = 1900 + state_effect + annual + weekly + supply_pressure + 4.0 * rainfall
            target_level += shock_effect + market_noise[day_index]
            modal = max(350.0, 0.68 * previous_price + 0.32 * target_level)
            previous_price = modal
            spread_low = rng.uniform(80, 190)
            spread_high = rng.uniform(90, 220)

            rows.append(
                {
                    "state": state,
                    "district": district,
                    "market": market,
                    "commodity": "Onion",
                    "variety": "Other",
                    "grade": "FAQ",
                    "arrival_date": date,
                    "min_price": round(modal - spread_low, 2),
                    "max_price": round(modal + spread_high, 2),
                    "modal_price": round(modal, 2),
                    "arrival_quantity": round(arrivals, 2),
                    "rainfall_mm": round(rainfall, 2),
                    "temp_max_c": round(temp_max, 2),
                    "latitude": lat,
                    "longitude": lon,
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(rows).sort_values(["market", "arrival_date"]).reset_index(drop=True)
