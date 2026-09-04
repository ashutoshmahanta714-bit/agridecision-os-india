"""Open-Meteo historical weather adapter and market-date join."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


class OpenMeteoArchiveClient:
    """Fetch daily weather; Open-Meteo does not require an API key."""

    url = "https://archive-api.open-meteo.com/v1/archive"
    daily_variables = (
        "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max"
    )

    def __init__(self, *, timeout: tuple[float, float] = (10.0, 60.0)) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AgriDecision-OS/0.1"})

    def fetch_daily(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        cache_path: str | Path | None = None,
    ) -> pd.DataFrame:
        cache = Path(cache_path) if cache_path else None
        if cache and cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = self.session.get(
                self.url,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": self.daily_variables,
                    "timezone": "Asia/Kolkata",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(payload), encoding="utf-8")

        daily = payload.get("daily", {})
        if "time" not in daily:
            raise RuntimeError("Weather response has no daily time series")
        result = pd.DataFrame(daily).rename(
            columns={
                "time": "arrival_date",
                "temperature_2m_max": "temp_max_c",
                "temperature_2m_min": "temp_min_c",
                "precipitation_sum": "precipitation_mm",
                "rain_sum": "rainfall_mm",
                "wind_speed_10m_max": "wind_speed_max_kmh",
            }
        )
        result["arrival_date"] = pd.to_datetime(result["arrival_date"])
        result["latitude"] = latitude
        result["longitude"] = longitude
        return result


def merge_market_weather(
    mandi: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    market: str | None = None,
) -> pd.DataFrame:
    prices = mandi.loc[mandi["market"].eq(market)].copy() if market else mandi.copy()
    weather_columns = [
        column
        for column in weather.columns
        if column not in {"latitude", "longitude"} or column not in prices.columns
    ]
    return prices.merge(
        weather[weather_columns], on="arrival_date", how="left", validate="many_to_one"
    )
