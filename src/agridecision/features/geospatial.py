"""Dependency-free geospatial features for market comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    latitude_a: float | np.ndarray,
    longitude_a: float | np.ndarray,
    latitude_b: float | np.ndarray,
    longitude_b: float | np.ndarray,
) -> np.ndarray:
    lat_a, lon_a, lat_b, lon_b = map(np.radians, [latitude_a, longitude_a, latitude_b, longitude_b])
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def add_distance_to_hub(
    frame: pd.DataFrame,
    *,
    hub_latitude: float,
    hub_longitude: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["distance_to_hub_km"] = haversine_km(
        result["latitude"].to_numpy(),
        result["longitude"].to_numpy(),
        hub_latitude,
        hub_longitude,
    )
    return result
