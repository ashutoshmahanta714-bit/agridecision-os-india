"""Satellite vegetation-index functions for preprocessed spectral arrays."""

from __future__ import annotations

import numpy as np


def normalised_difference(numerator_band: np.ndarray, denominator_band: np.ndarray) -> np.ndarray:
    numerator_band = np.asarray(numerator_band, dtype=float)
    denominator_band = np.asarray(denominator_band, dtype=float)
    denominator = numerator_band + denominator_band
    return np.divide(
        numerator_band - denominator_band,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=np.abs(denominator) > 1e-12,
    )


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index: (NIR - red) / (NIR + red)."""

    return normalised_difference(nir, red)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    nir, red, blue = (np.asarray(value, dtype=float) for value in (nir, red, blue))
    denominator = nir + 6 * red - 7.5 * blue + 1
    return np.divide(
        2.5 * (nir - red),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )


def summarise_field_index(
    index: np.ndarray, valid_mask: np.ndarray | None = None
) -> dict[str, float]:
    values = np.asarray(index, dtype=float)
    if valid_mask is not None:
        values = values[np.asarray(valid_mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No valid satellite pixels are available")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "healthy_fraction": float(np.mean(values >= 0.5)),
    }
