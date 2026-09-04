"""Distribution-free split-conformal forecast intervals."""

from __future__ import annotations

import numpy as np


def conformal_radius(actual: np.ndarray, predicted: np.ndarray, *, coverage: float = 0.90) -> float:
    if not 0.5 < coverage < 1:
        raise ValueError("coverage must be between 0.5 and 1")
    residuals = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("No finite calibration residuals")
    quantile_level = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(residuals, quantile_level, method="higher"))


def prediction_interval(prediction: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    return np.clip(prediction - radius, 0, None), prediction + radius
