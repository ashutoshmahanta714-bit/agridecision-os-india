"""Regression and probabilistic classification metrics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    """Choose a threshold on validation predictions, never on the test set."""

    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.linspace(0.10, 0.90, 81)
    scores = [f1_score(actual, probability >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    naive_errors: np.ndarray | None = None,
) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = np.abs(actual) > 1e-8
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])))
    mae = float(mean_absolute_error(actual, predicted))
    mase: float | None = None
    if naive_errors is not None:
        scale = float(np.mean(np.abs(np.asarray(naive_errors, dtype=float))))
        if scale > 1e-8:
            mase = mae / scale
    return {
        "mae": mae,
        "rmse": math.sqrt(float(mean_squared_error(actual, predicted))),
        "mape": mape,
        "r2": float(r2_score(actual, predicted)),
        "mase": mase,
    }


def classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    both_classes = len(np.unique(actual)) == 2
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)) if both_classes else None,
        "pr_auc": float(average_precision_score(actual, probability)) if both_classes else None,
        "brier_score": float(brier_score_loss(actual, probability)),
        "decision_threshold": threshold,
    }
