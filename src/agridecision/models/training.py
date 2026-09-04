"""End-to-end, time-aware model training and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agridecision.evaluation.metrics import (
    best_f1_threshold,
    classification_metrics,
    regression_metrics,
)
from agridecision.evaluation.split import temporal_holdout
from agridecision.features.tabular import GROUP_COLUMNS, select_model_features
from agridecision.models.anomaly import (
    anomaly_scores,
    available_anomaly_features,
    make_anomaly_model,
)
from agridecision.models.forecast import (
    make_forecast_model,
    recent_median_prediction,
    seasonal_naive_prediction,
)
from agridecision.models.risk import make_risk_model
from agridecision.models.uncertainty import conformal_radius, prediction_interval


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _validation_split_for_threshold(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        subtrain, validation, _ = temporal_holdout(train, test_fraction=0.20)
    except ValueError:
        return None
    if (
        subtrain["target_price_shock"].nunique() < 2
        or validation["target_price_shock"].nunique() < 2
    ):
        return None
    return subtrain, validation


def train_model_suite(
    supervised: pd.DataFrame,
    output_dir: str | Path,
    *,
    test_fraction: float = 0.20,
    horizon_days: int = 7,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train models, evaluate on the newest dates, and write reusable artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    required = {"target_modal_price", "target_price_shock", "arrival_date"}
    missing = sorted(required - set(supervised.columns))
    if missing:
        raise ValueError(f"Supervised frame is missing: {missing}")

    train, test, cutoff = temporal_holdout(supervised, test_fraction=test_fraction)
    categorical, numeric = select_model_features(train)
    feature_columns = [*categorical, *numeric]
    if not feature_columns:
        raise ValueError("No model features were detected")

    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y_train_price = train["target_modal_price"].astype(float)
    y_test_price = test["target_modal_price"].astype(float)

    forecast_model = make_forecast_model(categorical, numeric, random_state=random_state)
    forecast_model.fit(x_train, y_train_price)
    forecast_prediction = np.clip(forecast_model.predict(x_test), 0, None)
    baseline_prediction = seasonal_naive_prediction(test, horizon_days=horizon_days)
    median_prediction = recent_median_prediction(test, window_days=7)
    naive_scale_errors = (
        (train["current_modal_price"] - train.get("price_lag_7", train["current_modal_price"]))
        .dropna()
        .to_numpy()
    )

    price_metrics = regression_metrics(
        y_test_price.to_numpy(), forecast_prediction, naive_errors=naive_scale_errors
    )
    baseline_metrics = regression_metrics(
        y_test_price.to_numpy(), baseline_prediction, naive_errors=naive_scale_errors
    )
    median_baseline_metrics = regression_metrics(
        y_test_price.to_numpy(), median_prediction, naive_errors=naive_scale_errors
    )

    interval_radius: float | None = None
    try:
        calibration_train, calibration, _ = temporal_holdout(train, test_fraction=0.20)
        calibration_model = make_forecast_model(categorical, numeric, random_state=random_state)
        calibration_model.fit(
            calibration_train[feature_columns],
            calibration_train["target_modal_price"].astype(float),
        )
        calibration_prediction = calibration_model.predict(calibration[feature_columns])
        interval_radius = conformal_radius(
            calibration["target_modal_price"].to_numpy(), calibration_prediction, coverage=0.90
        )
    except ValueError:
        interval_radius = None

    y_train_risk = train["target_price_shock"].astype(int)
    y_test_risk = test["target_price_shock"].astype(int)
    if y_train_risk.nunique() < 2:
        raise ValueError(
            "The training period contains only one shock class. Increase history or revise the threshold."
        )

    decision_threshold = 0.5
    validation_parts = _validation_split_for_threshold(train)
    if validation_parts is not None:
        subtrain, validation = validation_parts
        temporary_model = make_risk_model(categorical, numeric, random_state=random_state)
        temporary_model.fit(subtrain[feature_columns], subtrain["target_price_shock"].astype(int))
        validation_probability = temporary_model.predict_proba(validation[feature_columns])[:, 1]
        decision_threshold = best_f1_threshold(
            validation["target_price_shock"].astype(int).to_numpy(), validation_probability
        )

    risk_model = make_risk_model(categorical, numeric, random_state=random_state)
    risk_model.fit(x_train, y_train_risk)
    risk_probability = risk_model.predict_proba(x_test)[:, 1]
    risk_metrics = classification_metrics(
        y_test_risk.to_numpy(), risk_probability, threshold=decision_threshold
    )

    anomaly_features = available_anomaly_features(train)
    anomaly_model = make_anomaly_model(random_state=random_state)
    anomaly_model.fit(train[anomaly_features])
    test_anomaly_score = anomaly_scores(anomaly_model, test[anomaly_features])

    identity_columns = [
        column
        for column in ["arrival_date", *GROUP_COLUMNS, "current_modal_price"]
        if column in test
    ]
    predictions = test[identity_columns].copy()
    predictions["actual_price"] = y_test_price.to_numpy()
    predictions["predicted_price"] = forecast_prediction
    predictions["baseline_price"] = baseline_prediction
    predictions["recent_median_price"] = median_prediction
    if interval_radius is not None:
        lower, upper = prediction_interval(forecast_prediction, interval_radius)
        predictions["prediction_lower_90"] = lower
        predictions["prediction_upper_90"] = upper
    predictions["actual_shock"] = y_test_risk.to_numpy()
    predictions["shock_probability"] = risk_probability
    predictions["predicted_shock"] = (risk_probability >= decision_threshold).astype(int)
    predictions["anomaly_score"] = test_anomaly_score

    metrics: dict[str, Any] = {
        "evaluation": {
            "strategy": "chronological_holdout",
            "cutoff_date": cutoff,
            "train_rows": len(train),
            "test_rows": len(test),
            "horizon_days": horizon_days,
        },
        "forecast_model": price_metrics,
        "seasonal_naive_baseline": baseline_metrics,
        "recent_median_baseline": median_baseline_metrics,
        "price_shock_model": risk_metrics,
        "data_provenance": {
            "contains_synthetic_rows": bool(supervised.get("is_synthetic", False).any())
            if "is_synthetic" in supervised
            else False,
            "metrics_are_portfolio_claims": False,
        },
    }
    if interval_radius is not None:
        metrics["forecast_uncertainty"] = {
            "method": "split_conformal",
            "nominal_coverage": 0.90,
            "empirical_test_coverage": float(
                np.mean(
                    (y_test_price.to_numpy() >= predictions["prediction_lower_90"].to_numpy())
                    & (y_test_price.to_numpy() <= predictions["prediction_upper_90"].to_numpy())
                )
            ),
            "mean_interval_width": float(2 * interval_radius),
        }
    contract = {
        "categorical_features": categorical,
        "numeric_features": numeric,
        "feature_columns": feature_columns,
        "anomaly_features": anomaly_features,
        "forecast_horizon_days": horizon_days,
        "risk_decision_threshold": decision_threshold,
        "prediction_interval_half_width_90": interval_radius,
        "target_units": "INR per quintal",
    }

    joblib.dump(forecast_model, destination / "forecast_model.joblib")
    joblib.dump(risk_model, destination / "risk_model.joblib")
    joblib.dump(anomaly_model, destination / "anomaly_model.joblib")
    predictions.to_csv(destination / "backtest_predictions.csv", index=False)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    (destination / "feature_contract.json").write_text(
        json.dumps(contract, indent=2, default=_json_default), encoding="utf-8"
    )
    return {"metrics": metrics, "contract": contract, "predictions": predictions}
