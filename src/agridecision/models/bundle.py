"""Load a trained suite and produce consistent batch predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from agridecision.models.anomaly import anomaly_scores


class ModelBundle:
    def __init__(self, artifact_dir: str | Path) -> None:
        source = Path(artifact_dir)
        self.forecast = joblib.load(source / "forecast_model.joblib")
        self.risk = joblib.load(source / "risk_model.joblib")
        self.anomaly = joblib.load(source / "anomaly_model.joblib")
        self.contract = json.loads((source / "feature_contract.json").read_text(encoding="utf-8"))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.contract["feature_columns"]) - set(frame.columns))
        if missing:
            raise ValueError(f"Prediction input is missing features: {missing}")
        features = frame[self.contract["feature_columns"]]
        result = frame.copy()
        result["predicted_price"] = np.clip(self.forecast.predict(features), 0, None)
        radius = self.contract.get("prediction_interval_half_width_90")
        if radius is not None:
            result["prediction_lower_90"] = np.clip(result["predicted_price"] - radius, 0, None)
            result["prediction_upper_90"] = result["predicted_price"] + radius
        probability = self.risk.predict_proba(features)[:, 1]
        result["shock_probability"] = probability
        result["predicted_shock"] = (
            probability >= self.contract["risk_decision_threshold"]
        ).astype(int)
        result["anomaly_score"] = anomaly_scores(
            self.anomaly, frame[self.contract["anomaly_features"]]
        )
        return result
