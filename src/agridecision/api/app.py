"""Inference and decision API. Run with: uvicorn agridecision.api.app:app --reload."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agridecision.data.quality import validate_mandi_data
from agridecision.data.schema import standardise_mandi_frame
from agridecision.decision.ranking import rank_markets
from agridecision.features.tabular import GROUP_COLUMNS, build_supervised_frame
from agridecision.models.bundle import ModelBundle

ARTIFACT_DIR = Path(os.getenv("AGRIDECISION_ARTIFACT_DIR", "artifacts"))
REQUIRED_ARTIFACTS = [
    "forecast_model.joblib",
    "risk_model.joblib",
    "anomaly_model.joblib",
    "feature_contract.json",
]


class FeatureBatch(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=500)


class RankingBatch(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    transport_cost_per_km_quintal: float = Field(default=1.2, ge=0)
    risk_penalty: float = Field(default=250.0, ge=0)


class HistoryBatch(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=30, max_length=20_000)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle(ARTIFACT_DIR)


app = FastAPI(
    title="AgriDecision OS API",
    description="Seven-day mandi price, shock-risk, anomaly, and market-ranking API.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    return {"status": "ready" if not missing else "not_ready", "missing_artifacts": missing}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    contract_path = ARTIFACT_DIR / "feature_contract.json"
    if not contract_path.exists():
        raise HTTPException(status_code=503, detail="Model contract is unavailable")
    return json.loads(contract_path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(batch: FeatureBatch) -> dict[str, Any]:
    try:
        output = get_bundle().predict(pd.DataFrame(batch.records))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    columns = [
        column
        for column in [
            "market",
            "commodity",
            "predicted_price",
            "prediction_lower_90",
            "prediction_upper_90",
            "shock_probability",
            "predicted_shock",
            "anomaly_score",
        ]
        if column in output
    ]
    return {"predictions": output[columns].to_dict(orient="records")}


@app.post("/predict-from-history")
def predict_from_history(batch: HistoryBatch) -> dict[str, Any]:
    """Build features from recent canonical history and predict each latest series row."""

    try:
        standard = standardise_mandi_frame(pd.DataFrame(batch.records))
        clean, quarantine, _ = validate_mandi_data(standard)
        if clean.empty:
            raise ValueError("No valid history rows remain after validation")
        counts = clean.groupby(GROUP_COLUMNS, observed=True)["arrival_date"].nunique()
        eligible_groups = counts[counts >= 29].index
        if not len(eligible_groups):
            raise ValueError("Each predicted series requires at least 29 distinct dates")
        indexed = clean.set_index(GROUP_COLUMNS)
        clean = indexed.loc[indexed.index.isin(eligible_groups)].reset_index()
        engineered = build_supervised_frame(clean, drop_incomplete=False)
        latest = (
            engineered.sort_values("arrival_date").groupby(GROUP_COLUMNS, as_index=False).tail(1)
        )
        output = get_bundle().predict(latest)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    columns = [
        column
        for column in [
            *GROUP_COLUMNS,
            "arrival_date",
            "predicted_price",
            "prediction_lower_90",
            "prediction_upper_90",
            "shock_probability",
            "predicted_shock",
            "anomaly_score",
        ]
        if column in output
    ]
    result = output[columns].copy()
    if "arrival_date" in result:
        result["arrival_date"] = result["arrival_date"].astype(str)
    return {
        "predictions": result.to_dict(orient="records"),
        "quarantined_input_rows": len(quarantine),
    }


@app.post("/rank-markets")
def rank(batch: RankingBatch) -> dict[str, Any]:
    try:
        output = rank_markets(
            pd.DataFrame(batch.records),
            transport_cost_per_km_quintal=batch.transport_cost_per_km_quintal,
            risk_penalty=batch.risk_penalty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"rankings": output.to_dict(orient="records")}
