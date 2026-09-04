"""Remove only known generated outputs; source code and raw data are untouched."""

from pathlib import Path


GENERATED = [
    Path("artifacts/backtest_predictions.csv"),
    Path("artifacts/feature_contract.json"),
    Path("artifacts/forecast_model.joblib"),
    Path("artifacts/risk_model.joblib"),
    Path("artifacts/anomaly_model.joblib"),
    Path("artifacts/metrics.json"),
    Path("artifacts/synthetic_demo_raw.csv"),
    Path("artifacts/clean_mandi_prices.csv"),
    Path("artifacts/quarantined_rows.csv"),
    Path("artifacts/data_quality_report.json"),
    Path("artifacts/market_coverage.csv"),
    Path("artifacts/training_readiness.json"),
    Path("artifacts/snapshot_manifest.json"),
    Path("artifacts/supervised_features.csv"),
]


for path in GENERATED:
    if path.is_file():
        path.unlink()
        print(f"Removed {path}")
