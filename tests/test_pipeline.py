import json

from agridecision.data.demo import generate_demo_mandi_data
from agridecision.data.quality import validate_mandi_data
from agridecision.data.schema import standardise_mandi_frame
from agridecision.features.tabular import build_supervised_frame
from agridecision.models.bundle import ModelBundle
from agridecision.models.training import train_model_suite


def test_demo_training_and_reload(tmp_path):
    raw = generate_demo_mandi_data(days=220, seed=7)
    clean, rejected, report = validate_mandi_data(standardise_mandi_frame(raw))
    assert rejected.empty
    assert report.rows_accepted == len(raw)
    supervised = build_supervised_frame(clean)
    result = train_model_suite(supervised, tmp_path)
    assert (tmp_path / "forecast_model.joblib").exists()
    assert (tmp_path / "backtest_predictions.csv").exists()
    assert result["metrics"]["evaluation"]["test_rows"] > 0
    contract = json.loads((tmp_path / "feature_contract.json").read_text())
    sample = supervised.tail(4)[contract["feature_columns"]]
    prediction = ModelBundle(tmp_path).predict(sample)
    assert prediction["predicted_price"].gt(0).all()
    assert prediction["shock_probability"].between(0, 1).all()
