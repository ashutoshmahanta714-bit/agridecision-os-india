import numpy as np
import pandas as pd

from agridecision.evaluation.slices import regression_slice_report
from agridecision.evaluation.split import rolling_origin_splits
from agridecision.monitoring.performance import evaluate_prediction_log


def test_rolling_splits_slice_and_delayed_outcomes():
    dates = pd.date_range("2025-01-01", periods=260)
    frame = pd.DataFrame({"arrival_date": dates, "value": np.arange(260)})
    splits = list(
        rolling_origin_splits(frame, minimum_train_days=120, validation_days=30, step_days=30)
    )
    assert len(splits) >= 3
    assert max(splits[0][0]) < min(splits[0][1])

    predictions = pd.DataFrame(
        {
            "market": ["A"] * 10,
            "actual_price": np.arange(10) + 100,
            "predicted_price": np.arange(10) + 101,
            "baseline_price": np.arange(10) + 105,
        }
    )
    slices = regression_slice_report(predictions)
    assert bool(slices.loc[0, "model_beats_baseline"])

    log = pd.DataFrame(
        {
            "market": ["A", "B"],
            "commodity": ["Onion", "Onion"],
            "target_date": ["2026-01-01", "2026-01-01"],
            "predicted_price": [100, 120],
            "shock_probability": [0.1, 0.8],
        }
    )
    outcomes = pd.DataFrame(
        {
            "market": ["A", "B"],
            "commodity": ["Onion", "Onion"],
            "target_date": ["2026-01-01", "2026-01-01"],
            "actual_price": [105, 125],
            "actual_shock": [0, 1],
        }
    )
    report = evaluate_prediction_log(log, outcomes)
    assert report["matched_predictions"] == 2
    assert report["forecast"]["mae"] == 5
