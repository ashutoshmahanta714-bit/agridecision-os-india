import pandas as pd
from fastapi.testclient import TestClient

import agridecision.api.app as api_module


class FakeBundle:
    def predict(self, frame):
        result = frame.copy()
        result["predicted_price"] = 2000.0
        result["prediction_lower_90"] = 1800.0
        result["prediction_upper_90"] = 2200.0
        result["shock_probability"] = 0.2
        result["predicted_shock"] = 0
        result["anomaly_score"] = 0.1
        return result


def test_api_health_prediction_history_and_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(api_module, "get_bundle", lambda: FakeBundle())
    client = TestClient(api_module.app)

    assert client.get("/health").json()["status"] == "not_ready"
    assert client.get("/metadata").status_code == 503

    direct = client.post(
        "/predict",
        json={"records": [{"market": "A", "commodity": "Onion"}]},
    )
    assert direct.status_code == 200
    assert direct.json()["predictions"][0]["predicted_price"] == 2000

    history = []
    for date in pd.date_range("2026-06-01", periods=35):
        history.append(
            {
                "state": "Odisha",
                "district": "Khordha",
                "market": "A",
                "commodity": "Onion",
                "variety": "Other",
                "arrival_date": str(date.date()),
                "min_price": 1800,
                "max_price": 2200,
                "modal_price": 2000,
            }
        )
    history_response = client.post("/predict-from-history", json={"records": history})
    assert history_response.status_code == 200
    assert len(history_response.json()["predictions"]) == 1

    ranking = client.post(
        "/rank-markets",
        json={
            "records": [
                {"market": "A", "predicted_price": 2200, "distance_km": 20},
                {"market": "B", "predicted_price": 2100, "distance_km": 10},
            ]
        },
    )
    assert ranking.status_code == 200
    assert ranking.json()["rankings"][0]["recommendation_rank"] == 1
