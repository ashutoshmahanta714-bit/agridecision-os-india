import pandas as pd

from agridecision.decision.optimization import optimise_market_allocation
from agridecision.decision.ranking import rank_markets


def test_ranking_and_allocation_respect_total_quantity():
    forecasts = pd.DataFrame(
        {
            "market": ["A", "B", "C"],
            "predicted_price": [2200, 2150, 2300],
            "distance_km": [50, 10, 100],
            "shock_probability": [0.1, 0.05, 0.5],
            "anomaly_score": [0.1, 0.2, 0.9],
            "capacity": [80, 80, 80],
        }
    )
    ranked = rank_markets(forecasts)
    assert ranked["recommendation_rank"].min() == 1
    allocation = optimise_market_allocation(ranked, available_quantity=100)
    assert allocation.success
    assert abs(allocation.allocations["allocated_quantity"].sum() - 100) < 1e-8
