import numpy as np

from agridecision.advanced.satellite import ndvi, summarise_field_index
from agridecision.advanced.simulation import simulate_market_policy
from agridecision.advanced.survival import kaplan_meier


def test_satellite_indices_and_summary():
    index = ndvi(np.array([0.8, 0.4]), np.array([0.2, 0.4]))
    assert np.allclose(index, [0.6, 0.0])
    summary = summarise_field_index(index)
    assert 0 <= summary["healthy_fraction"] <= 1


def test_kaplan_meier_is_monotonic():
    curve = kaplan_meier(np.array([1, 2, 2, 4]), np.array([1, 1, 0, 1]))
    assert curve["survival_probability"].is_monotonic_decreasing


def test_bandit_simulator_runs():
    history, agent = simulate_market_policy({"A": 100.0, "B": 150.0}, steps=30)
    assert len(history) == 30
    assert agent.counts.sum() == 30
