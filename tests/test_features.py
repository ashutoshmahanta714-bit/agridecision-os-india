import pandas as pd

from agridecision.data.demo import generate_demo_mandi_data
from agridecision.features.tabular import build_supervised_frame


def test_exact_seven_day_target_and_shifted_lag():
    raw = generate_demo_mandi_data(days=120, end_date="2026-08-01", seed=1)
    one_market = raw.loc[raw["market"] == raw["market"].iloc[0]].copy()
    supervised = build_supervised_frame(one_market, horizon_days=7)
    row = supervised.iloc[20]
    source_date = row["arrival_date"]
    expected_future = one_market.loc[
        one_market["arrival_date"] == source_date + pd.Timedelta(days=7), "modal_price"
    ].iloc[0]
    expected_lag = one_market.loc[
        one_market["arrival_date"] == source_date - pd.Timedelta(days=1), "modal_price"
    ].iloc[0]
    assert row["target_modal_price"] == expected_future
    assert row["price_lag_1"] == expected_lag
