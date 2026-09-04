"""Interactive portfolio dashboard for backtest results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="AgriDecision OS", page_icon="🌾", layout="wide")
artifact_dir = Path(os.getenv("AGRIDECISION_ARTIFACT_DIR", "artifacts"))
metrics_path = artifact_dir / "metrics.json"
predictions_path = artifact_dir / "backtest_predictions.csv"

st.title("AgriDecision OS India")
st.caption("Mandi price forecasting, price-shock risk, anomaly detection, and decision intelligence")

if not metrics_path.exists() or not predictions_path.exists():
    st.warning("Model artifacts are missing. Run `make demo` or train on validated real data first.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
predictions = pd.read_csv(predictions_path, parse_dates=["arrival_date"])

if metrics.get("data_provenance", {}).get("contains_synthetic_rows"):
    st.info("Demonstration mode: these metrics use synthetic data and are not portfolio performance claims.")

forecast = metrics["forecast_model"]
baseline = metrics["seasonal_naive_baseline"]
risk = metrics["price_shock_model"]

left, middle, right, fourth = st.columns(4)
left.metric("Forecast MAE", f"₹{forecast['mae']:,.0f}")
middle.metric("Baseline MAE", f"₹{baseline['mae']:,.0f}")
right.metric("Shock PR-AUC", "N/A" if risk["pr_auc"] is None else f"{risk['pr_auc']:.3f}")
fourth.metric("Test rows", f"{metrics['evaluation']['test_rows']:,}")

market_options = sorted(predictions["market"].dropna().unique())
selected_market = st.selectbox("Market", ["All markets", *market_options])
view = predictions if selected_market == "All markets" else predictions.query("market == @selected_market")

trend = view.groupby("arrival_date", as_index=False)[["actual_price", "predicted_price", "baseline_price"]].mean()
trend_long = trend.melt("arrival_date", var_name="series", value_name="price")
st.plotly_chart(
    px.line(trend_long, x="arrival_date", y="price", color="series", title="Chronological backtest"),
    use_container_width=True,
)

chart_left, chart_right = st.columns(2)
chart_left.plotly_chart(
    px.scatter(
        view,
        x="actual_price",
        y="predicted_price",
        color="market",
        title="Actual vs predicted price",
    ),
    use_container_width=True,
)
chart_right.plotly_chart(
    px.histogram(view, x="shock_probability", color="actual_shock", nbins=30, title="Shock-risk calibration view"),
    use_container_width=True,
)

st.subheader("Highest-risk observations")
st.dataframe(
    view.sort_values("shock_probability", ascending=False).head(20),
    use_container_width=True,
    hide_index=True,
)

