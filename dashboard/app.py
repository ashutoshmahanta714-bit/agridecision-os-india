"""Interactive portfolio dashboard for backtest results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import plotly.express as px
import streamlit as st

from agridecision.data.demo import generate_demo_mandi_data
from agridecision.data.quality import validate_mandi_data
from agridecision.data.schema import standardise_mandi_frame
from agridecision.features.tabular import build_supervised_frame
from agridecision.models.training import train_model_suite

st.set_page_config(page_title="AgriDecision OS", page_icon="🌾", layout="wide")
artifact_dir = Path(os.getenv("AGRIDECISION_ARTIFACT_DIR", "artifacts"))
metrics_path = artifact_dir / "metrics.json"
predictions_path = artifact_dir / "backtest_predictions.csv"

st.title("AgriDecision OS India")
st.caption(
    "Mandi price forecasting, price-shock risk, anomaly detection, and decision intelligence"
)


@st.cache_resource(show_spinner="Preparing the reproducible demonstration model…")
def build_demo_artifacts() -> Path:
    """Build deterministic synthetic artifacts when a deployment has no trained files."""

    destination = Path(gettempdir()) / "agridecision-demo-artifacts"
    metrics_file = destination / "metrics.json"
    predictions_file = destination / "backtest_predictions.csv"
    if metrics_file.exists() and predictions_file.exists():
        return destination

    raw = generate_demo_mandi_data(days=480, seed=42)
    accepted, _, _ = validate_mandi_data(standardise_mandi_frame(raw))
    supervised = build_supervised_frame(accepted, horizon_days=7, shock_threshold=0.15)
    train_model_suite(supervised, destination, horizon_days=7, random_state=42)
    return destination


if not metrics_path.exists() or not predictions_path.exists():
    artifact_dir = build_demo_artifacts()
    metrics_path = artifact_dir / "metrics.json"
    predictions_path = artifact_dir / "backtest_predictions.csv"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
predictions = pd.read_csv(predictions_path, parse_dates=["arrival_date"])

if metrics.get("data_provenance", {}).get("contains_synthetic_rows"):
    st.info(
        "Demonstration mode: the app is fully interactive, but these metrics use synthetic "
        "data and are not real-world performance claims."
    )

st.sidebar.header("Project status")
st.sidebar.success("Interactive 7-day onion pipeline")
st.sidebar.info("14- and 30-day models follow after validation on official historical data.")
st.sidebar.markdown(
    "[View source code](https://github.com/ashutoshmahanta714-bit/agridecision-os-india)"
)

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
view = (
    predictions
    if selected_market == "All markets"
    else predictions.query("market == @selected_market")
)

latest = view.sort_values("arrival_date").iloc[-1]
st.subheader("Latest backtest decision snapshot")
snapshot_left, snapshot_middle, snapshot_right = st.columns(3)
snapshot_left.metric("7-day predicted price", f"₹{latest['predicted_price']:,.0f}/q")
snapshot_middle.metric("Price-shock probability", f"{latest['shock_probability']:.1%}")
snapshot_right.metric("Anomaly score", f"{latest['anomaly_score']:.3f}")

trend = view.groupby("arrival_date", as_index=False)[
    ["actual_price", "predicted_price", "baseline_price"]
].mean()
trend_long = trend.melt("arrival_date", var_name="series", value_name="price")
st.plotly_chart(
    px.line(
        trend_long, x="arrival_date", y="price", color="series", title="Chronological backtest"
    ),
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
    px.histogram(
        view,
        x="shock_probability",
        color="actual_shock",
        nbins=30,
        title="Shock-risk calibration view",
    ),
    use_container_width=True,
)

st.subheader("Highest-risk observations")
st.dataframe(
    view.sort_values("shock_probability", ascending=False).head(20),
    use_container_width=True,
    hide_index=True,
)
