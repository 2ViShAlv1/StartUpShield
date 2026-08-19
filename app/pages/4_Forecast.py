"""Streamlit forecast page."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import company_selector, empty_state, get_company_bundle, page_setup

page_setup("Forecast")
company_name = company_selector()
bundle = get_company_bundle(company_name)
forecast = bundle["forecast"]
anomalies = bundle["anomalies"]

st.title("Revenue Forecast")
st.caption(f"{company_name} — 30-day backtest and 30-day forward projection.")

if forecast.empty:
    empty_state("no forecast model available for this company.")
    st.stop()

forecast = forecast.copy()
forecast["date"] = pd.to_datetime(forecast["date"])
backtest = forecast[forecast["actual"].notna()]
projection = forecast[forecast["actual"].isna()]

if not backtest.empty:
    errors = backtest["actual"] - backtest["forecast"]
    mape = float(np.mean(np.abs(errors / backtest["actual"])) * 100)
    mae = float(np.mean(np.abs(errors)))
else:
    mape = mae = float("nan")

current_level = float(backtest["actual"].mean()) if not backtest.empty else float("nan")
projected_level = float(projection["forecast"].mean()) if not projection.empty else float("nan")
projected_change = (
    (projected_level - current_level) / current_level
    if np.isfinite(current_level) and current_level
    else float("nan")
)

metric_columns = st.columns(4)
metric_columns[0].metric("Backtest MAPE", f"{mape:.2f}%" if np.isfinite(mape) else "n/a")
metric_columns[1].metric("Backtest MAE", f"{mae:,.0f}" if np.isfinite(mae) else "n/a")
metric_columns[2].metric(
    "Current daily revenue", f"{current_level:,.0f}" if np.isfinite(current_level) else "n/a"
)
metric_columns[3].metric(
    "Projected (next 30d)",
    f"{projected_level:,.0f}" if np.isfinite(projected_level) else "n/a",
    delta=f"{projected_change:+.1%}" if np.isfinite(projected_change) else None,
)

st.divider()

figure, axis = plt.subplots(figsize=(11, 4.4))

if not anomalies.empty and "revenue" in anomalies.columns:
    history = anomalies.tail(90)
    axis.plot(
        pd.to_datetime(history["date"]),
        history["revenue"],
        color="#666",
        lw=1.0,
        label="recent actual (90d)",
    )

if not backtest.empty:
    axis.plot(backtest["date"], backtest["actual"], color="#000", lw=1.8, label="actual (held out)")
    axis.plot(
        backtest["date"], backtest["forecast"], color="#1f4e79", lw=1.6, ls="--", label="backtest forecast"
    )

if not projection.empty:
    axis.plot(
        projection["date"],
        projection["forecast"],
        color="#d9821f",
        lw=1.8,
        ls="--",
        label="forward projection",
    )
    axis.fill_between(
        projection["date"],
        projection["lower_bound"],
        projection["upper_bound"],
        color="#d9821f",
        alpha=0.18,
        label="95% confidence band",
    )

axis.set_ylabel("daily revenue")
axis.grid(alpha=0.25)
axis.legend(fontsize=8, ncol=2)
figure.autofmt_xdate()
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)

st.divider()
st.subheader("Forward projection")
st.dataframe(
    projection[["date", "forecast", "lower_bound", "upper_bound"]],
    width="stretch",
    hide_index=True,
)

st.info(
    "Every forecast row carries a confidence band. On the demo data the model beats a "
    "last-value-carried-forward baseline for all three companies — though much of that "
    "gain comes from capturing weekly seasonality rather than model sophistication.",
    icon="ℹ️",
)
