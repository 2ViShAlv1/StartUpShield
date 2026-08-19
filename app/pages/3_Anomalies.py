"""Streamlit anomalies page."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import company_selector, empty_state, get_company_bundle, page_setup

page_setup("Anomalies")
company_name = company_selector()
bundle = get_company_bundle(company_name)
anomalies = bundle["anomalies"]

st.title("Anomaly Detection")
st.caption(f"{company_name} — unusual daily movements flagged by Isolation Forest.")

if anomalies.empty or "is_anomaly" not in anomalies.columns:
    empty_state("no daily metrics for this company.")
    st.stop()

anomalies = anomalies.copy()
anomalies["date"] = pd.to_datetime(anomalies["date"])
flagged = anomalies[anomalies["is_anomaly"]]
recent_window = anomalies.tail(30)

metric_columns = st.columns(4)
metric_columns[0].metric("Days monitored", f"{len(anomalies):,}")
metric_columns[1].metric("Anomalies flagged", f"{len(flagged):,}")
metric_columns[2].metric("In last 30 days", f"{int(recent_window['is_anomaly'].sum())}")
metric_columns[3].metric(
    "Flag rate", f"{anomalies['is_anomaly'].mean():.1%}"
)

st.divider()

metric_choice = st.selectbox(
    "Metric",
    [column for column in ["revenue", "active_users", "signups"] if column in anomalies.columns],
)

figure, axis = plt.subplots(figsize=(11, 4))
axis.plot(anomalies["date"], anomalies[metric_choice], color="#1f4e79", lw=1.1, label=metric_choice)
if not flagged.empty:
    axis.scatter(
        flagged["date"],
        flagged[metric_choice],
        color="#c0392b",
        s=42,
        zorder=5,
        label="flagged anomaly",
    )
axis.set_ylabel(metric_choice.replace("_", " "))
axis.grid(alpha=0.25)
axis.legend(fontsize=8)
figure.autofmt_xdate()
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)

st.divider()
st.subheader("Flagged days (most anomalous first)")

if flagged.empty:
    st.success("No anomalous days detected for this company.")
else:
    columns_to_show = [
        column
        for column in ["date", "revenue", "active_users", "signups", "anomaly_score"]
        if column in flagged.columns
    ]
    st.dataframe(
        flagged.nsmallest(20, "anomaly_score")[columns_to_show],
        width="stretch",
        hide_index=True,
    )
    st.caption("Lower anomaly score = more unusual.")

st.info(
    "Model note: the detector is tuned for recall (it recovered 5/5 injected test "
    "anomalies). Its precision is bounded by the 3% contamination setting, so expect "
    "some flagged days to be ordinary business variation.",
    icon="ℹ️",
)
