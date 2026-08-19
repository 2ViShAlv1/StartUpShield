"""Streamlit overview page for StartupShield AI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    company_selector,
    empty_state,
    get_company_bundle,
    page_setup,
    risk_badge,
)

PRIORITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "🟢",
}

page_setup("Overview")
company_name = company_selector()
bundle = get_company_bundle(company_name)
assessment = bundle["assessment"]

st.title("🛡️ StartupShield AI")
st.caption(
    "One explainable risk score built from churn, sentiment, anomalies, and forecasting."
)

left_column, right_column = st.columns([2, 3])

with left_column:
    st.subheader(company_name)
    risk_badge(assessment)
    st.caption(f"Company profile: **{bundle['profile']}**")

with right_column:
    st.subheader("Why this score")
    st.write(assessment["explanation_text"])

    contributions = assessment.get("contributions", {})
    if contributions:
        st.caption("Points contributed to the score, by signal")
        st.bar_chart(
            pd.DataFrame({"points": contributions}).sort_values("points", ascending=False),
            horizontal=True,
            height=200,
        )

st.divider()

signals = assessment["signals"]
metric_columns = st.columns(4)
metric_columns[0].metric("Mean churn probability", f"{signals['churn_prob']:.0%}")
metric_columns[1].metric("Review positivity", f"{signals['sentiment_positivity']:.0%}")
metric_columns[2].metric("Anomalous days (last 30)", f"{signals['anomaly_flag_ratio'] * 30:.0f}")
metric_columns[3].metric("Forecast downside", f"{signals['forecast_deviation']:.1%}")

st.divider()
st.subheader("Recommended actions")

recommendations = bundle.get("recommendations", [])
if not recommendations:
    empty_state("no recommendations generated for this company.")
else:
    for recommendation in recommendations[:3]:
        icon = PRIORITY_ICONS.get(recommendation["priority"], "•")
        with st.container(border=True):
            st.markdown(f"{icon} **{recommendation['action']}**")
            st.caption(recommendation["reason"])

    if len(recommendations) > 3:
        st.caption(
            f"{len(recommendations) - 3} more on the "
            "**Risk & Recommendations** page."
        )

st.divider()
st.caption(
    "Navigate the pages in the sidebar for the per-model detail behind this score."
)
