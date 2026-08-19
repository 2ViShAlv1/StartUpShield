"""Streamlit risk and recommendations page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import (
    company_selector,
    empty_state,
    get_company_bundle,
    get_config,
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

page_setup("Risk & Recommendations")
company_name = company_selector()
bundle = get_company_bundle(company_name)
assessment = bundle["assessment"]

st.title("Risk & Recommendations")
st.caption(f"{company_name} — how the score was built, and what to do about it.")

risk_badge(assessment)
st.write(assessment["explanation_text"])

st.divider()
st.subheader("How the score was built")

weights = get_config().get("risk_weights", {})
contributions = assessment.get("contributions", {})
signals = assessment["signals"]

breakdown = pd.DataFrame(
    [
        {
            "Signal": "Churn",
            "Raw value": f"{signals['churn_prob']:.0%} mean churn probability",
            "Weight": weights.get("w_churn", 0.35),
            "Points": contributions.get("Churn", 0.0),
        },
        {
            "Signal": "Sentiment",
            "Raw value": f"{signals['sentiment_positivity']:.0%} review positivity",
            "Weight": weights.get("w_sentiment", 0.20),
            "Points": contributions.get("Sentiment", 0.0),
        },
        {
            "Signal": "Anomalies",
            "Raw value": f"{signals['anomaly_flag_ratio'] * 30:.0f} of last 30 days flagged",
            "Weight": weights.get("w_anomaly", 0.20),
            "Points": contributions.get("Anomalies", 0.0),
        },
        {
            "Signal": "Forecast",
            "Raw value": f"{signals['forecast_deviation']:.1%} projected downside",
            "Weight": weights.get("w_forecast", 0.25),
            "Points": contributions.get("Forecast", 0.0),
        },
    ]
)

st.dataframe(
    breakdown,
    width="stretch",
    hide_index=True,
    column_config={
        "Weight": st.column_config.NumberColumn(format="%.2f"),
        "Points": st.column_config.ProgressColumn(
            "Points contributed", min_value=0.0, max_value=35.0, format="%.1f"
        ),
    },
)
st.caption(f"Total: **{assessment['risk_score']:.1f} / 100**")

st.divider()
st.subheader("Top churn drivers (SHAP)")

top_factors = assessment.get("top_factors", [])
if not top_factors:
    st.caption("Explainability unavailable for this model.")
else:
    factor_frame = pd.DataFrame(
        {
            "SHAP impact": {
                name.split("__", 1)[-1].replace("_", " "): value
                for name, value in top_factors
            }
        }
    )
    st.bar_chart(factor_frame, horizontal=True, height=240)
    st.caption(
        "Positive values push churn risk up; negative values pull it down. "
        "`monthly spend` and `plan type` are derived from usage minutes, so read "
        "them as usage proxies rather than pricing effects."
    )

st.divider()
st.subheader("All recommended actions")

recommendations = bundle.get("recommendations", [])
if not recommendations:
    empty_state("no recommendations generated.")
else:
    for recommendation in recommendations:
        icon = PRIORITY_ICONS.get(recommendation["priority"], "•")
        with st.container(border=True):
            st.markdown(
                f"{icon} **{recommendation['action']}**  \n"
                f"<span style='color:#666;font-size:13px;'>{recommendation['reason']}</span>",
                unsafe_allow_html=True,
            )

st.divider()
with st.expander("Known limitations of this score"):
    st.markdown(
        """
- **The formula is a linear weighted combination, not a learned model.** There is no
  labelled "company failed" dataset to train a meta-model on, so the weights encode a
  product judgement rather than a fitted result.
- **Signals are rescaled before weighting.** Raw churn probability spans ~0.1–0.9 while
  the anomaly ratio tops out near 0.15, so without rescaling churn would dominate the
  score regardless of the configured weights.
- **Three churn features are usage proxies.** `monthly_spend`, `plan_type`, and
  `support_tickets` are derived from the source dataset rather than being independent
  business signals.
- **Review and time-series data are synthetic.** Customer/review portfolios are curated
  demo slices chosen from behavioural features, never from model output.
        """
    )
