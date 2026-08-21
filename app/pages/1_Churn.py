"""Streamlit churn page."""

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
    explanation_method_label,
    get_company_bundle,
    page_setup,
)

page_setup("Churn")
company_name = company_selector()
bundle = get_company_bundle(company_name)
customers = bundle["customers"]

st.title("Churn Prediction")
st.caption(f"{company_name} — per-customer cancellation risk from the LightGBM model.")

if customers.empty or "churn_prob" not in customers.columns:
    empty_state("no customer records for this company.")
    st.stop()

churn_probabilities = pd.to_numeric(customers["churn_prob"], errors="coerce").dropna()
if churn_probabilities.empty:
    empty_state("the churn model could not be loaded.")
    st.stop()

high_risk_mask = churn_probabilities >= 0.5
metric_columns = st.columns(4)
metric_columns[0].metric("Customers", f"{len(customers):,}")
metric_columns[1].metric("Mean churn probability", f"{churn_probabilities.mean():.0%}")
metric_columns[2].metric("High risk (≥50%)", f"{int(high_risk_mask.sum()):,}")
metric_columns[3].metric("Highest single risk", f"{churn_probabilities.max():.0%}")

st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Risk distribution")
    histogram = pd.cut(
        churn_probabilities,
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        include_lowest=True,
    ).value_counts().sort_index()
    st.bar_chart(histogram, height=280)

with right_column:
    top_factors = bundle["assessment"].get("top_factors", [])
    method_name, method_caption = explanation_method_label(bundle["assessment"])
    st.subheader("Top risk factors" + (f" ({method_name})" if method_name else ""))

    if top_factors:
        factor_frame = pd.DataFrame(
            {
                "impact": {
                    name.split("__", 1)[-1].replace("_", " "): value
                    for name, value in top_factors
                }
            }
        )
        st.bar_chart(factor_frame, horizontal=True, height=280)
        st.caption(
            f"{method_caption} "
            "Note: `monthly spend` and `plan type` are usage-minute proxies, "
            "not independent pricing signals (see README)."
        )
    else:
        st.caption(
            "Factor breakdown unavailable — the churn model could not be "
            "interrogated for this company. The risk score above is unaffected."
        )

st.divider()
st.subheader("Highest-risk customers")

display_columns = [
    column
    for column in [
        "customer_id",
        "churn_prob",
        "tenure",
        "usage_frequency",
        "support_tickets",
        "plan_type",
        "monthly_spend",
    ]
    if column in customers.columns
]
top_customers = customers.nlargest(25, "churn_prob")[display_columns].copy()
if "customer_id" in top_customers.columns:
    top_customers["customer_id"] = top_customers["customer_id"].astype(str).str[:10] + "…"

st.dataframe(
    top_customers,
    width="stretch",
    hide_index=True,
    column_config={
        "churn_prob": st.column_config.ProgressColumn(
            "Churn probability", min_value=0.0, max_value=1.0, format="%.0f%%"
        )
    },
)
