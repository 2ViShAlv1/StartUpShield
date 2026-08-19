"""Streamlit page for scoring a company from freshly uploaded data.

This is the real onboarding path: a founder who is not one of the three curated
demo companies uploads their own CSVs and gets a live risk score. Churn and
sentiment reuse the pretrained global models (inference only); anomaly detection
and forecasting are fit fresh on the uploaded time series, synchronously, right
here -- see `pipeline.score_uploaded_company` for why that split is the right one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd
import streamlit as st

from shared import get_config, get_models, page_setup, risk_badge
from src import data_validation, pipeline, smart_import


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
PRIORITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "🟢",
}

page_setup("Upload Your Company", icon="📤")
st.title("📤 Upload Your Company")
st.caption(
    "Not one of the demo companies? Upload your own data and get a live risk score "
    "using the same trained models."
)

with st.expander("What this does, in plain terms", expanded=False):
    st.markdown(
        """
- **Churn & Sentiment** use the models already trained on Phase 3-4 data — your data
  is only used for *prediction*, not retraining. This works even for a completely new
  company the models have never seen (categorical values outside the training set are
  handled gracefully).
- **Anomalies & Forecast** are fit fresh on *your* time series, on the spot — every
  company's normal day is different, so there is no useful "pretrained" version of this.
- Nothing you upload is saved to disk or sent anywhere outside this session.
        """
    )

st.divider()
st.subheader("1. Download templates (optional)")

template_columns = st.columns(3)
template_files = [
    ("customers_template.csv", "Customer data", template_columns[0]),
    ("reviews_template.csv", "Reviews (optional)", template_columns[1]),
    ("timeseries_template.csv", "Daily metrics (optional)", template_columns[2]),
]
for filename, label, column in template_files:
    template_path = TEMPLATES_DIR / filename
    if template_path.exists():
        column.download_button(
            f"⬇ {label} template",
            data=template_path.read_bytes(),
            file_name=filename,
            mime="text/csv",
            width="stretch",
        )

st.divider()
st.subheader("2. Upload your files")

company_name = st.text_input("Company name", value="", placeholder="e.g. Acme SaaS")

upload_columns = st.columns(3)
with upload_columns[0]:
    st.markdown("**Customer data** *(required)*")
    st.caption("`tenure, monthly_spend, usage_frequency, support_tickets, plan_type`")
    customers_file = st.file_uploader("Customers CSV", type="csv", key="customers_upload", label_visibility="collapsed")
with upload_columns[1]:
    st.markdown("**Reviews** *(optional)*")
    st.caption("`review_text`")
    reviews_file = st.file_uploader("Reviews CSV", type="csv", key="reviews_upload", label_visibility="collapsed")
with upload_columns[2]:
    st.markdown("**Daily metrics** *(optional, 30+ days for forecasting)*")
    st.caption("`date, revenue, active_users, signups`")
    timeseries_file = st.file_uploader("Time-series CSV", type="csv", key="timeseries_upload", label_visibility="collapsed")


# --- Smart column mapping -------------------------------------------------------
# Real exports (Stripe, HubSpot, Zendesk) never use our column names, and none of
# them export `tenure` at all. Rather than making the user rename columns in Excel
# first, detect what their columns mean and let them confirm or correct the guess.
CANONICAL_FIELDS = [
    "customer_id",
    "tenure",
    "monthly_spend",
    "usage_frequency",
    "support_tickets",
    "plan_type",
]
FIELDS_SAFE_TO_DEFAULT = {"support_tickets": 0, "usage_frequency": 0}

raw_customers = None
mapped_customers = None

if customers_file is not None:
    try:
        raw_customers = pd.read_csv(customers_file)
    except Exception as exc:
        st.error(f"Could not read the customer file: {exc}")

if raw_customers is not None and not raw_customers.empty:
    st.divider()
    st.subheader("3. Confirm what your columns mean")

    detected = smart_import.detect_column_mapping(raw_customers, CANONICAL_FIELDS)
    signup_date_column = smart_import.detect_signup_date_column(raw_customers)
    matched, unmatched = smart_import.mapping_summary(detected)

    if len(matched) == len(CANONICAL_FIELDS):
        st.success(f"All {len(matched)} fields detected automatically — review and score.")
    else:
        st.info(
            f"Detected {len(matched)} of {len(CANONICAL_FIELDS)} fields. "
            "Set the rest below (or leave them blank to use a safe default)."
        )

    column_options = ["— not in my data —"] + list(raw_customers.columns)
    chosen_mapping: dict[str, str | None] = {}

    mapping_columns = st.columns(2)
    for index, field in enumerate(CANONICAL_FIELDS):
        info = detected[field]
        default_index = (
            column_options.index(info["column"]) if info["column"] in column_options else 0
        )
        with mapping_columns[index % 2]:
            selection = st.selectbox(
                f"**{field}**"
                + (f"  ·  detected ({info['confidence']})" if info["column"] else ""),
                column_options,
                index=default_index,
                key=f"map_{field}",
            )
            chosen_mapping[field] = None if selection == column_options[0] else selection

    derive_from = None
    if chosen_mapping.get("tenure") is None and signup_date_column:
        use_derived = st.checkbox(
            f"Calculate **tenure** in months from `{signup_date_column}`",
            value=True,
            help="Most exports have a signup/created date rather than a tenure count.",
        )
        derive_from = signup_date_column if use_derived else None

    missing_after_mapping = [
        field
        for field in CANONICAL_FIELDS
        if chosen_mapping.get(field) is None
        and field != "customer_id"
        and not (field == "tenure" and derive_from)
        and field not in FIELDS_SAFE_TO_DEFAULT
    ]

    mapped_customers = smart_import.apply_mapping(
        raw_customers,
        chosen_mapping,
        derive_tenure_from=derive_from,
        defaults=FIELDS_SAFE_TO_DEFAULT,
    )

    if missing_after_mapping:
        st.warning(
            "Still unmapped: **"
            + "**, **".join(missing_after_mapping)
            + "** — these have no sensible default, so scoring is blocked until they are set."
        )
    else:
        defaulted = [
            field for field in FIELDS_SAFE_TO_DEFAULT if chosen_mapping.get(field) is None
        ]
        if defaulted:
            st.caption(
                "Defaulted to 0: " + ", ".join(f"`{f}`" for f in defaulted)
                + ". The score will not reflect those signals."
            )
        with st.expander("Preview the mapped data"):
            st.dataframe(mapped_customers.head(10), width="stretch", hide_index=True)

st.divider()
can_score = mapped_customers is not None and not missing_after_mapping if raw_customers is not None else False
score_clicked = st.button("Score my company", type="primary", disabled=not can_score)
if customers_file is None:
    st.caption("Upload customer data to enable scoring.")

if score_clicked:
    resolved_name = company_name.strip() or "Uploaded Company"
    errors: list[str] = []

    customers_df = mapped_customers
    if customers_df is None:
        errors.append("Could not read the customer file.")
    else:
        errors.extend(data_validation.validate_customers(customers_df))

    reviews_df = pd.DataFrame()
    if reviews_file is not None:
        try:
            reviews_df = pd.read_csv(reviews_file)
            errors.extend(data_validation.validate_reviews(reviews_df))
        except Exception as exc:
            errors.append(f"Could not read the reviews file: {exc}")

    timeseries_df = pd.DataFrame()
    if timeseries_file is not None:
        try:
            timeseries_df = pd.read_csv(timeseries_file)
            errors.extend(data_validation.validate_timeseries(timeseries_df))
        except Exception as exc:
            errors.append(f"Could not read the time-series file: {exc}")

    if errors:
        st.error("Fix these before scoring:")
        for error in errors:
            st.markdown(f"- {error}")
        st.session_state.pop("upload_result", None)
    else:
        with st.spinner("Scoring... (fitting anomaly/forecast models on your data)"):
            try:
                result = pipeline.score_uploaded_company(
                    resolved_name,
                    customers_df,
                    reviews_df,
                    timeseries_df,
                    models=get_models(),
                    config=get_config(),
                )
                st.session_state["upload_result"] = result
            except Exception as exc:
                st.session_state.pop("upload_result", None)
                st.error(f"Scoring failed: {exc}")

result = st.session_state.get("upload_result")
if result:
    st.divider()
    st.subheader(f"Results — {result['company_name']}")
    assessment = result["assessment"]

    left_column, right_column = st.columns([2, 3])
    with left_column:
        risk_badge(assessment)
    with right_column:
        st.write(assessment["explanation_text"])

    data_quality = result["data_quality"]
    st.caption(
        f"Scored from {data_quality['customers']} customers, "
        f"{data_quality['reviews']} reviews, {data_quality['timeseries_days']} days of metrics. "
        f"{data_quality['forecast_note']}"
    )

    st.divider()
    signal_columns = st.columns(4)
    signals = assessment["signals"]
    signal_columns[0].metric("Mean churn probability", f"{signals['churn_prob']:.0%}")
    signal_columns[1].metric("Review positivity", f"{signals['sentiment_positivity']:.0%}")
    signal_columns[2].metric("Anomalous days (last 30)", f"{signals['anomaly_flag_ratio'] * 30:.0f}")
    signal_columns[3].metric("Forecast downside", f"{signals['forecast_deviation']:.1%}")

    st.divider()
    st.subheader("Recommended actions")
    for recommendation in result["recommendations"]:
        icon = PRIORITY_ICONS.get(recommendation["priority"], "•")
        with st.container(border=True):
            st.markdown(f"{icon} **{recommendation['action']}**")
            st.caption(recommendation["reason"])

    st.divider()
    report_json = json.dumps(
        {
            "company_name": result["company_name"],
            "risk_score": assessment["risk_score"],
            "risk_label": assessment["risk_label"],
            "signals": assessment["signals"],
            "explanation_text": assessment["explanation_text"],
            "recommendations": result["recommendations"],
            "data_quality": data_quality,
        },
        indent=2,
    )
    st.download_button(
        "⬇ Download this report (JSON)",
        data=report_json,
        file_name=f"{result['company_name'].lower().replace(' ', '_')}_risk_report.json",
        mime="application/json",
    )
