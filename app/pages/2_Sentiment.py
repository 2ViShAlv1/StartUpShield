"""Streamlit sentiment page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Streamlit does not guarantee the app directory is importable from every page,
# so put it on sys.path before importing the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import company_selector, empty_state, get_company_bundle, page_setup

page_setup("Sentiment")
company_name = company_selector()
bundle = get_company_bundle(company_name)
reviews = bundle["reviews"]

st.title("Customer Sentiment")
st.caption(f"{company_name} — review tone from the TF-IDF + Logistic Regression model.")

if reviews.empty or "label" not in reviews.columns:
    empty_state("no reviews for this company.")
    st.stop()

label_counts = reviews["label"].value_counts()
positivity = pd.to_numeric(reviews.get("score"), errors="coerce").dropna()

metric_columns = st.columns(4)
metric_columns[0].metric("Reviews analysed", f"{len(reviews):,}")
metric_columns[1].metric("Negative", f"{int(label_counts.get('negative', 0)):,}")
metric_columns[2].metric("Neutral", f"{int(label_counts.get('neutral', 0)):,}")
metric_columns[3].metric(
    "Mean positivity",
    f"{positivity.mean():.0%}" if len(positivity) else "n/a",
)

st.divider()
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Sentiment mix")
    ordered_counts = label_counts.reindex(
        ["negative", "neutral", "positive"]
    ).fillna(0).astype(int)
    st.bar_chart(ordered_counts, height=280)

with right_column:
    st.subheader("Positivity trend")
    if "review_date" in reviews.columns and len(positivity):
        trend = (
            reviews.assign(score=pd.to_numeric(reviews["score"], errors="coerce"))
            .dropna(subset=["score"])
            .set_index(pd.to_datetime(reviews["review_date"]))
            .sort_index()["score"]
            .resample("W")
            .mean()
        )
        st.line_chart(trend, height=280)
        st.caption("Weekly mean positivity score (1.0 = fully positive).")
    else:
        st.caption("No dated reviews available for a trend.")

st.divider()
st.subheader("Flagged negative reviews")

negative_reviews = reviews[reviews["label"] == "negative"]
if negative_reviews.empty:
    st.success("No negative reviews detected for this company.")
else:
    columns_to_show = [
        column
        for column in ["review_date", "review_text", "score", "star_rating"]
        if column in negative_reviews.columns
    ]
    st.dataframe(
        negative_reviews.nsmallest(15, "score")[columns_to_show],
        width="stretch",
        hide_index=True,
    )

st.info(
    "Model note: this classifier scored a perfect macro-F1 on its test set because "
    "the review dataset is template-generated. Treat it as a working integration, "
    "not evidence of real-world accuracy.",
    icon="ℹ️",
)
