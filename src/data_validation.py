"""Validation for uploaded company data.

A real onboarding flow means a stranger uploads a CSV they made in Excel. This
module is the boundary between "arbitrary user file" and "safe to feed into the
trained models" -- it never raises; it returns a list of plain-language problems so
the UI can show all of them at once instead of one cryptic stack trace at a time.
"""

from __future__ import annotations

import pandas as pd

from src.churn_module import BASE_FEATURE_COLUMNS, TARGET_COLUMN
from src.anomaly_module import SERIES_COLUMNS


REQUIRED_REVIEW_COLUMNS = ["review_text"]
REQUIRED_TIMESERIES_COLUMNS = ["date", *SERIES_COLUMNS]

MIN_CUSTOMERS = 10
MIN_REVIEWS = 5
MIN_TIMESERIES_ROWS = 14

CHURN_NUMERIC_COLUMNS = ["tenure", "monthly_spend", "usage_frequency", "support_tickets"]


def validate_customers(df: pd.DataFrame) -> list[str]:
    """Check an uploaded customer table against the churn model's schema.

    The `churn_label` column is intentionally NOT required -- an onboarding
    customer wants a *prediction*, not to re-supply the answer.
    """
    errors: list[str] = []
    if df is None or df.empty:
        return ["Customer file is empty."]

    missing = [column for column in BASE_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Customer file is missing required columns: {', '.join(missing)}.")
        return errors

    if len(df) < MIN_CUSTOMERS:
        errors.append(
            f"Only {len(df)} customer rows found; at least {MIN_CUSTOMERS} are needed "
            "for the churn score to be meaningful."
        )

    for column in CHURN_NUMERIC_COLUMNS:
        errors.extend(_numeric_column_errors(df, column))

    if TARGET_COLUMN in df.columns:
        bad_labels = ~df[TARGET_COLUMN].dropna().isin([0, 1])
        if bad_labels.any():
            errors.append(f"'{TARGET_COLUMN}' must be 0 or 1 where provided.")

    if "tenure" in df.columns and pd.to_numeric(df["tenure"], errors="coerce").lt(0).any():
        errors.append("'tenure' cannot be negative.")

    return errors


def validate_reviews(df: pd.DataFrame) -> list[str]:
    """Check an uploaded review table against the sentiment model's schema."""
    errors: list[str] = []
    if df is None or df.empty:
        return ["Review file is empty."]

    missing = [column for column in REQUIRED_REVIEW_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Review file is missing required columns: {', '.join(missing)}.")
        return errors

    if len(df) < MIN_REVIEWS:
        errors.append(
            f"Only {len(df)} reviews found; at least {MIN_REVIEWS} are needed for a "
            "usable sentiment signal."
        )

    empty_text = df["review_text"].isna() | (df["review_text"].astype(str).str.strip() == "")
    if empty_text.all():
        errors.append("Every 'review_text' value is empty.")

    return errors


def validate_timeseries(df: pd.DataFrame) -> list[str]:
    """Check an uploaded daily-metrics table against the anomaly/forecast schema."""
    errors: list[str] = []
    if df is None or df.empty:
        return ["Time-series file is empty."]

    missing = [column for column in REQUIRED_TIMESERIES_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Time-series file is missing required columns: {', '.join(missing)}.")
        return errors

    if len(df) < MIN_TIMESERIES_ROWS:
        errors.append(
            f"Only {len(df)} daily rows found; at least {MIN_TIMESERIES_ROWS} are "
            "needed for anomaly detection to run."
        )

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        errors.append(f"'date' has {int(parsed_dates.isna().sum())} value(s) that don't parse as dates.")
    elif parsed_dates.duplicated().any():
        errors.append("'date' has duplicate values -- expected one row per day.")

    for column in SERIES_COLUMNS:
        errors.extend(_numeric_column_errors(df, column))
        numeric = pd.to_numeric(df[column], errors="coerce")
        if (numeric < 0).any():
            errors.append(
                f"Column '{column}' has negative value(s), which is not a valid count/amount."
            )

    return errors


def _numeric_column_errors(df: pd.DataFrame, column: str) -> list[str]:
    """Report unparseable and missing values in a column that must be numeric.

    Both cases are reported, and separately, because they reach us differently and
    a user fixes them differently. `pd.read_csv` converts the common placeholders
    ("N/A", "NA", "null", empty cells) straight to NaN, so a file full of "N/A"
    arrives indistinguishable from one with blank cells -- if only unparseable
    strings were checked, that whole class of bad upload would pass validation
    silently and produce a confident-looking score built on missing data.
    """
    errors: list[str] = []
    coerced = pd.to_numeric(df[column], errors="coerce")

    unparseable = coerced.isna() & df[column].notna()
    if unparseable.any():
        errors.append(f"Column '{column}' has {int(unparseable.sum())} non-numeric value(s).")

    missing = df[column].isna()
    if missing.any():
        errors.append(
            f"Column '{column}' has {int(missing.sum())} missing/blank value(s) "
            "(this includes cells like 'N/A' or 'null')."
        )

    return errors


def forecast_readiness(n_rows: int) -> tuple[bool, int, str]:
    """Decide whether -- and how -- to run forecasting for a given history length.

    Returns:
        (can_forecast, holdout_days, message). `message` explains the decision so
        the UI can show it even when forecasting is skipped.
    """
    if n_rows >= 60:
        return True, 30, "Using a 30-day backtest (60+ days of history available)."
    if n_rows >= 30:
        return True, 14, "Using a 14-day backtest (30-59 days of history available)."
    return (
        False,
        0,
        f"Only {n_rows} days of history -- need at least 30 to forecast. "
        "The forecast signal is excluded from this company's risk score.",
    )
