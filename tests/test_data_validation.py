"""Tests for uploaded-data validation.

These matter more than most: this module is the boundary between an arbitrary
user-supplied CSV and the trained models, so a gap here becomes a crash during a
live demo.
"""

import pandas as pd

from src.data_validation import (
    MIN_CUSTOMERS,
    forecast_readiness,
    validate_customers,
    validate_reviews,
    validate_timeseries,
)


def _valid_customers(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": list(range(1, n_rows + 1)),
            "monthly_spend": [50.0] * n_rows,
            "usage_frequency": [10] * n_rows,
            "support_tickets": [0] * n_rows,
            "plan_type": ["pro"] * n_rows,
        }
    )


def _valid_timeseries(n_rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n_rows).astype(str),
            "revenue": [1000.0] * n_rows,
            "active_users": [100] * n_rows,
            "signups": [5] * n_rows,
        }
    )


def test_valid_uploads_produce_no_errors() -> None:
    """A well-formed upload must pass cleanly."""
    assert validate_customers(_valid_customers()) == []
    assert validate_reviews(pd.DataFrame({"review_text": ["good"] * 10})) == []
    assert validate_timeseries(_valid_timeseries()) == []


def test_customer_validation_does_not_require_churn_labels() -> None:
    """An onboarding company wants a prediction, not to supply the answer."""
    customers = _valid_customers()
    assert "churn_label" not in customers.columns
    assert validate_customers(customers) == []


def test_missing_columns_are_all_reported_at_once() -> None:
    """Users should see every missing column, not discover them one at a time."""
    errors = validate_customers(pd.DataFrame({"tenure": [1, 2, 3]}))

    assert len(errors) == 1
    message = errors[0]
    for column in ("monthly_spend", "usage_frequency", "support_tickets", "plan_type"):
        assert column in message


def test_too_few_rows_is_flagged_with_the_actual_threshold() -> None:
    """The message must tell the user what number to aim for."""
    errors = validate_customers(_valid_customers(n_rows=3))

    assert any(str(MIN_CUSTOMERS) in error for error in errors)


def test_unparseable_text_in_a_numeric_column_is_caught(tmp_path) -> None:
    """Stray text that survives CSV parsing must be reported as non-numeric.

    Written through a real CSV round-trip rather than by assigning into a typed
    column, because that is how a bad value actually reaches us -- pandas infers an
    object dtype on read, which is the case validation has to handle.
    """
    customers = _valid_customers()
    customers["monthly_spend"] = customers["monthly_spend"].astype(object)
    customers.loc[0, "monthly_spend"] = "unknown"
    customers.loc[1, "tenure"] = -5

    csv_path = tmp_path / "customers.csv"
    customers.to_csv(csv_path, index=False)
    errors = validate_customers(pd.read_csv(csv_path))

    assert any("monthly_spend" in error and "non-numeric" in error for error in errors)
    assert any("tenure" in error and "negative" in error for error in errors)


def test_na_placeholders_are_caught_as_missing_not_silently_accepted(tmp_path) -> None:
    """"N/A" reads back as NaN, so it must be flagged as missing, not ignored.

    This is the case that would otherwise slip through: pandas converts "N/A",
    "null", and blank cells to NaN on read, so checking only for unparseable
    strings would let an entire column of placeholders reach the model and produce
    a confident-looking score built on nothing.
    """
    customers = _valid_customers()
    customers["monthly_spend"] = customers["monthly_spend"].astype(object)
    customers.loc[0, "monthly_spend"] = "N/A"

    csv_path = tmp_path / "customers.csv"
    customers.to_csv(csv_path, index=False)
    errors = validate_customers(pd.read_csv(csv_path))

    assert any("monthly_spend" in error and "missing" in error for error in errors)


def test_invalid_churn_labels_are_rejected_when_supplied() -> None:
    """If a label column is provided it still has to be binary."""
    customers = _valid_customers()
    customers["churn_label"] = [2] * len(customers)

    assert any("churn_label" in error for error in validate_customers(customers))


def test_timeseries_duplicate_dates_and_bad_values_are_caught() -> None:
    """One row per day, all metrics non-negative numbers."""
    timeseries = _valid_timeseries()
    timeseries.loc[1, "date"] = timeseries.loc[0, "date"]
    timeseries.loc[2, "revenue"] = -100

    errors = validate_timeseries(timeseries)

    assert any("duplicate" in error for error in errors)
    assert any("revenue" in error and "negative" in error for error in errors)


def test_unparseable_dates_are_caught() -> None:
    """A date column of free text must not reach the forecaster."""
    timeseries = _valid_timeseries()
    timeseries.loc[0, "date"] = "last tuesday"

    assert any("date" in error and "parse" in error for error in validate_timeseries(timeseries))


def test_empty_frames_return_a_single_clear_error() -> None:
    """Empty uploads should say so plainly rather than listing every column."""
    assert validate_customers(pd.DataFrame()) == ["Customer file is empty."]
    assert validate_reviews(pd.DataFrame()) == ["Review file is empty."]
    assert validate_timeseries(pd.DataFrame()) == ["Time-series file is empty."]


def test_forecast_readiness_scales_the_holdout_to_available_history() -> None:
    """Short histories still forecast, with a smaller backtest; very short ones don't."""
    can_long, holdout_long, _ = forecast_readiness(90)
    can_mid, holdout_mid, _ = forecast_readiness(40)
    can_short, holdout_short, message = forecast_readiness(20)

    assert (can_long, holdout_long) == (True, 30)
    assert (can_mid, holdout_mid) == (True, 14)
    assert (can_short, holdout_short) == (False, 0)
    assert "30" in message and "excluded" in message
