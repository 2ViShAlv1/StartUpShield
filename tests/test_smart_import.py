"""Tests for smart import (column detection, tenure derivation, event aggregation).

The cases here are modelled on what real exports actually look like -- Stripe,
HubSpot, and Zendesk all name things differently and none of them export the
canonical schema.
"""

import pandas as pd
import pytest

from src.smart_import import (
    CONFIDENCE_EXACT,
    aggregate_event_counts,
    apply_mapping,
    derive_tenure,
    detect_column_mapping,
    detect_signup_date_column,
    join_counts,
    mapping_summary,
    normalize,
)


def _stripe_export() -> pd.DataFrame:
    """A Stripe-style export: no column name matches our schema."""
    return pd.DataFrame(
        {
            "Customer ID": ["cus_1", "cus_2", "cus_3"],
            "Email": ["a@x.com", "b@x.com", "c@x.com"],
            "Created (UTC)": ["2024-03-15", "2025-01-10", "2023-11-01"],
            "Plan Name": ["Growth", "Starter", "Business"],
            "Monthly Recurring Revenue": [4900, 900, 12000],
        }
    )


def test_normalize_makes_names_comparable() -> None:
    """Spacing, case, and punctuation must not affect matching."""
    assert normalize("Monthly Spend") == normalize("monthly_spend") == "monthlyspend"
    assert normalize("Created (UTC)") == "createdutc"


def test_detects_real_export_columns_that_do_not_match_the_schema() -> None:
    """The whole point: a Stripe export must map without the user renaming anything."""
    mapping = detect_column_mapping(
        _stripe_export(), ["customer_id", "monthly_spend", "plan_type"]
    )

    assert mapping["customer_id"]["column"] == "Customer ID"
    assert mapping["monthly_spend"]["column"] == "Monthly Recurring Revenue"
    assert mapping["plan_type"]["column"] == "Plan Name"
    assert all(info["confidence"] for info in mapping.values())


def test_exact_canonical_names_are_reported_as_exact() -> None:
    """A file already in our schema should map with full confidence."""
    df = pd.DataFrame({"tenure": [1], "monthly_spend": [10.0], "plan_type": ["pro"]})

    mapping = detect_column_mapping(df, ["tenure", "monthly_spend", "plan_type"])

    assert all(info["confidence"] == CONFIDENCE_EXACT for info in mapping.values())


def test_unmatched_fields_return_none_rather_than_a_bad_guess() -> None:
    """A wrong silent mapping is worse than an empty one the user can fill in."""
    df = pd.DataFrame({"foo": [1], "bar": [2], "baz": [3]})

    mapping = detect_column_mapping(df, ["monthly_spend", "support_tickets"])

    assert mapping["monthly_spend"]["column"] is None
    assert mapping["support_tickets"]["column"] is None
    _, unmatched = mapping_summary(mapping)
    assert set(unmatched) == {"monthly_spend", "support_tickets"}


def test_one_column_is_not_claimed_by_two_fields() -> None:
    """Each source column may only satisfy one canonical field."""
    df = pd.DataFrame({"revenue": [100], "plan": ["pro"], "logins": [5]})

    mapping = detect_column_mapping(df)
    used = [info["column"] for info in mapping.values() if info["column"]]

    assert len(used) == len(set(used))


def test_more_specific_alias_wins_over_a_shorter_fragment() -> None:
    """"Monthly Recurring Revenue" must beat a bare "Revenue" column."""
    df = pd.DataFrame({"Revenue": [1], "Monthly Recurring Revenue": [2]})

    mapping = detect_column_mapping(df, ["monthly_spend"])

    assert mapping["monthly_spend"]["column"] == "Monthly Recurring Revenue"


def test_detects_a_signup_date_column_by_name_and_by_content() -> None:
    """Named date columns and oddly-named ones that hold dates both work."""
    assert detect_signup_date_column(_stripe_export()) == "Created (UTC)"

    odd = pd.DataFrame({"cust": ["a"], "activity date": ["2024-01-01"]})
    assert detect_signup_date_column(odd) == "activity date"

    assert detect_signup_date_column(pd.DataFrame({"a": [1], "b": [2]})) is None


def test_derive_tenure_counts_months_from_the_latest_date_in_the_file() -> None:
    """Tenure is measured against the export's own horizon, not the wall clock.

    Using the real current date would make a historical export's tenures drift
    upward every day it is re-run, which would silently change the churn score.
    """
    df = pd.DataFrame({"signup_date": ["2025-01-01", "2025-07-01", "2025-12-01"]})

    tenure = derive_tenure(df, "signup_date")

    assert list(tenure) == [11, 5, 1]
    assert tenure.dtype.kind == "i"


def test_derive_tenure_floors_at_one_and_survives_bad_dates() -> None:
    """Brand-new and unparseable rows must not become 0 or NaN."""
    df = pd.DataFrame({"signup_date": ["2025-12-01", "not a date", None]})

    tenure = derive_tenure(df, "signup_date")

    assert (tenure >= 1).all()
    assert tenure.notna().all()


def test_aggregate_event_counts_collapses_one_row_per_ticket() -> None:
    """Zendesk exports one row per ticket; the model needs one count per customer."""
    tickets = pd.DataFrame(
        {"requester": ["a@x.com", "a@x.com", "b@x.com"], "subject": ["x", "y", "z"]}
    )

    counts = aggregate_event_counts(tickets, "requester", "support_tickets")

    assert dict(zip(counts["requester"], counts["support_tickets"])) == {
        "a@x.com": 2,
        "b@x.com": 1,
    }


def test_aggregate_event_counts_rejects_a_missing_key_column() -> None:
    """A typo'd key should fail loudly, not produce an empty join."""
    with pytest.raises(ValueError, match="not found"):
        aggregate_event_counts(pd.DataFrame({"a": [1]}), "missing_key")


def test_join_counts_gives_customers_with_no_events_a_zero() -> None:
    """No tickets on record means zero tickets, not a dropped customer."""
    customers = pd.DataFrame({"email": ["a@x.com", "b@x.com", "c@x.com"]})
    counts = pd.DataFrame({"requester": ["a@x.com"], "support_tickets": [3]})

    joined = join_counts(customers, counts, "email", "requester", "support_tickets")

    assert list(joined["support_tickets"]) == [3, 0, 0]
    assert len(joined) == 3


def test_apply_mapping_builds_the_canonical_frame_end_to_end() -> None:
    """The full path: real export in, canonical schema out."""
    stripe = _stripe_export()
    mapping = detect_column_mapping(
        stripe, ["customer_id", "monthly_spend", "plan_type"]
    )
    chosen = {field: info["column"] for field, info in mapping.items()}

    result = apply_mapping(
        stripe,
        chosen,
        derive_tenure_from=detect_signup_date_column(stripe),
        defaults={"support_tickets": 0, "usage_frequency": 0},
    )

    assert set(result.columns) >= {
        "customer_id",
        "monthly_spend",
        "plan_type",
        "tenure",
        "support_tickets",
        "usage_frequency",
    }
    assert len(result) == len(stripe)
    assert (result["tenure"] >= 1).all()
    assert (result["support_tickets"] == 0).all()


def test_mapped_output_passes_the_real_validator() -> None:
    """Smart import must produce something `data_validation` actually accepts."""
    from src.data_validation import validate_customers

    rows = 20
    raw = pd.DataFrame(
        {
            "Customer ID": [f"cus_{i}" for i in range(rows)],
            "Created (UTC)": ["2024-06-01"] * rows,
            "Plan Name": ["Growth"] * rows,
            "Monthly Recurring Revenue": [49.0] * rows,
            "Login Count": [10] * rows,
            "Ticket Count": [1] * rows,
        }
    )
    mapping = detect_column_mapping(raw)
    chosen = {field: info["column"] for field, info in mapping.items()}
    canonical = apply_mapping(
        raw, chosen, derive_tenure_from=detect_signup_date_column(raw)
    )

    assert validate_customers(canonical) == []
