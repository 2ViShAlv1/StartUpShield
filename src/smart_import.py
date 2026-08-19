"""Smart import: turn a company's real exports into the canonical schema.

The hard part of onboarding is never running the models -- it is that no tool
exports the exact columns they need. A Stripe export calls it `amount`, a CRM calls
it `MRR`, nobody exports `tenure` at all (they export a signup date), and support
tickets arrive as one row per ticket rather than a count per customer.

This module removes that friction in three ways:

1. `detect_column_mapping` guesses which of the user's columns means what, so they
   confirm a guess instead of typing names.
2. `derive_tenure` computes tenure in months from any date column.
3. `aggregate_event_counts` collapses a raw event/ticket export into per-customer
   counts that can be joined onto the customer table.

Every guess is returned with a confidence level and is meant to be overridden in the
UI -- the module suggests, it never silently decides.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# Canonical field -> alias fragments seen in real exports, most specific first.
# Matching is done on normalized names, so "Monthly Spend" and "monthly_spend" are
# the same string by the time they get here.
COLUMN_ALIASES: dict[str, list[str]] = {
    "customer_id": [
        "customerid", "custid", "userid", "accountid", "subscriberid",
        "clientid", "id", "customer", "email",
    ],
    "tenure": [
        "tenure", "tenuremonths", "monthsactive", "monthssubscribed",
        "accountagemonths", "customerage", "ageinmonths",
    ],
    "monthly_spend": [
        "monthlyspend", "mrr", "monthlyrecurringrevenue", "monthlyrevenue",
        "monthlyamount", "subscriptionamount", "recurringamount",
        "planprice", "amount", "revenue", "spend", "price",
    ],
    "usage_frequency": [
        "usagefrequency", "loginfrequency", "logincount", "sessions",
        "sessioncount", "activedays", "eventcount", "usagecount",
        "visits", "logins", "usage",
    ],
    "support_tickets": [
        "supporttickets", "ticketcount", "tickets", "numtickets",
        "supportrequests", "casecount", "complaints", "issues",
    ],
    "plan_type": [
        "plantype", "plan", "plyname", "planname", "subscriptiontier",
        "tier", "productname", "package", "subscription", "product",
    ],
    "review_text": [
        "reviewtext", "review", "comment", "comments", "feedback",
        "description", "body", "message", "text", "content", "verbatim",
    ],
    "date": ["date", "day", "reportdate", "activitydate", "period", "ds"],
    "active_users": [
        "activeusers", "dau", "dailyactiveusers", "users", "activeaccounts",
    ],
    "signups": ["signups", "newusers", "registrations", "newcustomers", "acquisitions"],
}

# Fields whose value is a date we may need to convert into tenure.
SIGNUP_DATE_ALIASES = [
    "signupdate", "createdat", "created", "startdate", "subscriptionstart",
    "joindate", "firstseen", "registrationdate", "customersince", "activationdate",
]

CONFIDENCE_EXACT = "exact"
CONFIDENCE_LIKELY = "likely"
CONFIDENCE_GUESS = "guess"


def normalize(name: str) -> str:
    """Reduce a column name to comparable form: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_column_mapping(
    df: pd.DataFrame,
    targets: list[str] | None = None,
) -> dict[str, dict]:
    """Guess which of the user's columns maps to each canonical field.

    Args:
        df: The user's raw uploaded frame.
        targets: Canonical fields to look for. Defaults to every known field.

    Returns:
        `{canonical_field: {"column": <matched column or None>,
        "confidence": <exact|likely|guess|None>}}`. A field with no plausible match
        maps to None rather than to a bad guess -- a wrong silent mapping is far
        worse than an unfilled dropdown.
    """
    targets = targets or list(COLUMN_ALIASES)
    normalized_columns = {normalize(column): column for column in df.columns}
    claimed: set[str] = set()
    mapping: dict[str, dict] = {}

    for field in targets:
        aliases = COLUMN_ALIASES.get(field, [])
        match, confidence = _best_match(field, aliases, normalized_columns, claimed)
        if match is not None:
            claimed.add(match)
        mapping[field] = {"column": match, "confidence": confidence}

    return mapping


def detect_signup_date_column(df: pd.DataFrame) -> str | None:
    """Find a column that looks like a signup/created date, for deriving tenure."""
    normalized_columns = {normalize(column): column for column in df.columns}

    for alias in SIGNUP_DATE_ALIASES:
        if alias in normalized_columns:
            return normalized_columns[alias]

    for normalized, original in normalized_columns.items():
        if "date" in normalized or "created" in normalized or "since" in normalized:
            if _looks_like_dates(df[original]):
                return original

    return None


def derive_tenure(
    df: pd.DataFrame,
    date_column: str,
    reference_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Convert a signup-date column into tenure in whole months.

    Args:
        df: Frame containing the date column.
        date_column: Column holding signup/created dates.
        reference_date: "Today" for the calculation. Defaults to the latest date in
            the column, not the real current date, so a historical export produces
            stable tenures instead of inflating with the wall clock.

    Returns:
        Integer month counts, floored at 1 (a brand-new customer has 1 month, not 0,
        which also keeps the churn module's `tenure + 1` denominators well-behaved).
    """
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    if reference_date is None:
        reference_date = parsed.max()
    if pd.isna(reference_date):
        return pd.Series(np.ones(len(df), dtype=int), index=df.index)

    months = (reference_date - parsed).dt.days / 30.44
    return months.round().fillna(1).clip(lower=1).astype(int)


def aggregate_event_counts(
    events_df: pd.DataFrame,
    key_column: str,
    output_name: str = "event_count",
) -> pd.DataFrame:
    """Collapse a raw event export into one count per customer.

    Support tools and analytics export one row per ticket/session, but the models
    need one number per customer. This is that collapse.

    Returns:
        Two-column frame: the key, and the count named `output_name`.
    """
    if key_column not in events_df.columns:
        raise ValueError(f"Key column {key_column!r} not found in the event file.")

    counts = (
        events_df.groupby(events_df[key_column].astype(str))
        .size()
        .reset_index(name=output_name)
    )
    counts.columns = [key_column, output_name]
    return counts


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    derive_tenure_from: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a canonical frame from the user's columns and their chosen mapping.

    Args:
        df: The user's raw frame.
        mapping: `{canonical_field: user_column_or_None}`.
        derive_tenure_from: Date column to compute `tenure` from, when the user has
            no tenure column of their own.
        defaults: Fallback values for fields the user genuinely does not have
            (e.g. `{"support_tickets": 0}`).

    Returns:
        A frame using canonical column names.
    """
    defaults = defaults or {}
    result = pd.DataFrame(index=df.index)

    for field, source_column in mapping.items():
        if source_column and source_column in df.columns:
            result[field] = df[source_column]

    if "tenure" not in result.columns and derive_tenure_from:
        result["tenure"] = derive_tenure(df, derive_tenure_from)

    for field, value in defaults.items():
        if field not in result.columns:
            result[field] = value

    return result


def join_counts(
    customers: pd.DataFrame,
    counts: pd.DataFrame,
    customer_key: str,
    counts_key: str,
    target_column: str,
) -> pd.DataFrame:
    """Attach per-customer counts onto the customer table.

    Customers with no matching events get 0, not NaN: no support tickets on record
    means zero tickets, and treating that as missing would drop real customers from
    the score.
    """
    merged = customers.copy()
    lookup = counts.set_index(counts.iloc[:, 0].astype(str))[target_column]
    merged[target_column] = (
        merged[customer_key].astype(str).map(lookup).fillna(0).astype(int)
    )
    return merged


def mapping_summary(mapping: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Split a detected mapping into (matched fields, unmatched fields)."""
    matched = [field for field, info in mapping.items() if info["column"]]
    unmatched = [field for field, info in mapping.items() if not info["column"]]
    return matched, unmatched


def _best_match(
    field: str,
    aliases: list[str],
    normalized_columns: dict[str, str],
    claimed: set[str],
) -> tuple[str | None, str | None]:
    """Find the best unclaimed column for one canonical field."""
    normalized_field = normalize(field)

    # 1. The column is literally named the canonical field.
    if normalized_field in normalized_columns:
        candidate = normalized_columns[normalized_field]
        if candidate not in claimed:
            return candidate, CONFIDENCE_EXACT

    # 2. The column exactly equals a known alias.
    for alias in aliases:
        if alias in normalized_columns:
            candidate = normalized_columns[alias]
            if candidate not in claimed:
                return candidate, CONFIDENCE_LIKELY

    # 3. An alias appears inside the column name. Longer aliases first so
    #    "monthlyrecurringrevenue" wins over the bare "revenue" fragment.
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) < 4:
            continue
        for normalized, original in normalized_columns.items():
            if original in claimed:
                continue
            if alias in normalized:
                return original, CONFIDENCE_GUESS

    return None, None


def _looks_like_dates(series: pd.Series, sample_size: int = 25) -> bool:
    """True when most of a sample parses as dates."""
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() > 0.8)
