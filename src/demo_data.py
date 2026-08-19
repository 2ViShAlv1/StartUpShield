"""Demo company portfolios for the StartupShield AI dashboard.

`data/raw/churn.csv` and `data/raw/reviews.csv` have no company column -- they are
flat customer and review tables. The dashboard needs per-company slices, so this
module deterministically assigns customers and reviews to the three demo companies
defined in the master spec.

HONESTY NOTE: these are *curated demo portfolios*, not random samples. Customers are
assigned using raw behavioural features (usage frequency, support tickets, tenure) --
never using the churn model's own output -- so the model still has to discover the risk
rather than being handed a pre-sorted answer. Reviews are assigned by their sentiment
label. This is disclosed in the dashboard UI and in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("data/raw")

DEMO_COMPANIES = {
    "GreenLeaf SaaS": {
        "profile": "Healthy",
        "timeseries_file": "timeseries_greenleaf_saas.csv",
        # Engaged customers: high usage, few tickets.
        "customer_filter": "healthy",
        # Review mix: (negative, neutral, positive) proportions.
        "review_mix": (0.10, 0.20, 0.70),
        "n_customers": 220,
        "n_reviews": 180,
    },
    "MixedCo": {
        "profile": "Mixed",
        "timeseries_file": "timeseries_mixedco.csv",
        "customer_filter": "mixed",
        "review_mix": (0.35, 0.30, 0.35),
        "n_customers": 200,
        "n_reviews": 160,
    },
    "RedFlag Analytics": {
        "profile": "At Risk",
        "timeseries_file": "timeseries_redflag_analytics.csv",
        # Disengaged customers: low usage, more support friction.
        "customer_filter": "at_risk",
        "review_mix": (0.65, 0.20, 0.15),
        "n_customers": 190,
        "n_reviews": 170,
    },
}


def company_names() -> list[str]:
    """Return the demo company names in dashboard display order."""
    return list(DEMO_COMPANIES)


def load_company_customers(
    company_name: str,
    raw_dir: Path = RAW_DATA_DIR,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the customer slice belonging to one demo company.

    Selection uses raw behavioural columns only (usage_frequency, support_tickets,
    tenure). The churn model never influences who lands in which portfolio.
    """
    config = _company_config(company_name)
    churn_df = pd.read_csv(raw_dir / "churn.csv")

    engagement_rank = (
        churn_df["usage_frequency"].rank(pct=True)
        - churn_df["support_tickets"].rank(pct=True) * 0.5
    )
    ordered = churn_df.assign(_engagement=engagement_rank).sort_values(
        "_engagement",
        ascending=False,
        kind="mergesort",
    )

    n_customers = min(config["n_customers"], len(ordered))
    if config["customer_filter"] == "healthy":
        selected = ordered.head(n_customers)
    elif config["customer_filter"] == "at_risk":
        selected = ordered.tail(n_customers)
    else:
        middle_start = max((len(ordered) - n_customers) // 2, 0)
        selected = ordered.iloc[middle_start : middle_start + n_customers]

    return (
        selected.drop(columns="_engagement")
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def load_company_reviews(
    company_name: str,
    raw_dir: Path = RAW_DATA_DIR,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the review slice belonging to one demo company."""
    config = _company_config(company_name)
    reviews_df = pd.read_csv(raw_dir / "reviews.csv")

    rng = np.random.default_rng(seed)
    total_reviews = config["n_reviews"]
    label_order = ["negative", "neutral", "positive"]

    sampled_frames = []
    for label, proportion in zip(label_order, config["review_mix"]):
        pool = reviews_df[reviews_df["sentiment_label"] == label]
        if pool.empty:
            continue
        take = min(int(round(total_reviews * proportion)), len(pool))
        sampled_frames.append(pool.sample(n=take, random_state=seed))

    if not sampled_frames:
        return reviews_df.head(0)

    combined = pd.concat(sampled_frames, ignore_index=True)
    # Spread reviews across the last 90 days so the trend chart has an x-axis.
    review_dates = pd.Timestamp("2025-12-31") - pd.to_timedelta(
        rng.integers(0, 90, size=len(combined)),
        unit="D",
    )
    combined["review_date"] = review_dates
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_company_timeseries(
    company_name: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Return the daily business-metric series for one demo company."""
    config = _company_config(company_name)
    timeseries_df = pd.read_csv(raw_dir / config["timeseries_file"])
    timeseries_df["date"] = pd.to_datetime(timeseries_df["date"])
    return timeseries_df.sort_values("date").reset_index(drop=True)


def company_profile(company_name: str) -> str:
    """Return the human-readable profile label for a demo company."""
    return _company_config(company_name)["profile"]


def _company_config(company_name: str) -> dict:
    """Look up a demo company's configuration, failing clearly if unknown."""
    if company_name not in DEMO_COMPANIES:
        raise ValueError(
            f"Unknown demo company {company_name!r}. "
            f"Available: {list(DEMO_COMPANIES)}"
        )
    return DEMO_COMPANIES[company_name]
