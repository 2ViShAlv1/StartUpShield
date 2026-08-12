"""Generate Phase 1 fallback datasets for StartupShield AI."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/config.yaml")
SAAS_CHURN_TRAIN_PATH = Path("train.csv")
SAAS_CHURN_TEST_PATH = Path("test_.csv")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load project configuration from YAML.

    Args:
        path: Location of the YAML config file.

    Returns:
        Parsed configuration values.
    """
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn rows using the Phase 1 fallback logic.

    Args:
        n_customers: Number of customers to generate.
        seed: Random seed for reproducible output.

    Returns:
        DataFrame with the required churn schema.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n_customers)
    monthly_spend = np.clip(rng.normal(70, 25, size=n_customers), 10, 200)
    usage_frequency = rng.poisson(12, size=n_customers)
    plan_type = rng.choice(
        ["basic", "pro", "enterprise"],
        size=n_customers,
        p=[0.50, 0.35, 0.15],
    )

    plan_risk = np.select(
        [plan_type == "basic", plan_type == "pro", plan_type == "enterprise"],
        [0.35, 0.05, -0.20],
    )
    early_tenure_risk = (18 - np.minimum(tenure, 18)) / 18
    low_usage_risk = (12 - np.minimum(usage_frequency, 12)) / 12
    ticket_lambda = 1.3 + 2.2 * early_tenure_risk + 1.5 * low_usage_risk
    support_tickets = rng.poisson(ticket_lambda)

    logits = (
        -2.0
        + 2.2 * early_tenure_risk
        + 1.6 * low_usage_risk
        + 0.35 * support_tickets
        + plan_risk
        + rng.normal(0, 0.45, size=n_customers)
    )
    churn_probability = 1 / (1 + np.exp(-logits))
    churn_label = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:05d}" for index in range(1, n_customers + 1)],
            "tenure": tenure,
            "monthly_spend": monthly_spend.round(2),
            "usage_frequency": usage_frequency,
            "support_tickets": support_tickets,
            "plan_type": plan_type,
            "churn_label": churn_label,
        }
    )


def _hash_identifier(value: str) -> str:
    """Hash a customer identifier before storing it in project data.

    Args:
        value: Raw identifier from a source dataset.

    Returns:
        Stable SHA-256 hash string.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def load_real_saas_churn_data(
    train_path: Path = SAAS_CHURN_TRAIN_PATH,
    test_path: Path = SAAS_CHURN_TEST_PATH,
) -> pd.DataFrame | None:
    """Load and anonymize the Kaggle SaaS churn dataset if files exist.

    Args:
        train_path: Path to the extracted training CSV.
        test_path: Path to the extracted test CSV.

    Returns:
        Phase-compatible churn DataFrame, or None if source files are missing.
    """
    if not train_path.exists() or not test_path.exists():
        return None

    raw_df = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)],
        ignore_index=True,
    )
    required_columns = {
        "Customer_ID",
        "Account_Age_Days",
        "Login_Frequency",
        "Daily_Usage_Mins",
        "Last_Support_Ticket",
        "Churn",
    }
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        raise ValueError(f"SaaS churn files are missing columns: {missing_columns}")

    login_frequency_map = {
        "Daily": 30,
        "Weekly": 4,
        "Monthly": 1,
        "Rarely": 0,
    }
    support_text = raw_df["Last_Support_Ticket"].astype(str).str.lower()
    support_risk_terms = [
        "support",
        "bug",
        "issue",
        "confusing",
        "cancel",
        "waiting",
        "problem",
        "error",
        "broken",
    ]
    risky_ticket = support_text.str.contains("|".join(support_risk_terms), regex=True)
    usage_minutes = raw_df["Daily_Usage_Mins"].astype(float)

    # The source is SaaS-like but does not include spend, ticket counts, or plan.
    # These proxy columns preserve the master spec schema for later modules.
    transformed_df = pd.DataFrame(
        {
            "customer_id": raw_df["Customer_ID"].map(_hash_identifier),
            "tenure": np.maximum((raw_df["Account_Age_Days"] / 30).round().astype(int), 1),
            "monthly_spend": np.clip(25 + usage_minutes * 0.85, 10, 200).round(2),
            "usage_frequency": raw_df["Login_Frequency"].map(login_frequency_map).fillna(0).astype(int),
            "support_tickets": np.where(risky_ticket, 1, 0),
            "plan_type": pd.cut(
                usage_minutes,
                bins=[-1, 25, 80, np.inf],
                labels=["basic", "pro", "enterprise"],
            ).astype(str),
            "churn_label": raw_df["Churn"].astype(int),
        }
    )
    return transformed_df


def generate_reviews_data(n_reviews: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate balanced synthetic review data for sentiment modeling.

    Args:
        n_reviews: Number of review rows to generate.
        seed: Random seed for reproducible output.

    Returns:
        DataFrame with review text, star rating, and sentiment label.
    """
    rng = np.random.default_rng(seed)
    templates = {
        "positive": [
            "The {noun} is reliable and the team responds quickly.",
            "Excellent {noun}, smooth onboarding, and helpful support.",
            "We love the {noun}; it saves time every week.",
            "Great experience with fast fixes and clear communication.",
            "The product feels stable, useful, and worth the price.",
        ],
        "neutral": [
            "The {noun} works, but some workflows still feel slow.",
            "Average experience overall, with a few helpful features.",
            "Support answered eventually and the product is okay.",
            "The {noun} is usable, though reporting could improve.",
            "Nothing major went wrong, but nothing stood out either.",
        ],
        "negative": [
            "The {noun} keeps failing and support has been slow.",
            "Poor experience with confusing billing and missing features.",
            "We had repeated issues, outages, and delayed responses.",
            "The product is frustrating and our team may cancel soon.",
            "Critical bugs blocked our workflow for several days.",
        ],
    }
    nouns = [
        "dashboard",
        "billing flow",
        "analytics module",
        "customer portal",
        "automation feature",
        "integration",
        "reporting tool",
    ]
    labels = np.array(["positive", "neutral", "negative"])
    sampled_labels = np.resize(labels, n_reviews)
    rng.shuffle(sampled_labels)

    rows = []
    for index, label in enumerate(sampled_labels, start=1):
        text = rng.choice(templates[label]).format(noun=rng.choice(nouns))
        if rng.random() < 0.25:
            text = f"{text} {rng.choice(templates[label]).format(noun=rng.choice(nouns))}"
        star_rating = {
            "positive": int(rng.choice([4, 5], p=[0.35, 0.65])),
            "neutral": 3,
            "negative": int(rng.choice([1, 2], p=[0.60, 0.40])),
        }[label]
        rows.append(
            {
                "review_id": f"REV-{index:05d}",
                "review_text": text,
                "star_rating": star_rating,
                "sentiment_label": label,
            }
        )

    return pd.DataFrame(rows)


def generate_company_timeseries(
    company_name: str,
    start_date: str,
    n_days: int = 365,
    base_revenue: float = 10000.0,
    growth_rate: float = 0.001,
    weekly_seasonality_amp: float = 0.15,
    noise_std: float = 0.05,
    n_anomalies: int = 3,
    anomaly_magnitude: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic SaaS business metrics for one company.

    Args:
        company_name: Company name written into the dataset.
        start_date: First date in YYYY-MM-DD format.
        n_days: Number of daily rows to generate.
        base_revenue: Starting daily revenue.
        growth_rate: Daily compounding growth rate.
        weekly_seasonality_amp: Strength of weekly seasonality.
        noise_std: Gaussian noise as a fraction of the signal.
        n_anomalies: Number of injected anomalies.
        anomaly_magnitude: Fractional spike or drop for injected anomalies.
        seed: Random seed for reproducible output.

    Returns:
        DataFrame with date, revenue, active_users, signups, and anomaly labels.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_index = np.arange(n_days)

    trend = base_revenue * np.power(1 + growth_rate, day_index)
    weekly = 1 + weekly_seasonality_amp * np.sin(2 * np.pi * day_index / 7)
    revenue = trend * weekly * (1 + rng.normal(0, noise_std, size=n_days))
    revenue = np.clip(revenue, 500, None)

    is_injected_anomaly = np.zeros(n_days, dtype=bool)
    eligible_days = np.arange(30, n_days - 30)
    anomaly_days = rng.choice(eligible_days, size=n_anomalies, replace=False)
    for anomaly_day in anomaly_days:
        direction = rng.choice([-1, 1])
        revenue[anomaly_day] *= 1 + direction * anomaly_magnitude
        is_injected_anomaly[anomaly_day] = True

    active_users = revenue / 18 + rng.normal(0, revenue.std() / 45, size=n_days)
    active_users = np.clip(active_users, 20, None).round().astype(int)
    signups = active_users * 0.045 + rng.normal(0, 4, size=n_days)
    signups = np.clip(signups, 0, None).round().astype(int)

    return pd.DataFrame(
        {
            "company_name": company_name,
            "date": dates.date.astype(str),
            "revenue": revenue.round(2),
            "active_users": active_users,
            "signups": signups,
            "is_injected_anomaly": is_injected_anomaly,
        }
    )


def write_phase1_datasets(output_dir: Path, seed: int = 42) -> None:
    """Write all Phase 1 fallback datasets to the raw data directory.

    Args:
        output_dir: Directory where raw CSV files should be saved.
        seed: Random seed for reproducible output.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    real_churn_df = load_real_saas_churn_data()
    if real_churn_df is not None:
        churn_df = real_churn_df
        logging.info("Using anonymized Kaggle SaaS churn data from train.csv and test_.csv")
    else:
        churn_df = generate_churn_data(seed=seed)
        logging.warning("Using synthetic churn fallback because Kaggle SaaS files were not found")
    churn_df.to_csv(output_dir / "churn.csv", index=False)
    logging.info("Wrote %s", output_dir / "churn.csv")

    reviews_df = generate_reviews_data(seed=seed)
    reviews_df.to_csv(output_dir / "reviews.csv", index=False)
    logging.info("Wrote %s", output_dir / "reviews.csv")

    company_configs = [
        {
            "company_name": "GreenLeaf SaaS",
            "base_revenue": 13500.0,
            "growth_rate": 0.0018,
            "n_anomalies": 0,
            "anomaly_magnitude": 0.0,
            "seed": seed + 1,
        },
        {
            "company_name": "RedFlag Analytics",
            "base_revenue": 11500.0,
            "growth_rate": -0.0012,
            "n_anomalies": 4,
            "anomaly_magnitude": 0.45,
            "seed": seed + 2,
        },
        {
            "company_name": "MixedCo",
            "base_revenue": 10000.0,
            "growth_rate": 0.0001,
            "n_anomalies": 1,
            "anomaly_magnitude": 0.25,
            "seed": seed + 3,
        },
    ]
    for config in company_configs:
        company_df = generate_company_timeseries(start_date="2025-01-01", **config)
        slug = config["company_name"].lower().replace(" ", "_")
        path = output_dir / f"timeseries_{slug}.csv"
        company_df.to_csv(path, index=False)
        logging.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate StartupShield Phase 1 data.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the Phase 1 dataset generator.

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    raw_data_path = Path(config["paths"]["raw_data"])
    write_phase1_datasets(raw_data_path, seed=seed)


if __name__ == "__main__":
    main()
