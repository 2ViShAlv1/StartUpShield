# Phase 1 — Data Foundation

## What We Have Done

- Implemented `src/generate_synthetic_data.py`.
- Generated Phase 1 raw data files under `data/raw/`.
- Initially used synthetic fallback churn data because `kagglehub` was not installed.
- Later replaced churn with your Kaggle SaaS Customer Churn dataset.
- Combined `train.csv` and `test_.csv` into `data/raw/churn.csv`.
- Removed PII-like fields from the usable churn file:
  - `Name`
  - `Email`
- Hashed `Customer_ID` into `customer_id`.
- Created Phase-compatible churn columns:
  - `customer_id`
  - `tenure`
  - `monthly_spend`
  - `usage_frequency`
  - `support_tickets`
  - `plan_type`
  - `churn_label`
- Generated synthetic sentiment review data in `data/raw/reviews.csv`.
- Generated synthetic time-series data for:
  - GreenLeaf SaaS
  - RedFlag Analytics
  - MixedCo
- Verified all CSV files load successfully with pandas.
- Updated `DECISIONS.md` with the real churn data and fallback decisions.

## Current Data Status

- Real-source churn data: yes, based on the Kaggle SaaS churn dataset.
- Real sentiment data: no, currently synthetic template-based reviews.
- Real time-series data: no, currently synthetic SaaS metric data.

## What You Should Study

- `pandas.read_csv()`, `.head()`, `.shape`, `.info()`.
- Rows, columns, features, and labels.
- Churn prediction basics.
- What PII is and why `Name`/`Email` should not be stored in model-ready data.
- Hashing identifiers using SHA-256.
- Synthetic data generation with NumPy.
- Random seeds and reproducibility.
- Time-series basics:
  - trend
  - seasonality
  - noise
  - anomaly
- Why fallback data is useful in ML projects.

## Useful Files

- `src/generate_synthetic_data.py`
- `data/raw/churn.csv`
- `data/raw/reviews.csv`
- `data/raw/timeseries_greenleaf_saas.csv`
- `data/raw/timeseries_redflag_analytics.csv`
- `data/raw/timeseries_mixedco.csv`
- `DECISIONS.md`
- `PROGRESS.md`

