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

## Why We Did This (Rationale)

- **Script-based Generation (`generate_synthetic_data.py`)**: To automate the data prep process so anyone can run one command and get the exact same raw data (reproducibility).
- **Using Real Kaggle Data**: Real data is messy and has true patterns. Training our machine learning models on real churn data gives much better and more realistic results than using fake (synthetic) data.
- **Combining Train and Test**: Having a single unified `churn.csv` allows us to have full control over how we split the data later for our own training, validation, and testing phases.
- **Removing PII (`Name`, `Email`)**: This is crucial for privacy and security. A machine learning model doesn't need to know a person's name to predict if they will churn. Storing PII is a security risk.
- **Hashing `Customer_ID`**: By converting the original IDs into hashes, we anonymize the data. We can still track unique users without exposing their actual database IDs.
- **Phase-compatible Columns**: Renaming and standardizing columns ensures that the data perfectly matches the schema our future ML pipeline expects.
- **Synthetic Data for Reviews & Time-Series**: We needed data to test Sentiment Analysis (NLP) and Forecasting modules. Since we didn't have real data for these yet, generating synthetic data allows our pipeline development to continue without being blocked.

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

