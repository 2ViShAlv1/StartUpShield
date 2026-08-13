# Phase 2 — EDA & Preprocessing

## What We Have Done

- Explored churn, sentiment, and time-series datasets.
- Confirmed all Phase 2 input datasets load successfully with pandas.
- Checked missing values, class balance, basic distributions, and key feature relationships.
- Built `src/preprocessing.py` with helpers for:
  - missing value handling
  - categorical encoding
  - numeric scaling
  - text cleaning
  - stratified train-test splitting
- Added focused tests in `tests/test_preprocessing.py`.
- Wrote `reports/eda_findings.md` with concrete dataset numbers and modeling implications.
- Populated the Phase 2 EDA notebooks:
  - `notebooks/01_eda_churn.ipynb`
  - `notebooks/02_eda_sentiment.ipynb`
  - `notebooks/03_eda_timeseries.ipynb`

## Current Phase 2 Status

- Phase 2 deliverables are complete.
- Manual preprocessing checks pass with `python3`.
- `pytest` is listed in `requirements.txt`, but it is not installed in the current environment, so the formal test command could not be executed here.

## Key Findings

- Churn data has 2,500 rows, no missing values, and a churn rate of 36.32%.
- Churn risk is strongest for lower `monthly_spend`, lower `usage_frequency`, higher `support_tickets`, and `basic` plan customers.
- Review data is balanced across negative, neutral, and positive sentiment classes.
- Time-series files contain three useful demo profiles:
  - GreenLeaf SaaS: healthy growth, no injected anomalies
  - RedFlag Analytics: declining revenue, four injected anomalies
  - MixedCo: mostly stable, one injected anomaly

## What You Should Study

- Exploratory Data Analysis.
- Missing values and imputation.
- Categorical encoding.
- Numeric scaling.
- Train-test split.
- Stratified splitting.
- Text cleaning basics.
- Simple plots using pandas, matplotlib, or plotly.

## Useful Files

- `notebooks/01_eda_churn.ipynb`
- `notebooks/02_eda_sentiment.ipynb`
- `notebooks/03_eda_timeseries.ipynb`
- `src/preprocessing.py`
- `tests/test_preprocessing.py`
- `reports/eda_findings.md`
