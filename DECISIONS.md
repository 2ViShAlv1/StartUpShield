# StartupShield AI Decisions

This file records implementation decisions, fallbacks, and deviations from the master build specification.

## Phase 1 — Dataset Fallbacks

- Kaggle churn download was not attempted because `kagglehub` is not installed in the current environment. Used the spec-approved synthetic churn fallback instead.
- HuggingFace sentiment dataset loading was not attempted because `datasets` is not installed in the current environment. Used the spec-approved template-based synthetic review fallback instead.

## Phase 1 — Real SaaS Churn Replacement

- User provided the Kaggle SaaS Customer Churn Prediction dataset as `train.csv` and `test_.csv`.
- Replaced synthetic `data/raw/churn.csv` with an anonymized, Phase-compatible version of the real SaaS churn data. Dropped `Name` and `Email`, hashed `Customer_ID`, and created proxy columns for `monthly_spend`, `support_tickets`, and `plan_type` because the source dataset does not include those fields directly.
