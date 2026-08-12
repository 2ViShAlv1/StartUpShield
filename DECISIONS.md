# StartupShield AI Decisions

This file records implementation decisions, fallbacks, and deviations from the master build specification.

## Phase 1 — Dataset Fallbacks

- Kaggle churn download was not attempted because `kagglehub` is not installed in the current environment. Used the spec-approved synthetic churn fallback instead.
- HuggingFace sentiment dataset loading was not attempted because `datasets` is not installed in the current environment. Used the spec-approved template-based synthetic review fallback instead.
