# StartupShield AI Decisions

This file records implementation decisions, fallbacks, and deviations from the master build specification.

## Phase 1 — Dataset Fallbacks

- Kaggle churn download was not attempted because `kagglehub` is not installed in the current environment. Used the spec-approved synthetic churn fallback instead.
- HuggingFace sentiment dataset loading was not attempted because `datasets` is not installed in the current environment. Used the spec-approved template-based synthetic review fallback instead.

## Phase 1 — Real SaaS Churn Replacement

- User provided the Kaggle SaaS Customer Churn Prediction dataset as `train.csv` and `test_.csv`.
- Replaced synthetic `data/raw/churn.csv` with an anonymized, Phase-compatible version of the real SaaS churn data. Dropped `Name` and `Email`, hashed `Customer_ID`, and created proxy columns for `monthly_spend`, `support_tickets`, and `plan_type` because the source dataset does not include those fields directly.
- **Proxy columns are not independent features.** `monthly_spend` (`25 + Daily_Usage_Mins * 0.85`) and `plan_type` (`pd.cut` of `Daily_Usage_Mins`) are both deterministic functions of the same source column, and `support_tickets` is a keyword flag over the last support-ticket text that matches churn-intent words including `"cancel"`.
- Decision: **keep the schema, label the proxies** rather than dropping the columns. The master build spec mandates the `customer_id, tenure, monthly_spend, usage_frequency, support_tickets, plan_type, churn_label` schema for downstream modules, so removing columns would break Phases 7–9. Instead the derivation is documented in `churn_module.PROXY_FEATURE_COLUMNS`, in a warning comment at the derivation site, and in `reports/model_evaluation_churn.md`. Phase 7 SHAP output must label these columns as usage proxies.

## Phase 4 — Sentiment Model Scope

- Shipped the TF-IDF + Logistic Regression baseline as the MVP sentiment model.
- Skipped DistilBERT fine-tuning for this phase because the TF-IDF baseline exceeded the required macro-F1 threshold and provides very fast CPU inference.
- The resulting macro-F1 of 1.0000 is an artifact of template-generated review text, not evidence of real-world generalization. This caveat must travel with the number wherever it is quoted, including the pitch deck.

## Phase 5 — Anomaly Detector Feature Scope

- `build_features()` derives features for revenue, active users, and signups, but the detector trains on **revenue-derived features only** (`MODEL_SERIES_COLUMNS`).
- Reason: the generator defines `active_users = revenue / 18 + noise` and `signups = active_users * 0.045 + noise`, so the two extra series are noisy restatements of revenue. Training on all three was measured and dropped injected-anomaly recall from 1.00 to 0.80.
- Revisit when the series become genuinely independent (real customer data).

## Cross-Phase — Reproducibility

- Added `src/train_all.py` as the single training entrypoint. Previously the pickles in `models/` were trained ad hoc and could not be regenerated from committed code, while `.gitignore` excludes both `data/raw/*.csv` and `models/*.pkl` — so a fresh clone had neither data nor models nor a way to rebuild them.
- `src/train_all.py` reproduces every metric published in `reports/` exactly.
- Split `requirements.txt` (core, actually imported) from `requirements-optional.txt` (later phases and skipped alternatives such as `torch`, `transformers`, `prophet`, `fastapi`). The original single file listed 21 mostly-uninstalled packages.
- Added `pyproject.toml` with `pythonpath = ["."]` so `from src...` imports in tests resolve explicitly instead of depending on pytest's implicit rootdir insertion.
