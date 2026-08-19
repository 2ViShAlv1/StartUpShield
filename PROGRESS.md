# StartupShield AI Progress

## Master Checklist

- [x] Phase 0 — Repo scaffolded, environment ready
- [x] Phase 1 — All 3 datasets loading cleanly
- [x] Phase 2 — `preprocessing.py` + EDA report done
- [x] Phase 3 — Churn model, ROC-AUC > 0.80
- [x] Phase 4 — Sentiment model, macro-F1 > 0.75
- [x] Phase 5 — Anomaly detector validated against injected anomalies
- [x] Phase 6 — Forecast module beats naive baseline (or documented why not)
- [x] Phase 7 — `risk_aggregator.py` with SHAP explanations working
- [x] Phase 8 — `recommendation_engine.py` tested on 3-4 scenarios
- [x] Phase 9 — Full Streamlit dashboard live (7 pages, incl. upload flow)
- [ ] Phase 10 — End-to-end demo script passes for all 3 demo companies, README complete
- [ ] Phase 11 (optional) — FastAPI + Docker + CI

## Cross-Phase Hardening (post Phase-5 review)

- [x] Add `src/train_all.py` so every model is reproducible from a fresh clone
- [x] Fix row-order bug in `anomaly_module.build_features` with duplicate indices
- [x] Document churn proxy features in code, `DECISIONS.md`, and the churn report
- [x] Correct the anomaly report to match the model's actual feature set
- [x] Reconcile `config/config.yaml` churn model with the selected LightGBM model
- [x] Split `requirements.txt` into core + optional
- [x] Add `pyproject.toml` with explicit pytest `pythonpath`
- [x] Execute the EDA notebooks so committed copies carry outputs

## Phase 0 — Agent Bootstrap & Environment

- [x] Create repo folder structure
- [x] Initialize git repo
- [ ] Create Python 3.10+ virtual environment
- [x] Write `requirements.txt`
- [x] Write `config/config.yaml`
- [x] Write starter `README.md`
- [x] Create `PROGRESS.md`
- [x] Create `DECISIONS.md`

## Phase 1 — Project Setup & Data Foundation

- [x] Attempt to download Kaggle Telco Churn dataset into `data/raw/churn.csv`
- [x] Attempt to load a HuggingFace sentiment dataset and save to `data/raw/reviews.csv`
- [x] Write `src/generate_synthetic_data.py` for fallback churn, fallback reviews, and time-series data
- [x] Confirm all three load cleanly

## Phase 2 — EDA & Preprocessing

- [x] Explore churn, sentiment, and time-series datasets
- [x] Check missing values, class balance, and basic distributions
- [x] Build `src/preprocessing.py`
- [x] Add preprocessing helper tests
- [x] Write `reports/eda_findings.md`
- [x] Populate Phase 2 EDA notebooks

## Phase 3 — Churn Prediction Module

- [x] Engineer churn features
- [x] Train Logistic Regression baseline
- [x] Train Random Forest stronger model
- [x] Train and compare XGBoost model
- [x] Train and compare additional churn models including LightGBM
- [x] Handle class imbalance with balanced class weights
- [x] Evaluate ROC-AUC, PR-AUC, F1, and confusion matrix
- [x] Save best model to `models/churn_model.pkl`
- [x] Write `reports/model_evaluation_churn.md`
- [x] Add churn module tests

## Phase 4 — Sentiment Analysis Module

- [x] Reuse `preprocessing.clean_text` in the text pipeline
- [x] Train TF-IDF + Logistic Regression baseline
- [x] Evaluate accuracy, macro-F1, and per-class F1
- [x] Save chosen model to `models/sentiment_model/`
- [x] Write `reports/model_evaluation_sentiment.md`
- [x] Add sentiment module tests

## Phase 5 — Anomaly Detection Module

- [x] Build time-series rolling and daily-change features
- [x] Train IsolationForest anomaly detector
- [x] Validate against injected anomaly labels
- [x] Save model to `models/anomaly_model.pkl`
- [x] Write `reports/anomaly_validation.md`
- [x] Add anomaly module tests

## Phase 6 — Forecasting Module

- [x] Chronological train-test split (last 30 days held out, never shuffled)
- [x] Prophet baseline with native 95% confidence bounds
- [x] statsmodels Holt-Winters (ETS) fallback
- [x] Dependency-free seasonal-naive last-resort backend
- [x] Evaluate MAE, RMSE, MAPE against last-value-carried-forward naive baseline
- [x] Confidence bounds returned for every future date (FR4)
- [x] Save/load per-company models to `models/forecast_model_<company>.pkl`
- [x] Write `reports/model_comparison_forecast.md` with actual-vs-predicted plots
- [x] Add forecast module tests
- [x] Wire forecast into `src/train_all.py`
- [ ] LSTM (PyTorch) — stretch goal, deliberately skipped (see DECISIONS.md)

## Phase 7 — Risk Aggregation & Explainability

- [x] Weighted risk formula reading weights from `config.yaml`
- [x] Signal rescaling so all four weights have real influence (documented)
- [x] Downside-aware forecast term (documented deviation from the original spec)
- [x] SHAP `TreeExplainer` on the churn model, with a feature-importance fallback
- [x] Plain-language explanation string per company
- [x] `assess_company()` orchestrator, never NaN on partial input
- [x] Tests in `tests/test_risk_aggregator.py`

## Phase 8 — Recommendation Engine

- [x] Explicit rule table mapping signal combinations to actions
- [x] Anomaly recommendations name the metric that actually dropped
- [x] Priority ordering (critical / high / medium / info)
- [x] Tests in `tests/test_recommendation_engine.py`

## Phase 9 — Dashboard Integration

- [x] Streamlit multipage app (`app/app.py` + 5 pages)
- [x] Sidebar company selector persisted in session state
- [x] Overview with colour-coded risk badge, explanation, top 3 recommendations
- [x] Churn, Sentiment, Anomalies, Forecast, and Risk & Recommendations pages
- [x] `st.cache_resource` / `st.cache_data` so scoring stays well under the 5s budget
- [x] Graceful empty states on every page
- [x] Smoke tests covering all 7 pages x 3 companies (`tests/test_dashboard_pages.py`)
- [x] `run_full_pipeline_demo.py` end-to-end script writing per-company JSON

## Onboarding — Upload & Score Any Company

- [x] `src/data_validation.py` with plain-language, all-at-once error reporting
- [x] Missing-value detection (catches `"N/A"`/blank cells, not just unparseable text)
- [x] `pipeline.score_uploaded_company()` — pretrained churn/sentiment, fresh anomaly/forecast fits
- [x] Graceful degradation: reviews and time-series optional, short history skips forecasting
- [x] `app/pages/0_Upload_Your_Company.py` with CSV templates and JSON report export
- [x] Tests in `tests/test_data_validation.py` and `tests/test_upload_scoring.py`
- [x] `src/smart_import.py` — auto-detect column meanings from real exports
- [x] Derive `tenure` from any signup/created date column
- [x] Aggregate per-ticket/per-session exports into per-customer counts and join them
- [x] Confirm-or-override mapping UI on the upload page with a live preview
- [x] Tests in `tests/test_smart_import.py`

## Phase 10 — Testing, Polish & Documentation

- [x] End-to-end demo script runs for all 3 demo companies
- [x] README covers setup, training, dashboard, demo, and the risk formula
- [ ] Demo video / screenshots for the hackathon submission
- [ ] Final polish pass

## Phase 11 — Post-MVP (optional)

- [ ] FastAPI service
- [ ] Dockerfile
- [ ] CI pipeline
