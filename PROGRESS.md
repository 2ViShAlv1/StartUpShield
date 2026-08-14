# StartupShield AI Progress

## Master Checklist

- [x] Phase 0 — Repo scaffolded, environment ready
- [x] Phase 1 — All 3 datasets loading cleanly
- [x] Phase 2 — `preprocessing.py` + EDA report done
- [x] Phase 3 — Churn model, ROC-AUC > 0.80
- [x] Phase 4 — Sentiment model, macro-F1 > 0.75
- [ ] Phase 5 — Anomaly detector validated against injected anomalies
- [ ] Phase 6 — Forecast module beats naive baseline (or documented why not)
- [ ] Phase 7 — `risk_aggregator.py` with SHAP explanations working
- [ ] Phase 8 — `recommendation_engine.py` tested on 3-4 scenarios
- [ ] Phase 9 — Full 6-page Streamlit dashboard live
- [ ] Phase 10 — End-to-end demo script passes for all 3 demo companies, README complete
- [ ] Phase 11 (optional) — FastAPI + Docker + CI

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
