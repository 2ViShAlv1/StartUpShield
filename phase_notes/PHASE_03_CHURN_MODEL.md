# Phase 3 — Churn Prediction Module

## What We Have Done

- Implemented `src/churn_module.py`.
- Added churn feature engineering:
  - `spend_per_login`
  - `tickets_per_tenure`
  - `usage_per_tenure`
  - `low_usage_flag`
  - `high_ticket_flag`
  - `early_customer_flag`
- Built sklearn preprocessing with:
  - numeric scaling
  - one-hot encoding for `plan_type`
  - unknown-category handling
- Trained two candidate models:
  - Logistic Regression baseline
  - Random Forest stronger model
- Handled class imbalance with `class_weight="balanced"`.
- Evaluated ROC-AUC, PR-AUC, F1, and confusion matrix.
- Selected Random Forest as the best model.
- Saved the best model to `models/churn_model.pkl`.
- Confirmed saved and loaded model predictions match.
- Added tests in `tests/test_churn_module.py`.
- Wrote `reports/model_evaluation_churn.md`.

## Current Phase 3 Status

- Phase 3 deliverables are complete.
- Best model test ROC-AUC: 0.8494.
- Required ROC-AUC threshold: > 0.80.
- Status: passed.
- `pytest` is listed in `requirements.txt`, but it is not installed in the current environment, so formal pytest execution could not be run here.

## Model Results

| Model | ROC-AUC | PR-AUC | F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.8354 | 0.7632 | 0.7423 |
| Random Forest | 0.8494 | 0.7640 | 0.7784 |

## Dependency Notes

- `xgboost` is listed in `requirements.txt`, but it is not installed in the current environment.
- Because of that, XGBoost was not trained during this Phase 3 run.
- `imbalanced-learn` is also not installed, so SMOTE was not used.
- Random Forest with balanced class weights meets the Phase 3 target and is saved as the current best model.

## What You Should Study

- Binary classification.
- Logistic Regression.
- Random Forest.
- XGBoost basics.
- Class imbalance.
- ROC-AUC and PR-AUC.
- Confusion matrix.
- Model saving/loading with pickle or joblib.

## Useful Files

- `src/churn_module.py`
- `tests/test_churn_module.py`
- `models/churn_model.pkl`
- `reports/model_evaluation_churn.md`
