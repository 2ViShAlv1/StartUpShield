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
- Trained three candidate models:
  - Logistic Regression baseline
  - Random Forest stronger model
- Installed `xgboost` in the project `.venv`.
- Trained and evaluated XGBoost as an additional candidate model.
- Ran a small cross-validated XGBoost tuning pass.
- Installed `lightgbm` in the project `.venv`.
- Tested additional churn models:
  - Extra Trees
  - HistGradientBoosting
  - AdaBoost
  - SVC
  - MLP
  - KNN
  - tuned Random Forest
  - LightGBM
- Handled class imbalance with `class_weight="balanced"`.
- Used `scale_pos_weight` for XGBoost class imbalance handling.
- Evaluated ROC-AUC, PR-AUC, F1, and confusion matrix.
- Selected LightGBM as the best model.
- Saved the best model to `models/churn_model.pkl`.
- Saved comparison model artifacts for Random Forest, XGBoost, and LightGBM.
- Confirmed saved and loaded model predictions match.
- Added tests in `tests/test_churn_module.py`.
- Wrote `reports/model_evaluation_churn.md`.

## Current Phase 3 Status

- Phase 3 deliverables are complete.
- Previous best test ROC-AUC: 0.8494.
- Improved best test ROC-AUC: 0.8535.
- Required ROC-AUC threshold: > 0.80.
- Status: passed.
- `pytest` is installed in the project `.venv` and can be run with `.venv/bin/python -m pytest`.

## Model Results

| Model | ROC-AUC | PR-AUC | F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.8354 | 0.7632 | 0.7423 |
| Random Forest | 0.8494 | 0.7640 | 0.7784 |
| XGBoost | 0.8490 | 0.7701 | 0.7790 |
| Tuned XGBoost | 0.8480 | 0.7802 | 0.7806 |
| Tuned Random Forest | 0.8517 | 0.7712 | 0.7806 |
| Extra Trees | 0.8514 | 0.7558 | 0.7753 |
| HistGradientBoosting | 0.8513 | 0.7777 | 0.7819 |
| LightGBM | 0.8535 | 0.7879 | 0.7806 |

## Dependency Notes

- `xgboost` is installed in `.venv` for this project.
- `lightgbm` is installed in `.venv` for this project.
- `imbalanced-learn` is also not installed, so SMOTE was not used.
- LightGBM has the highest ROC-AUC and is saved as the current best model.
- Random Forest and XGBoost are saved separately for comparison/future use.

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
- `models/churn_model_random_forest.pkl`
- `models/churn_model_xgboost.pkl`
- `models/churn_model_lightgbm.pkl`
- `reports/model_evaluation_churn.md`
