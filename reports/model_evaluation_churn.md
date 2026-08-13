# Churn Model Evaluation

## Dataset

- Source file: `data/raw/churn.csv`
- Rows: 2,500
- Target: `churn_label`
- Class balance:
  - retained (`0`): 1,592
  - churned (`1`): 908
- Split: stratified 80/20 train-test split with seed `42`
  - train rows: 2,000
  - test rows: 500

## Feature Engineering

Base features:

- `tenure`
- `monthly_spend`
- `usage_frequency`
- `support_tickets`
- `plan_type`

Engineered features:

- `spend_per_login`
- `tickets_per_tenure`
- `usage_per_tenure`
- `low_usage_flag`
- `high_ticket_flag`
- `early_customer_flag`

Preprocessing:

- Numeric features are standardized.
- `plan_type` is one-hot encoded with unknown-category handling.
- Class imbalance is handled with `class_weight="balanced"` for both trained classifiers.

## Candidate Models

| Model | ROC-AUC | PR-AUC | F1 | Confusion Matrix |
| --- | ---: | ---: | ---: | --- |
| Logistic Regression | 0.8354 | 0.7632 | 0.7423 | `[[256, 62], [38, 144]]` |
| Random Forest | 0.8494 | 0.7640 | 0.7784 | `[[285, 33], [45, 137]]` |

## Selected Model

- Best model: Random Forest
- Saved artifact: `models/churn_model.pkl`
- Selection reason: highest test ROC-AUC and stronger F1 than Logistic Regression.
- Save/load check: predictions matched after reloading the saved model.

## Acceptance Criteria

- Required ROC-AUC: > 0.80
- Achieved ROC-AUC: 0.8494
- Status: passed

## Notes

- `xgboost` is listed in `requirements.txt`, but it is not installed in the current environment, so XGBoost training was not run here.
- `imbalanced-learn` is also not installed, so SMOTE was not used. The implemented class imbalance approach is `class_weight="balanced"`.
- The current model is strong enough for Phase 3 and can be upgraded to XGBoost later once dependencies are installed.
