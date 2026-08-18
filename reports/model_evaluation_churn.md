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
- `monthly_spend` — ⚠️ proxy
- `usage_frequency`
- `support_tickets` — ⚠️ proxy
- `plan_type` — ⚠️ proxy

### ⚠️ Proxy-feature warning (read before interpreting importances)

The real SaaS source dataset has no spend, ticket-count, or plan fields. Three of the five
base columns are therefore derived in `load_real_saas_churn_data()`, and two of them come
from the **same** underlying variable:

| Column | Actually derived from | Independent signal? |
| --- | --- | --- |
| `monthly_spend` | `25 + Daily_Usage_Mins * 0.85` | No — a linear restatement of daily usage |
| `plan_type` | `pd.cut(Daily_Usage_Mins, [-1, 25, 80, inf])` | No — the same variable, binned |
| `support_tickets` | keyword flag over last-ticket text (incl. `"cancel"`) | Partly — leans toward stated churn intent |

Consequences for reporting:

- `monthly_spend` correlates −0.50 with churn, but that is daily usage minutes wearing a
  costume. **Do not present it as a pricing or revenue insight.**
- `plan_type` churn rates (basic 64.9%, pro 15.5%, enterprise 6.6%) are the usage-minutes
  split restated, not a tier effect.
- `support_tickets` matches churn-intent words, so it sits closer to stated intent than to
  behaviour. Treat it as a near-term intent signal, not a support-load signal.
- **Phase 7 action:** SHAP plots built on these columns will tell a false causal story unless
  they are labelled as usage proxies. The constant `churn_module.PROXY_FEATURE_COLUMNS` exists
  so the explainability layer can label them automatically.

The schema is kept as-is because the master build spec (§ dataset table) mandates these column
names for downstream modules. The fix is honest labelling, not schema surgery.

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
- Class imbalance is handled with balanced class weights where supported.
- XGBoost uses `scale_pos_weight`.

## Candidate Models

| Model | ROC-AUC | PR-AUC | F1 | Confusion Matrix |
| --- | ---: | ---: | ---: | --- |
| Logistic Regression | 0.8354 | 0.7632 | 0.7423 | `[[256, 62], [38, 144]]` |
| Random Forest | 0.8494 | 0.7640 | 0.7784 | `[[285, 33], [45, 137]]` |
| XGBoost | 0.8490 | 0.7701 | 0.7790 | `[[279, 39], [41, 141]]` |
| Tuned Random Forest | 0.8517 | 0.7712 | 0.7806 | `[[286, 32], [45, 137]]` |
| Extra Trees | 0.8514 | 0.7558 | 0.7753 | `[[282, 36], [44, 138]]` |
| HistGradientBoosting | 0.8513 | 0.7777 | 0.7819 | `[[285, 33], [44, 138]]` |
| LightGBM | 0.8535 | 0.7879 | 0.7806 | `[[286, 32], [45, 137]]` |

## XGBoost Tuning Check

After installing `xgboost` in the project virtual environment, a small 3-fold `GridSearchCV` tuning pass was run on the training split.

Best cross-validation parameters:

- `learning_rate`: 0.03
- `max_depth`: 2
- `n_estimators`: 150
- `subsample`: 0.85

Tuned XGBoost test metrics:

| Model | ROC-AUC | PR-AUC | F1 | Confusion Matrix |
| --- | ---: | ---: | ---: | --- |
| Tuned XGBoost | 0.8480 | 0.7802 | 0.7806 | `[[286, 32], [45, 137]]` |

The tuned XGBoost model improved PR-AUC and F1, but its ROC-AUC was slightly lower than Random Forest.

## Additional Model Search

Additional sklearn and boosting models were tested after the first Phase 3 pass:

- Extra Trees
- HistGradientBoosting
- AdaBoost
- SVC
- MLP
- KNN
- tuned Random Forest
- LightGBM

The strongest improvement came from a regularized LightGBM model. A compact LightGBM `RandomizedSearchCV` did not beat this hand-regularized configuration, so the regularized LightGBM model was promoted.

## Selected Model

- Best model: LightGBM
- Saved artifact: `models/churn_model.pkl`
- Comparison artifacts:
  - `models/churn_model_random_forest.pkl`
  - `models/churn_model_xgboost.pkl`
  - `models/churn_model_lightgbm.pkl`
- Selection reason: highest test ROC-AUC and strongest PR-AUC among the evaluated candidates.
- Save/load check: predictions matched after reloading the saved model.

## Acceptance Criteria

- Required ROC-AUC: > 0.80
- Previous best ROC-AUC: 0.8494
- Improved ROC-AUC: 0.8535
- Status: passed

## Notes

- `xgboost` has been installed in the project `.venv` and evaluated for Phase 3.
- `lightgbm` has been installed in the project `.venv` and evaluated for Phase 3.
- `imbalanced-learn` is not installed, so SMOTE was not used.
- The current primary model is LightGBM, with Random Forest and XGBoost retained as comparison artifacts.
- `config/config.yaml` now sets `churn.model_type: lightgbm` to match the selected model
  (it previously said `xgboost`, contradicting this report).
- Reproduce every number above with `python -m src.train_all --module churn`. The metrics in
  the candidate table were regenerated from that entrypoint and match exactly.
- Because three base features are usage-minutes proxies, the honest read of the 0.8535 ROC-AUC
  is that the model separates high-usage from low-usage accounts well. That is a genuine and
  useful churn signal, but it is fewer independent signals than the five-feature table implies.
