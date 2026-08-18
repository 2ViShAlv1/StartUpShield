# StartupShield AI

StartupShield AI is an MVP risk-monitoring dashboard for startups and SaaS companies.

It combines churn prediction, sentiment analysis, anomaly detection, forecasting, SHAP explanations, and rule-based recommendations into a single 0-100 company risk score.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements-optional.txt` holds dependencies for later phases (forecasting, SHAP, API layer)
and skipped alternatives. It is not needed to run the current pipeline or the tests.

## Reproducing the models

`data/raw/*.csv` and `models/*.pkl` are gitignored, so a fresh clone builds them:

```bash
python -m src.generate_synthetic_data     # writes data/raw/
python -m src.train_all                   # trains + saves every model
```

`src/train_all.py` regenerates the exact metrics published in `reports/`. Useful flags:

```bash
python -m src.train_all --module churn              # one module only
python -m src.train_all --seed 7                    # override config seed
python -m src.train_all --summary-json out.json     # write metrics as JSON
```

Hyperparameters and paths come from `config/config.yaml`.

## Tests

```bash
pytest
```

## Reading the reports honestly

Two published numbers need context — both are documented in `DECISIONS.md`:

- **Churn:** three of five base features (`monthly_spend`, `plan_type`, `support_tickets`) are
  proxies derived from the source dataset; two are the same variable. See the proxy-feature
  warning in `reports/model_evaluation_churn.md` before quoting feature importances.
- **Sentiment macro-F1 = 1.0000:** an artifact of template-generated review text, not proof of
  real-world generalization.

## Status

Project progress is tracked in `PROGRESS.md`. Implementation decisions and fallback notes are tracked in `DECISIONS.md`.
