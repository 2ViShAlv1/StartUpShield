# StartupShield AI

StartupShield AI is an MVP risk-monitoring dashboard for startups and SaaS companies.

It combines churn prediction, sentiment analysis, anomaly detection, forecasting, SHAP explanations, and rule-based recommendations into a single 0-100 company risk score.

**Built so far:** churn, sentiment, anomaly detection, forecasting, risk aggregation with SHAP
explanations, a rule-based recommendation engine, and a 7-page Streamlit dashboard — including
an upload flow that scores a company the models have never seen.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements-optional.txt` holds dependencies for the optional API/deployment phase and for
alternatives that were evaluated and skipped. It is not needed to run the dashboard or tests.

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

## Running the dashboard

```bash
streamlit run app/app.py
```

Seven pages, with a company selector in the sidebar that persists across them:

| Page | What it shows |
| --- | --- |
| **Overview** | Headline 0-100 risk score, why it scored that way, top 3 recommended actions |
| **Upload Your Company** | Score *your own* data — CSV upload, validation, live result, JSON export |
| **Churn** | Risk distribution, SHAP drivers, highest-risk customer table |
| **Sentiment** | Sentiment mix, positivity trend, flagged negative reviews |
| **Anomalies** | Daily metrics with flagged days marked in red |
| **Forecast** | 30-day backtest plus a 30-day forward projection with confidence band |
| **Risk & Recommendations** | Full score breakdown by signal, SHAP chart, every recommendation |

Models are loaded once and cached; per-company scoring takes well under a second.

## End-to-end demo

```bash
python run_full_pipeline_demo.py
```

Scores all three demo companies, prints a full report for each, and writes summary JSON
to `reports/demo/`.

## Scoring your own company

The three built-in companies are demos. To score a real one, open the
**Upload Your Company** page and upload CSVs (templates are downloadable in the page):

| File | Required? | Columns |
| --- | --- | --- |
| Customers | **yes** | `tenure, monthly_spend, usage_frequency, support_tickets, plan_type` |
| Reviews | optional | `review_text` |
| Daily metrics | optional | `date, revenue, active_users, signups` |

You do **not** supply `churn_label` — that is what the model predicts.

### You don't have to rename anything

Real exports never use these column names, and none of them export `tenure` at all.
`src/smart_import.py` handles that, so you upload the export as-is:

- **Column names are auto-detected.** A Stripe export's `Monthly Recurring Revenue`,
  `Plan Name`, and `Customer ID` map themselves; each guess is shown with a confidence
  level and can be overridden from a dropdown. A field with no plausible match is left
  blank rather than guessed wrong.
- **`tenure` is derived from any signup/created date column** — months are counted from
  the latest date *in the file*, not the wall clock, so re-running a historical export
  gives the same answer every time.
- **Per-ticket and per-session exports can be aggregated.** A Zendesk file with one row
  per ticket collapses into a per-customer count and joins onto the customer table by
  email; customers with no tickets get 0, not dropped.
- **Fields you genuinely don't track** (`support_tickets`, `usage_frequency`) default to
  0, with a visible note that the score won't reflect those signals.

What happens to your data, and why it is split this way:

- **Churn and sentiment use the pretrained models — inference only, no retraining.**
  These are customer- and text-level classifiers, so a pattern learned on one company's
  history is a reasonable prior for another's, and retraining per company would need far
  more data than a new signup has. Unseen `plan_type` values are handled (the pipeline
  one-hot encodes with `handle_unknown="ignore"`).
- **Anomaly detection and forecasting are fit fresh on your time series, on the spot.**
  There is no useful pretrained notion of "what does *this* company's normal day look
  like" — every company's baseline is its own.
- **Nothing is written to disk or sent anywhere.** Scoring happens in-session; the only
  output is the JSON report you choose to download.

Missing data degrades gracefully rather than failing: with fewer than 30 days of history
forecasting is skipped and the forecast signal is excluded from the score, with the reason
shown on screen. Uploads are validated first (`src/data_validation.py`) and every problem
is reported at once.

## How the risk score works

```
risk_score = 100 × clip(
    w_churn     × churn_probability
  + w_sentiment × (1 − review_positivity)
  + w_anomaly   × anomalous_day_ratio
  + w_forecast  × projected_downside
, 0, 1)
```

Weights live in `config/config.yaml` (default `0.35 / 0.20 / 0.20 / 0.25`). Two things
are worth knowing before you quote this number — both are documented in `DECISIONS.md`:

- **It is a weighted formula, not a learned meta-model.** There is no labelled
  "company failed" dataset to train one on, so the weights encode a product judgement.
- **Signals are rescaled to comparable ranges before weighting.** Churn probability
  genuinely spans 0.1–0.9 while the anomaly ratio tops out near 0.15, so without
  rescaling, churn would dominate the score regardless of the configured weights.

## Tests

```bash
pytest
```

## Deploying

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for local testing steps, a manual test checklist,
and Streamlit Community Cloud deployment.

Models are gitignored and trained on first load in the deployment environment (~4 s,
cached per container) rather than committed as pickles, which are fragile across library
versions. The raw CSVs in `data/raw/` are committed as the reproducible seed.

## Reading the reports honestly

These published numbers need context — all are documented in `DECISIONS.md`:

- **Churn:** three of five base features (`monthly_spend`, `plan_type`, `support_tickets`) are
  proxies derived from the source dataset; two are the same variable. See the proxy-feature
  warning in `reports/model_evaluation_churn.md` before quoting feature importances.
- **Sentiment macro-F1 = 1.0000:** an artifact of template-generated review text, not proof of
  real-world generalization.
- **Forecast beats the naive baseline by 5-13 pp:** most of that gain is simply capturing weekly
  seasonality — a zero-modelling "repeat last week" backend already gets most of the way there.
  See the seasonal-naive row in `reports/model_comparison_forecast.md`.
- **Demo company portfolios are curated, and two of three datasets are synthetic.** The three
  demo companies are built by slicing the flat customer and review tables using *behavioural*
  features (usage, support tickets) — never using model output — so the models still have to
  discover the risk. This is implemented in `src/demo_data.py` and disclosed in the dashboard
  sidebar.

## Status

Project progress is tracked in `PROGRESS.md`. Implementation decisions and fallback notes are tracked in `DECISIONS.md`.
