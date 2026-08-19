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
- Split `requirements.txt` (core, actually imported) from `requirements-optional.txt` (later phases and skipped alternatives such as `torch`, `transformers`, `shap`, `fastapi`). The original single file listed 21 mostly-uninstalled packages.
- Added `pyproject.toml` with `pythonpath = ["."]` so `from src...` imports in tests resolve explicitly instead of depending on pytest's implicit rootdir insertion.

## Phase 6 — Forecasting Model Choice

- `prophet` 1.4.0 and `statsmodels` 0.14.6 both installed successfully in `.venv`, so no
  install-failure fallback was needed. Both were evaluated and both are shipped: Prophet is the
  configured default, ETS is a real tested fallback rather than dead code.
- Added a third dependency-free `seasonal_naive` backend so the module still satisfies the FR4
  confidence-bound contract even if neither library is available.
- **LSTM (PyTorch) skipped.** The spec marks it a stretch goal and "the first thing to drop".
  With 335 training points per company and the definition of done already met by a 0.18 s
  statistical fit, an LSTM would add `torch` as a heavyweight dependency for no demonstrable
  gain. Recorded here rather than silently omitted.
- **ETS slightly beat Prophet** on mean MAPE (4.48% vs 4.59%, winning 2 of 3 companies) with
  better interval coverage. Prophet is kept as the default anyway: the gap is within noise on
  three series, and Prophet's native uncertainty interval is better principled than the ETS
  residual approximation. This is documented rather than hidden because it affects which
  backend a future maintainer should reach for.
- **Prophet's 95% interval was under-covered on GreenLeaf SaaS** (0.80 actual coverage). Do not
  present those bounds as calibrated without this caveat.
- One model is fitted per company, not one global model, because the three series have
  different scales, growth rates, and trend directions.
- Prophet models are persisted via `prophet.serialize` JSON, not raw pickle, which does not
  transfer reliably across environments. `save_model`/`load_model` handle this transparently.

## Phase 6 — What the Forecast Numbers Actually Mean

- The `seasonal_naive` backend (repeat last week, zero modelling) reaches 5.19% mean MAPE
  versus the required naive baseline's 13.05%. Prophet and ETS only improve on that by a
  further ~0.6-0.7 pp.
- So beating the naive baseline by 5-13 percentage points is **mostly the value of capturing
  weekly seasonality**, not evidence that Prophet is doing something sophisticated. Quote it
  that way; the seasonal-naive column in the report exists precisely to keep this honest.

## Phase 7 — Risk Score: Two Deliberate Deviations From the Spec

Both were found by running the formula on real demo data and seeing it produce
answers that were clearly wrong for a risk product. Both are documented rather than
quietly patched.

**1. The forecast term is now downside-only, not symmetric surprise.**

The spec defined it as `abs(actual - forecast) / forecast`. Measured on the demo data,
RedFlag Analytics was down **33% year-on-year** and produced a deviation of **0.012** —
because the forecast model predicted the decline accurately. "We correctly predicted
your collapse" was scoring as "no risk". The symmetric form also punished companies for
*beating* their forecast. The term now measures two downside-only components: how far
actuals fell below forecast, and how far the forward projection sits below the current
revenue level.

**2. Signals are rescaled to comparable ranges before weighting.**

All four signals are nominally in `[0, 1]`, but their realistic ranges are not
comparable — churn probability genuinely spans ~0.1–0.9 across companies, while the
anomaly ratio tops out near 0.15 and projected decline near 0.10. Feeding raw values
into the weighted sum meant the anomaly and forecast terms together moved the score by
under 2 points out of 100, making 45% of the declared weight decorative. Each signal is
now divided by a saturation point chosen on its own merits (`ANOMALY_SATURATION = 0.10`
— three flagged days in a month is already serious; `FORECAST_SATURATION = 0.10` — a 10%
monthly decline is a ~70%/year trajectory).

The band thresholds (30 / 60) were **not** adjusted. RedFlag Analytics scores 57.4
("Medium"), just under the "High" band, and that is left as-is: its churn and sentiment
are terrible but its revenue is not currently collapsing fast, so "Medium-High" is the
honest verdict. Tuning thresholds to make a demo look dramatic is exactly the kind of
number-fudging this project has avoided elsewhere.

**Remaining known simplification:** the score is still a linear weighted combination
with hand-chosen weights, not a learned meta-model. No labelled "company failed" dataset
exists to train one on. This is surfaced in the dashboard UI, not just in the code.

## Phase 9 — Demo Company Portfolios

`churn.csv` and `reviews.csv` are flat tables with no company column, so `src/demo_data.py`
builds the three demo portfolios by slicing them.

- Customers are assigned using **raw behavioural features only** (usage frequency, support
  tickets) — never the churn model's own output. The model still has to discover the risk
  rather than being handed a pre-sorted answer. Resulting actual churn rates: GreenLeaf
  14%, MixedCo 29%, RedFlag 87%, which lines up with the spec's target profiles.
- Reviews are assigned by sentiment label proportion.
- These are curated demo portfolios, not random samples. This is disclosed in the
  dashboard sidebar and the README rather than left for a judge to discover.

## Phase 9 — Dependency Notes

- `shap` pulled `numpy` 2.x into the environment. All 30 pre-existing tests were re-run and
  pass, so the `numpy<2` pin in `requirements.txt` was removed rather than fighting it.
- Installing `shap` also surfaced a `numba`/`coverage` incompatibility (numba expected
  `coverage.types.Tracer`, the pinned coverage only had `TTracer`); fixed by upgrading
  `coverage`. `matplotlib` was additionally installed *into the venv* because the system
  copy was compiled against NumPy 1.x and failed to import under NumPy 2.
- Streamlit resolves imports relative to the entrypoint inconsistently across pages, so each
  page prepends the `app/` directory to `sys.path` before importing `app/shared.py`.

## Onboarding — Scoring a Company the Models Have Never Seen

The dashboard originally only worked for three hardcoded demo companies. The
**Upload Your Company** page (`app/pages/0_Upload_Your_Company.py`, backed by
`pipeline.score_uploaded_company`) makes the product actually usable by a real
company. Three decisions shaped it.

**1. Which models are reused and which are refit — and why.**

- *Churn and sentiment reuse the pretrained models (inference only).* These are
  customer-level and text-level classifiers: "low usage plus rising support tickets
  precedes cancellation" is a reasonable prior across companies, and a new signup
  simply does not have the labelled churn history needed to retrain. Unseen
  `plan_type` values work because the churn pipeline one-hot encodes with
  `handle_unknown="ignore"`; this is covered by a test rather than assumed.
- *Anomaly detection and forecasting are fit fresh on the uploaded series,
  synchronously.* There is no useful pretrained notion of "what does this specific
  company's normal Tuesday look like" — every company's baseline is its own. This is
  the same one-model-per-company shape the demo already used; the only difference is
  that the fit happens live instead of ahead of time in `train_all.py`. Both fits are
  fast enough (well under a second on 75 days) that no background job is warranted.

**2. Validation is a first-class module, not inline `try`/`except`.**

`src/data_validation.py` sits between "arbitrary CSV a founder made in Excel" and the
models. It never raises — it returns a list of plain-language problems so the UI can
show all of them at once instead of surfacing one cryptic traceback at a time.
`churn_label` is deliberately **not** required: an onboarding company wants a
prediction, not to re-supply the answer.

Writing the tests surfaced a real gap. The first version only checked for *unparseable*
strings in numeric columns, but `pd.read_csv` converts the common placeholders
(`"N/A"`, `"null"`, blank cells) straight to NaN on read — so a file where an entire
column was `"N/A"` passed validation silently and would have produced a
confident-looking score built on missing data. Missing values are now reported
separately from unparseable ones, since a user fixes those two problems differently.

**3. Partial data degrades, it does not fail.**

Reviews and daily metrics are optional; customer data alone still produces a score.
`forecast_readiness()` scales the backtest window to the available history (30 days of
holdout at 60+ days of data, 14 days at 30-59, no forecast below 30) and returns the
reason, which is shown on screen. `risk_aggregator` already treated a missing signal as
neutral rather than as maximum risk, so a company with no time series is not punished
for it — it just scores on the signals it has.

Nothing uploaded is written to disk or sent anywhere; scoring happens in-session and the
only output is a JSON report the user chooses to download.

## Onboarding — Smart Import (Column Mapping)

The remaining friction after the upload page was that **no real tool exports our
schema**. A Stripe export calls spend `Monthly Recurring Revenue`, a CRM calls it
`MRR`, nobody exports `tenure` (they export a created date), and Zendesk exports one
row per ticket rather than a count per customer. Requiring a founder to fix all that
in Excel first is the step that would actually stop them using the tool.

`src/smart_import.py` removes it, with three deliberate constraints:

**It suggests, it never silently decides.** Every detected mapping carries a
confidence level (`exact` / `likely` / `guess`) and is rendered as an overridable
dropdown. A field with no plausible match maps to `None` rather than to a bad guess —
a wrong silent mapping produces a confident, wrong score, which is worse than an empty
dropdown the user has to fill in.

**Alias matching is longest-first and single-claim.** `"Monthly Recurring Revenue"`
beats a bare `"Revenue"` column because longer aliases are tried first, and once a
source column is claimed by one canonical field it cannot be claimed by another.

**Derived tenure is measured against the file, not the clock.** `derive_tenure` counts
months back from the latest date *in the uploaded data*, not `now()`. Using the real
current date would make a historical export's tenures drift upward every day it was
re-run, silently changing the churn score for unchanged data.

Only `support_tickets` and `usage_frequency` are allowed to default to 0, and the UI
says so explicitly — those are plausibly untracked, whereas a missing `monthly_spend`
or `plan_type` means the upload is wrong and scoring stays blocked.

Verified end-to-end against a Stripe-style customer export plus a Zendesk-style
per-ticket export: 6 fields detected or derived with zero manual renaming, 183 tickets
aggregated and joined across 115 customers, and the result passes `data_validation`
unchanged.
