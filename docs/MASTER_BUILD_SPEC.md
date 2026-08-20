# StartupShield AI — Master Build Specification
*(AI Agent Execution Document — v1.0)*

> **Ye document kya hai:** Ye ek self-contained, execution-ready spec hai StartupShield AI (startup risk-monitoring dashboard) banane ke liye. Ye file directly kisi coding AI agent (Claude Code, Cursor, Devin, etc.) ko de sakte ho aur wo isse follow karke poora project — data se leke dashboard tak — bana sakta hai. Har phase me: goal, granular tasks, exact files/interfaces, datasets, aur testable Definition-of-Done diya gaya hai.

---

## 0. Agent Operating Instructions — Pehle Ye Padho

Agar tum (AI agent) ye document follow kar rahe ho, toh in rules ko strictly follow karo:

1. **Sequential execution.** Phase 0 → Phase 10 order me karo. Phase 11 optional hai (post-MVP), sirf tab karo jab sab pehle complete ho chuka ho aur time bache.
2. **Definition of Done (DoD) mandatory hai.** Kisi bhi phase ko "complete" mat maano jab tak uska DoD section pura satisfy na ho. DoD me diye gaye checks khud run karke verify karo (tests, metrics, script execution).
3. **Real data na mile toh synthetic fallback use karo — kabhi mat ruko.** Har dataset section me ek fallback diya gaya hai. Agar Kaggle/HuggingFace download fail ho (auth, network, rate-limit), fallback synthetic generator use karo aur `DECISIONS.md` me note likho ki fallback kyu use hua.
4. **Har phase ke end pe commit karo.** Git commit message format: `phase-N: <short summary>` (e.g. `phase-3: churn module trained, ROC-AUC 0.84`).
5. **Progress track karo.** Root me `PROGRESS.md` maintain karo — har phase ke checklist items ko `[x]` mark karte jao jaise complete hote hain.
6. **Decisions log karo.** Kahin bhi is spec se deviate karna pade (dataset switch, library switch, weight change) toh `DECISIONS.md` me 2-3 line me reason likho. Silently mat badlo.
7. **Code style:** modular, reproducible (fixed seeds), type-hinted function signatures, aur simple/plain-English docstrings + inline comments jo *WHY* explain karein, sirf *WHAT* nahi — beginner bhi padh ke samjhe. Koi cleverness/one-liner golfing nahi, readability priority hai.
8. **Har module standalone testable ho.** Koi bhi `src/*.py` module directly import karke, bina dashboard chalaye, test kiya ja sakta ho (`python -m pytest tests/`).
9. **PII/privacy:** agar kabhi real customer data use ho, `customer_id`/company names ko `hashlib.sha256` se hash karo before storing/logging (NFR6).
10. **Stuck ho toh:** pehle fallback try karo, phir spec ke "Section 11 — Fallback & Contingency Plan" dekho. Sirf genuinely ambiguous product decisions (jaise "kaunsa demo company naam use karu") ke liye user se pucho — technical blockers khud resolve karo.

---

## 1. Project Overview

**Naam:** StartupShield AI
**Kya hai:** Ek early-warning risk dashboard jo startups/SaaS companies ke liye ek single "Risk Score (0-100)" generate karta hai — customer churn prediction, review/ticket sentiment analysis, business-metric anomaly detection, aur revenue/DAU forecasting ko combine karke. Har risk score ke saath SHAP-based explanation aur rule-based recommended actions bhi milte hain.

**Kis liye ban raha hai:** Academic/portfolio project + competition submission (Unstop). MVP-level system hai — production SaaS nahi, lekin end-to-end demo-able, explainable, aur technically solid hona chahiye.

**Core value proposition:** "Ek dashboard jo bata de ki tumhara startup/company risk me hai ya nahi, kyun hai, aur ab kya karna chahiye."

**Non-goals (in scope se bahar):**
- Real production deployment / multi-tenant SaaS
- Real customer PII pipeline (sirf synthetic ya anonymized demo data)
- Perfect state-of-the-art accuracy — MVP-level metrics acceptable hain (see §2.5)
- Mobile app / native client — sirf Streamlit web dashboard

**End output jo demo me dikhna chahiye:** Ek company select karo dashboard pe → uska risk score, top contributing factors (SHAP), sentiment trend, anomaly timeline, revenue/DAU forecast chart, aur 2-3 recommended actions — sab ek page pe ya connected multi-page view me.

---

## 2. Requirements (Consolidated)

### 2.1 Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Customer-level tabular data se churn probability predict karna |
| FR2 | Review/ticket text se sentiment score/label output karna |
| FR3 | Business metrics (revenue/DAU) me anomaly flag karna |
| FR4 | Future revenue/DAU forecast karna with confidence range |
| FR5 | Sab module outputs ko combine karke composite risk score (0-100) generate karna |
| FR6 | Risk score ke top contributing factors SHAP se explain karna |
| FR7 | Risk drivers ke basis pe rule-based recommended actions dena |
| FR8 | Sab kuch ek dashboard pe visualize karna (per-company view) |
| FR9 | User CSV upload karke apna data analyze kar sake (MVP: local file, post-MVP: API) |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Dashboard load time < 5 sec for a single company's data |
| NFR2 | Model inference < 2 sec per prediction |
| NFR3 | Modular code — har model alag Python module me, reusable |
| NFR4 | Reproducibility — fixed random seeds everywhere (`SEED = 42`) |
| NFR5 | Explainability — koi bhi prediction black-box nahi, SHAP-traceable |
| NFR6 | PII hash/anonymize (real data use hone pe) |
| NFR7 | Synthetic data pe fully demo-able (bina real data ke bhi chal sake) |

### 2.3 Technical Stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **ML:** scikit-learn, xgboost
- **NLP:** HuggingFace `transformers` (DistilBERT), scikit-learn (TF-IDF), HuggingFace `datasets`
- **Anomaly detection:** scikit-learn `IsolationForest`
- **Forecasting:** `prophet` (primary), `statsmodels` ETS/ARIMA (lightweight fallback), PyTorch LSTM (stretch)
- **XAI:** `shap`
- **Dashboard:** `streamlit` (multipage)
- **Backend (post-MVP):** FastAPI
- **DB (post-MVP):** SQLite → PostgreSQL
- **Experiment tracking (post-MVP):** MLflow
- **Deployment (post-MVP):** Docker, GitHub Actions
- **Testing:** pytest
- **Config:** YAML (`config/config.yaml`)

### 2.4 Data Requirements

| Dataset | Schema | Primary Source | Fallback |
|---|---|---|---|
| Churn | `customer_id, tenure, monthly_spend, usage_frequency, support_tickets, plan_type, churn_label` | Kaggle "Telco Customer Churn" (blastchar/telco-customer-churn) | Synthetic generator (§6.3.1) |
| Text/Sentiment | `review_text/ticket_text` (+ optional `star_rating`) | HuggingFace `datasets` — `amazon_polarity` or `yelp_review_full`, sampled ~20k rows | Rule-based synthetic review templates (§6.4.1) |
| Time-series | `date, revenue, active_users, signups` | Fully synthetic (no real source exists publicly at SaaS-metric granularity) | N/A — synthetic is the primary plan |

### 2.5 Success Criteria (MVP-level)

- Churn model: **ROC-AUC > 0.80** on held-out test set
- Sentiment model: **macro-F1 > 0.75**
- Anomaly detection: correctly flags injected synthetic anomalies (qualitative + precision/recall on known injected points)
- Dashboard: end-to-end risk score + explanation + recommendation for ≥1 demo company, working live

---

## 3. Repo Structure (Authoritative)

```text
startupshield-ai/
├── config/
│   └── config.yaml                  # seeds, paths, risk weights, thresholds
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_churn.ipynb
│   ├── 02_eda_sentiment.ipynb
│   └── 03_eda_timeseries.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── churn_module.py
│   ├── sentiment_module.py
│   ├── anomaly_module.py
│   ├── forecast_module.py
│   ├── risk_aggregator.py
│   ├── recommendation_engine.py
│   └── generate_synthetic_data.py
├── models/
│   ├── churn_model.pkl
│   ├── sentiment_model/
│   └── anomaly_model.pkl
├── app/
│   ├── app.py                       # Streamlit Home/Overview
│   └── pages/
│       ├── 1_Churn.py
│       ├── 2_Sentiment.py
│       ├── 3_Anomalies.py
│       ├── 4_Forecast.py
│       └── 5_Risk_and_Recommendations.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_churn_module.py
│   ├── test_sentiment_module.py
│   ├── test_anomaly_module.py
│   ├── test_forecast_module.py
│   └── test_risk_aggregator.py
├── reports/
│   ├── eda_findings.md
│   ├── model_evaluation_churn.md
│   ├── model_evaluation_sentiment.md
│   └── model_comparison_forecast.md
├── run_full_pipeline_demo.py        # end-to-end integration script (Phase 10)
├── requirements.txt
├── .gitignore
├── README.md
├── PROGRESS.md
└── DECISIONS.md
```

---

## 4. Global Conventions (Apply to Every Phase)

- **Python:** 3.10+, PEP8, type hints on all public functions.
- **Seed:** single `SEED = 42` defined in `config/config.yaml`, imported everywhere — never hardcode a different seed anywhere else.
- **Docstrings:** Google-style, one-line summary + Args/Returns.
- **Comments:** plain-English, explain reasoning/why, not just restate code.
- **Config-driven:** paths, model hyperparameters defaults, and risk weights live in `config/config.yaml` — not hardcoded inside modules.
- **Every `src/*.py` module exposes at minimum:** `train()`, `predict()`/`infer()`, `evaluate()`, `save_model()`, `load_model()` (except `preprocessing.py`, `risk_aggregator.py`, `recommendation_engine.py` which have their own interfaces, given below).
- **Logging:** use Python's `logging` module (not print) for anything beyond a notebook cell — `logging.info` for progress, `logging.warning` for fallback triggers.
- **Testing:** every module gets a matching `tests/test_*.py` with at least: one happy-path test, one edge-case test (empty/missing data).

### Sample `config/config.yaml`

```yaml
seed: 42
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  models: "models"
churn:
  model_type: "xgboost"       # logistic_regression | random_forest | xgboost
  test_size: 0.2
sentiment:
  model_type: "tfidf_logreg"  # tfidf_logreg | distilbert
  max_features: 5000
anomaly:
  contamination: 0.03
forecast:
  model_type: "prophet"       # prophet | ets | lstm
  horizon_days: 30
risk_weights:
  w_churn: 0.35
  w_sentiment: 0.20
  w_anomaly: 0.20
  w_forecast: 0.25
```

---

## 5. Phase 0 — Agent Bootstrap & Environment (Day 0)

**Goal:** Environment ready, repo scaffolded, dependencies installed, nothing else.

**Tasks**
- [ ] Create repo folder structure exactly as in §3 (empty placeholder files where needed, e.g. `.gitkeep` in empty dirs)
- [ ] Initialize git repo, first commit `phase-0: repo scaffold`
- [ ] Create Python 3.10+ virtual environment
- [ ] Write `requirements.txt` (see Appendix A) and install
- [ ] Write `config/config.yaml` (see §4 sample)
- [ ] Write starter `README.md` with project one-liner, setup instructions, and a "Status" section pointing to `PROGRESS.md`
- [ ] Create `PROGRESS.md` with all phase checklists (copy from this document) and `DECISIONS.md` (empty, ready for entries)

**Setup commands**
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Definition of Done**
- `pip list` shows all packages from `requirements.txt` installed without conflict
- Repo tree matches §3 exactly
- `git log` shows the phase-0 commit

---

## 6. Phase 1 — Project Setup & Data Foundation (Week 1)

**Goal:** All 3 datasets ready and loading cleanly.

### 6.1 Tasks
- [ ] Attempt to download Kaggle Telco Churn dataset into `data/raw/churn.csv`
- [ ] Attempt to load a HuggingFace sentiment dataset (`amazon_polarity` or `yelp_review_full`), sample ~20,000 rows, save to `data/raw/reviews.csv`
- [ ] Write `src/generate_synthetic_data.py` for the time-series dataset
- [ ] Confirm all three load cleanly (no encoding errors, correct column count) in a quick notebook check

### 6.2 Churn dataset acquisition
- Primary: Kaggle `blastchar/telco-customer-churn` (via `kagglehub` or manual download + place in `data/raw/`)
- **Fallback (if Kaggle unavailable):** generate synthetic churn data —
  ```python
  # synthetic churn fallback logic
  # tenure ~ Uniform(1, 72) months
  # monthly_spend ~ Normal(70, 25), clipped to [10, 200]
  # usage_frequency ~ Poisson(lambda=12)
  # support_tickets ~ Poisson(lambda=2), higher for soon-to-churn customers
  # churn_label: logistic function of (low tenure, low usage, high tickets) + noise
  ```

### 6.3 Text/sentiment dataset acquisition
- Primary: `datasets.load_dataset("amazon_polarity")` or `("yelp_review_full")`, stratified sample ~20k rows, map to `{negative, neutral, positive}` (yelp 1-2★→negative, 3★→neutral, 4-5★→positive)
- **6.3.1 Fallback (if HF hub unreachable):** template-based synthetic reviews — generate short reviews from positive/neutral/negative phrase templates + product-domain nouns, enough variety for TF-IDF/DistilBERT to learn a real signal (min 2,000 rows, balanced classes)

### 6.4 Synthetic time-series generator — `generate_synthetic_data.py`

```python
def generate_company_timeseries(
    company_name: str,
    start_date: str,
    n_days: int = 365,
    base_revenue: float = 10000.0,
    growth_rate: float = 0.001,       # daily compounding growth
    weekly_seasonality_amp: float = 0.15,
    noise_std: float = 0.05,
    n_anomalies: int = 3,
    anomaly_magnitude: float = 0.4,   # fractional spike/drop
    seed: int = 42,
) -> pd.DataFrame:
    """Returns DataFrame with columns: date, revenue, active_users, signups,
    is_injected_anomaly (bool) — the last column is ground truth for
    anomaly-detection validation later in Phase 5, and must be dropped
    before feeding data to the model."""
```
- Signal = trend (compounding growth) + weekly seasonality (sine wave) + Gaussian noise + N injected anomalies (random day, random sign, magnitude spike/drop)
- `active_users` and `signups` correlated with `revenue` but with their own independent noise
- Generate for multiple synthetic companies (at least 3 — see Phase 10 demo companies) with different growth/health profiles

### Definition of Done
- `data/raw/churn.csv`, `data/raw/reviews.csv` exist and load with `pandas.read_csv` without error
- `generate_synthetic_data.py` runs standalone and produces `data/raw/timeseries_<company>.csv` for at least 3 companies
- A short notebook cell prints `.shape` and `.head()` of all 3 datasets successfully

**Deliverable:** All 3 datasets ready and loading cleanly in a notebook.

---

## 7. Phase 2 — EDA & Preprocessing (Week 2)

### 7.1 Tasks
- [ ] `notebooks/01_eda_churn.ipynb` — class balance, missing values, correlation of numeric features with `churn_label`, distribution plots
- [ ] `notebooks/02_eda_sentiment.ipynb` — class balance, review length distribution, top TF-IDF terms per class
- [ ] `notebooks/03_eda_timeseries.ipynb` — trend/seasonality visual check, verify injected anomalies are visually detectable
- [ ] Build `src/preprocessing.py`
- [ ] Write `reports/eda_findings.md` summarizing key findings per dataset (which features correlate with churn, class imbalance severity, etc.)

### 7.2 `src/preprocessing.py` interface

```python
def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame: ...
def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...
def scale_numeric(df: pd.DataFrame, columns: list[str], method: str = "standard") -> pd.DataFrame: ...
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML, remove extra whitespace. Keep punctuation
    that carries sentiment signal (e.g. '!', '?')."""
def train_test_split_stratified(X, y, test_size: float = 0.2, seed: int = 42): ...
```

### Definition of Done
- `preprocessing.py` importable, every function has ≥1 passing test in `tests/test_preprocessing.py`
- `reports/eda_findings.md` written with concrete numbers (not placeholders)

**Deliverable:** `preprocessing.py` module + EDA report.

---

## 8. Phase 3 — Churn Prediction Module (Week 3–4)

### 8.1 Tasks
- [ ] RFM-style feature engineering (Recency/Frequency/Monetary proxies from available columns)
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest and XGBoost, tune via cross-validation (`GridSearchCV` or `RandomizedSearchCV`)
- [ ] Handle class imbalance — `class_weight="balanced"` or SMOTE (`imbalanced-learn`)
- [ ] Evaluate: ROC-AUC, PR-AUC, confusion matrix
- [ ] Save best model to `models/churn_model.pkl`
- [ ] Write `reports/model_evaluation_churn.md`

### 8.2 `src/churn_module.py` interface

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame: ...
def train(X_train, y_train, model_type: str = "xgboost", seed: int = 42): ...
def evaluate(model, X_test, y_test) -> dict:
    """Returns {'roc_auc':.., 'pr_auc':.., 'confusion_matrix':.., 'f1':..}"""
def save_model(model, path: str = "models/churn_model.pkl") -> None: ...
def load_model(path: str = "models/churn_model.pkl"): ...
def predict(model, X) -> np.ndarray:
    """Returns churn probability array in [0, 1], one per row."""
```

### Definition of Done
- Test-set **ROC-AUC > 0.80** (§2.5) — if not met after tuning, document attempted approaches in `reports/model_evaluation_churn.md` and proceed with best achieved (don't block the pipeline indefinitely)
- `models/churn_model.pkl` saved and re-loadable, `predict()` reproducible (same input → same output)
- Unit tests cover `predict()` output range `[0,1]`

**Deliverable:** `churn_module.py` + saved model + evaluation report.

---

## 9. Phase 4 — Sentiment Analysis Module (Week 5–6)

### 9.1 Tasks
- [ ] Text cleaning pipeline (reuse `preprocessing.clean_text`)
- [ ] TF-IDF + Logistic Regression baseline, evaluate
- [ ] Fine-tune DistilBERT (`transformers.Trainer` or `pipeline`) on same data, compare — if GPU/compute unavailable or fine-tuning fails/too slow, document in `DECISIONS.md` and ship TF-IDF baseline as MVP model (this is explicitly allowed, per original scope)
- [ ] Save chosen model/pipeline to `models/sentiment_model/`
- [ ] Write `reports/model_evaluation_sentiment.md` (TF-IDF vs DistilBERT comparison table: F1, inference time, model size)

### 9.2 `src/sentiment_module.py` interface

```python
def train_tfidf_baseline(X_train, y_train, max_features: int = 5000): ...
def train_distilbert(train_texts, train_labels, val_texts, val_labels, epochs: int = 2): ...
def evaluate(model, X_test, y_test) -> dict:
    """Returns {'macro_f1':.., 'accuracy':.., 'per_class_f1':..}"""
def save_model(model, path: str = "models/sentiment_model") -> None: ...
def load_model(path: str = "models/sentiment_model"): ...
def predict(model, texts: list[str]) -> list[dict]:
    """Returns [{'label': 'positive'|'neutral'|'negative', 'score': float}, ...]
    score is a normalized [0,1] positivity score, used downstream by risk_aggregator."""
```

### Definition of Done
- **macro-F1 > 0.75** (§2.5) on whichever model ships
- `predict()` output format matches the schema above exactly (risk_aggregator depends on it)
- Inference for a batch of 100 reviews completes in reasonable time (< 5 sec on CPU for TF-IDF; note DistilBERT timing separately)

**Deliverable:** `sentiment_module.py` + comparison report.

---

## 10. Phase 5 — Anomaly Detection Module (Week 7)

### 10.1 Tasks
- [ ] Feature prep on time-series: rolling mean/std (7-day, 30-day windows), day-of-week, % change from previous day
- [ ] Train `IsolationForest`, tune `contamination` parameter
- [ ] Validate against the `is_injected_anomaly` ground-truth column from Phase 1's synthetic generator — compute precision/recall on known injected points
- [ ] Save model to `models/anomaly_model.pkl`
- [ ] Write validation notes (append to `reports/eda_findings.md` or a new `reports/anomaly_validation.md`)

### 10.2 `src/anomaly_module.py` interface

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: date, revenue, active_users, signups.
    Output: adds rolling_mean_7, rolling_std_7, pct_change_1d, day_of_week."""
def train(X_train, contamination: float = 0.03, seed: int = 42): ...
def evaluate_against_ground_truth(model, X, y_true_anomaly_flags) -> dict:
    """Returns {'precision':.., 'recall':.., 'f1':..} against injected anomalies."""
def save_model(model, path: str = "models/anomaly_model.pkl") -> None: ...
def load_model(path: str = "models/anomaly_model.pkl"): ...
def predict(model, X) -> np.ndarray:
    """Returns array of -1 (anomaly) / 1 (normal), plus an anomaly_score
    (model.decision_function output) for severity ranking."""
```

### Definition of Done
- Model recovers **majority of injected anomalies** (target recall > 0.6 — anomaly detection is inherently noisy, document actual number achieved)
- `predict()` runs on a full year of daily data in well under NFR2's 2-sec budget

**Deliverable:** `anomaly_module.py` + validation notes.

---

## 11. Phase 6 — Forecasting Module (Week 8–9)

### 11.1 Tasks
- [ ] Chronological train-test split (never random-shuffle time series)
- [ ] Prophet baseline — if `prophet` install fails (common pystan/cmdstanpy issues), fallback to `statsmodels` Holt-Winters (ETS) — document choice in `DECISIONS.md`
- [ ] LSTM model (PyTorch) — **stretch goal only**, do this last, skip without blocking if time-constrained (see §11 Fallback Plan — this is the first thing to drop)
- [ ] Evaluate: MAE, RMSE, MAPE against a naive baseline (last-value-carried-forward)
- [ ] Write `reports/model_comparison_forecast.md` with actual-vs-predicted plots

### 11.2 `src/forecast_module.py` interface

```python
def train(df: pd.DataFrame, model_type: str = "prophet", horizon_days: int = 30, seed: int = 42): ...
def evaluate(model, test_df) -> dict:
    """Returns {'mae':.., 'rmse':.., 'mape':.., 'vs_naive_baseline': {...}}"""
def save_model(model, path: str) -> None: ...
def load_model(path: str): ...
def predict(model, horizon_days: int = 30) -> pd.DataFrame:
    """Returns DataFrame: date, forecast, lower_bound, upper_bound —
    the confidence interval is required (FR4), not optional."""
```

### Definition of Done
- `predict()` output includes confidence bounds for every future date
- Forecast beats the naive baseline on MAPE (if it doesn't, document honestly — don't fudge numbers)

**Deliverable:** `forecast_module.py` + comparison plots.

---

## 12. Phase 7 — Risk Aggregation + Explainability (Week 10)

### 12.1 Tasks
- [ ] Implement weighted risk formula using `config.yaml` weights
- [ ] Apply SHAP `TreeExplainer` on the churn model (primary driver — it's the tree-based model with the richest feature set), generate summary + per-customer force-plot data
- [ ] Map top SHAP features + anomaly/sentiment flags → a plain-language explanation string
- [ ] Justify the weighting scheme in writing (this is a known simplification — document it explicitly, don't hide it)

### 12.2 Risk formula

```
risk_score = 100 * clip(
    w_churn    * churn_prob
  + w_sentiment* (1 - sentiment_positivity_score)
  + w_anomaly  * anomaly_flag_ratio_last_30d
  + w_forecast * normalized_forecast_deviation
, 0, 1)
```
- Default weights (from `config.yaml`): `w_churn=0.35, w_sentiment=0.20, w_anomaly=0.20, w_forecast=0.25`
- `anomaly_flag_ratio_last_30d` = fraction of last 30 days flagged anomalous by `anomaly_module`
- `normalized_forecast_deviation` = `abs(actual_recent - forecast_predicted) / forecast_predicted`, clipped to `[0,1]`
- **Document this formula's limitation explicitly** in the output report: it's a simple linear weighted combination, not a learned meta-model — acceptable for MVP, flagged as a known simplification.

### 12.3 `src/risk_aggregator.py` interface

```python
def compute_risk_score(
    churn_prob: float,
    sentiment_positivity: float,
    anomaly_flag_ratio: float,
    forecast_deviation: float,
    weights: dict,
) -> float:
    """Returns risk_score in [0, 100]."""

def get_shap_explanation(churn_model, X_row) -> dict:
    """Returns {'top_factors': [(feature_name, shap_value), ...],
    'base_value': float, 'prediction': float}"""

def generate_explanation_text(shap_result: dict, sentiment_label: str,
                               anomaly_flags: list, forecast_trend: str) -> str:
    """Returns a plain-language sentence, e.g.:
    'Risk driven primarily by low usage_frequency and 2 negative reviews
    this month; revenue is trending 12% below forecast.'"""

def assess_company(company_id: str, churn_data, sentiment_data,
                    anomaly_data, forecast_data, weights: dict) -> dict:
    """Top-level orchestrator. Returns:
    {'risk_score': float, 'top_factors': list, 'explanation_text': str}
    — this is the single function the dashboard calls per company."""
```

### Definition of Done
- `assess_company()` runs end-to-end for a synthetic company and returns a valid `{risk_score, top_factors, explanation_text}` dict
- `risk_score` always in `[0, 100]`, never NaN even with partial missing inputs (define sane defaults, e.g. treat missing anomaly data as `anomaly_flag_ratio=0`)

**Deliverable:** `risk_aggregator.py` producing `{risk_score, top_factors, explanation_text}` per company.

---

## 13. Phase 8 — Recommendation Engine (Week 10, parallel with Phase 7)

### 13.1 Tasks
- [ ] Define a rule table mapping risk-factor combinations → recommended actions
- [ ] Implement `src/recommendation_engine.py`
- [ ] Test on 3–4 example scenarios (reuse Phase 10's demo companies where possible)

### 13.2 Sample rule table

| Condition | Recommended Action |
|---|---|
| High churn_prob (>0.6) AND negative sentiment | "Prioritize retention outreach for at-risk segment; review recent negative feedback themes" |
| Anomaly on revenue (sudden drop) | "Investigate payment/billing pipeline for failures" |
| Anomaly on active_users (sudden drop) | "Check for product outage or recent breaking release" |
| Forecast trending down + high churn | "Revenue decline compounding with churn risk — consider immediate pricing/retention review" |
| Low churn_prob AND positive sentiment AND no anomalies | "Healthy — no immediate action, continue monitoring" |

### 13.3 `src/recommendation_engine.py` interface

```python
def recommend_actions(risk_factors: dict, rule_table: list[dict] | None = None) -> list[str]:
    """risk_factors example:
    {'churn_prob': 0.72, 'sentiment_label': 'negative',
     'anomaly_on': ['revenue'], 'forecast_trend': 'declining'}
    Returns ordered list of recommended action strings (highest priority first).
    Falls back to rule_table param if provided, else loads default table
    from a module-level constant — keep the table config-editable, not
    buried in conditional logic, so new rules are easy to add."""
```

### Definition of Done
- Tested against 3–4 scenarios covering: healthy, high-churn-only, anomaly-only, and compound-risk cases — each returns sensible, non-empty recommendations

**Deliverable:** Recommendation function tested on 3–4 example scenarios.

---

## 14. Phase 9 — Dashboard Integration (Week 11–12)

### 14.1 Tasks
- [ ] Build Streamlit multipage app per the structure in §3 (`app/app.py` + `app/pages/`)
- [ ] Company/customer selector (dropdown, session-state persisted across pages)
- [ ] Page: **Overview** (`app.py`) — company selector, headline risk score (large number + color-coded badge: green <30, yellow 30-60, red >60), explanation text, top 3 recommendations
- [ ] Page: **Churn** — churn probability, feature importance bar chart, per-customer table
- [ ] Page: **Sentiment** — sentiment distribution pie/bar, sentiment trend over time, sample flagged negative reviews
- [ ] Page: **Anomalies** — time-series line chart with anomalies highlighted (red markers)
- [ ] Page: **Forecast** — actual vs forecast line chart with confidence band
- [ ] Page: **Risk & Recommendations** — SHAP force plot (or SHAP summary bar as Streamlit-friendly alternative), full explanation text, full recommendation list

### 14.2 Notes
- Use `st.cache_data`/`st.cache_resource` for loaded models/data to satisfy NFR1 (<5s load)
- Every page must handle "no data for this company" gracefully (no crashes)

### Definition of Done
- `streamlit run app/app.py` launches without error
- All 6 pages render for every demo company from Phase 10
- Dashboard load time for a single company < 5 sec (NFR1) — measure and note actual time

**Deliverable:** Working `app.py` — full demo-able Streamlit dashboard.

---

## 15. Phase 10 — Testing, Polish & Documentation (Week 13)

### 15.1 Demo companies (build these via the Phase 1 synthetic generator)

| Company | Profile | churn_prob | sentiment | anomalies | forecast |
|---|---|---|---|---|---|
| "GreenLeaf SaaS" | Healthy | ~0.10 | mostly positive | none | steady growth |
| "RedFlag Analytics" | At-risk | ~0.70 | mostly negative | revenue drop injected | declining |
| "MixedCo" | Mixed | ~0.40 | mixed | 1 minor anomaly | flat |

### 15.2 Tasks
- [ ] `run_full_pipeline_demo.py` — single script that runs preprocessing → all 4 models → risk_aggregator → recommendation_engine end-to-end for all 3 demo companies and prints/saves a summary JSON per company
- [ ] Fix bugs surfaced by the end-to-end run
- [ ] UI polish pass on the dashboard
- [ ] Write final `README.md`: setup instructions, architecture diagram (ASCII or Mermaid is fine), screenshots, how to run
- [ ] Prepare a short demo script/talking points for the Unstop submission (bullet list: problem → approach → live demo flow → key metrics)

### Definition of Done
- `python run_full_pipeline_demo.py` completes with no errors for all 3 demo companies
- README lets a stranger clone the repo and run the dashboard in under 10 minutes following only the written steps
- All `pytest tests/` pass

**Deliverable:** Polished MVP + README + demo script/screenshots.

---

## 16. Phase 11 (Post-MVP, Optional) — API & Deployment

Only start this if Phase 10 is fully done and time/scope allows.

### 16.1 Tasks
- [ ] Wrap modules in FastAPI endpoints
- [ ] SQLite (→ Postgres later) for storing predictions/history
- [ ] Dockerize the app
- [ ] Basic GitHub Actions CI (lint + test on push)

### 16.2 Suggested endpoints

```
GET  /health
POST /predict/churn            body: {customer features}      → {churn_prob}
POST /predict/sentiment        body: {text}                   → {label, score}
POST /predict/anomaly          body: {date, revenue, ...}      → {is_anomaly, score}
POST /predict/forecast         body: {company_id, horizon}     → {forecast, bounds}
POST /risk-score                body: {company_id}              → {risk_score, top_factors, explanation_text, recommendations}
```

**Deliverable:** Dockerized, API-served version — only if time/scope allows post-competition.

---

## 17. Testing & Validation Strategy (Cross-Phase)

- **Unit tests** (`tests/`): one file per `src/` module, covering happy path + edge cases (empty input, missing columns, out-of-range values)
- **Integration test**: `run_full_pipeline_demo.py` (Phase 10) is the single source of truth that everything connects correctly
- **Metric gates**: don't silently accept a model that misses §2.5 thresholds — document the gap and what was tried, in the relevant `reports/*.md`
- **Reproducibility check**: run any `train()` function twice with the same seed, confirm identical metrics — this is a real test, not just a convention (add it to `tests/`)

---

## 18. Domain Glossary (Condensed — full version in `TERMS_GLOSSARY.md`)

| Term | Meaning |
|---|---|
| Churn | Customer product/service use band kar dena; churn prediction = kaun chhodega, predict karna |
| RFM | Recency, Frequency, Monetary — customer behavior features |
| Risk Score | Single 0-100 number, jitna high utna zyada danger |
| SHAP | Har feature ne prediction me kitna contribute kiya, ye explain karta hai |
| ROC-AUC | Model positive/negative classes ko kitna achhe se differentiate karta hai (0.5=random, 1.0=perfect) |
| F1 (macro) | Precision-Recall ka balance, averaged equally across classes |
| Isolation Forest | Anomaly detection model — jaldi "isolate" hone wale points anomaly maane jaate hain |
| Prophet | Facebook ka forecasting tool, trend+seasonality auto-detect karta hai |
| MAPE | Forecast error, percentage me |
| Baseline Model | Sabse simple model, comparison ke liye pehle banaya jaata hai |

*(Poore glossary ke liye `TERMS_GLOSSARY.md` refer karo — business terms, ML terms, NLP terms, time-series terms, XAI terms, evaluation metrics, aur MLOps terms sab detail me hain.)*

---

## 19. Fallback & Contingency Plan

Agar time/compute tight ho, is order me drop karo (pehla item sabse pehle drop karne layak, no big loss to MVP quality):

1. **LSTM forecasting model** — Prophet/ETS alone is enough; LSTM was always a stretch goal
2. **DistilBERT fine-tuning** — TF-IDF + Logistic Regression baseline is a fully acceptable MVP sentiment model
3. **Phase 6 (Forecasting) entirely** — MVP still complete lagega with churn + sentiment + anomaly + SHAP + dashboard; adjust `risk_aggregator` to drop the `w_forecast` term and re-normalize remaining weights
4. **Phase 11 (API/Deployment)** — this was always post-MVP/optional

Never drop: Churn module, Risk Aggregator + SHAP, Dashboard, README. These are the demo's core.

---

## 20. Master Checklist (Roll-up)

- [ ] Phase 0 — Repo scaffolded, environment ready
- [ ] Phase 1 — All 3 datasets loading cleanly
- [ ] Phase 2 — `preprocessing.py` + EDA report done
- [ ] Phase 3 — Churn model, ROC-AUC > 0.80
- [ ] Phase 4 — Sentiment model, macro-F1 > 0.75
- [ ] Phase 5 — Anomaly detector validated against injected anomalies
- [ ] Phase 6 — Forecast module beats naive baseline (or documented why not)
- [ ] Phase 7 — `risk_aggregator.py` with SHAP explanations working
- [ ] Phase 8 — `recommendation_engine.py` tested on 3-4 scenarios
- [ ] Phase 9 — Full 6-page Streamlit dashboard live
- [ ] Phase 10 — End-to-end demo script passes for all 3 demo companies, README complete
- [ ] Phase 11 (optional) — FastAPI + Docker + CI

---

## Appendix A — `requirements.txt` (starting point)

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
transformers>=4.35
datasets>=2.14
torch>=2.1
shap>=0.44
streamlit>=1.28
prophet>=1.1
statsmodels>=0.14
matplotlib>=3.7
plotly>=5.17
pytest>=7.4
pyyaml>=6.0
kagglehub>=0.2
fastapi>=0.104
uvicorn>=0.24
sqlalchemy>=2.0
```

## Appendix B — `.gitignore` (starting point)

```text
venv/
__pycache__/
*.pyc
data/raw/*.csv
data/processed/*.csv
models/*.pkl
models/sentiment_model/
.env
.streamlit/secrets.toml
*.log
```

---

*Is document ko `PROGRESS.md` ke saath side-by-side maintain karo — jaise-jaise phases complete hote jao, `PROGRESS.md` me tick karte jao. Agar kabhi is spec se deviate karna pade, `DECISIONS.md` me likhna mat bhoolna.*
