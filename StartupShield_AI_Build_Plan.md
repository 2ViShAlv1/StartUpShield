# StartupShield AI — Requirements & Phase-wise Build Plan

*Ab actual banane ka plan — requirements pehle, phir har phase me kya-kya banana hai, deliverables ke saath.*

---

## Part A: Requirements

### A1. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System customer-level tabular data leke churn probability predict karega |
| FR2 | System review/ticket text leke sentiment score/label output karega |
| FR3 | System business metrics (revenue/DAU) me anomaly flag karega |
| FR4 | System future revenue/DAU forecast karega with confidence range |
| FR5 | System sab module outputs ko combine karke ek composite risk score (0-100) generate karega |
| FR6 | System risk score ke top contributing factors SHAP se explain karega |
| FR7 | System risk drivers ke basis pe rule-based recommended actions dega |
| FR8 | System sab kuch ek dashboard pe visualize karega (per-company view) |
| FR9 | User CSV upload karke apna data analyze kar sakega (MVP me local, baad me API) |

### A2. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Dashboard load time < 5 sec for a single company's data |
| NFR2 | Model inference < 2 sec per prediction (real-time feel) |
| NFR3 | Code modular hona chahiye — har model alag Python module/file me, reusable |
| NFR4 | Reproducibility — same input pe same output (fixed random seeds) |
| NFR5 | Explainability — koi bhi prediction "black box" nahi honi chahiye, SHAP se traceable |
| NFR6 | Data privacy — koi bhi PII (agar real data use kare) hash/anonymize honi chahiye |
| NFR7 | System ko synthetic data pe fully demo-able hona chahiye (bina real data ke bhi chal sake) |

### A3. Technical Requirements (Stack)

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **ML:** scikit-learn, XGBoost
- **NLP:** HuggingFace Transformers (DistilBERT), scikit-learn (TF-IDF)
- **Anomaly:** scikit-learn (Isolation Forest)
- **Forecasting:** Prophet or PyTorch (LSTM)
- **XAI:** SHAP
- **Dashboard:** Streamlit
- **Backend (post-MVP):** FastAPI
- **DB (post-MVP):** SQLite → PostgreSQL
- **Experiment tracking (post-MVP):** MLflow
- **Deployment (post-MVP):** Docker, GitHub Actions

### A4. Data Requirements

- **Churn dataset:** customer_id, tenure, monthly_spend, usage_frequency, support_tickets, plan_type, churn_label
- **Text dataset:** review_text/ticket_text (+ optional star_rating as weak label)
- **Time-series dataset:** date, revenue, active_users, signups
- **Source plan:** Kaggle Telco Churn (churn), Amazon/Yelp Reviews (sentiment), synthetic generator script for revenue/usage time series (since real SaaS metrics data publicly rare)

### A5. Success Criteria (MVP-level, not production)

- Churn model: ROC-AUC > 0.80 on test set
- Sentiment model: F1 (macro) > 0.75
- Anomaly detection: sensibly flags injected synthetic anomalies (qualitative check)
- Dashboard: shows risk score + explanation + recommendation for at least 1 demo company end-to-end

---

## Part B: Phase-wise Build Plan

*(Ye "build" phases hain — pichli guide ke "learning" phases se related hain but yaha focus deliverable pe hai, seekhne pe nahi. Assume tu concepts already cover kar chuka hai ya parallel me seekh raha hai.)*

### Phase 1 — Project Setup & Data Foundation (Week 1)
**Goal:** Clean repo + all datasets ready
- [ ] GitHub repo setup with folder structure (`data/`, `notebooks/`, `src/`, `models/`, `app/`)
- [ ] Download/prepare Telco Churn dataset, Reviews dataset
- [ ] Write synthetic data generator script (`generate_synthetic_data.py`) for revenue/usage time series with configurable seasonality + injected anomalies
- [ ] requirements.txt / virtual environment setup
- **Deliverable:** All 3 datasets ready and loading cleanly in a notebook

### Phase 2 — EDA & Preprocessing (Week 2)
**Goal:** Understand data, build reusable preprocessing pipeline
- [ ] EDA notebook per dataset (distributions, correlations, missing values, class balance)
- [ ] Build `preprocessing.py`: missing value handling, encoding, scaling functions
- [ ] Document key findings (which features correlate with churn, etc.)
- **Deliverable:** `preprocessing.py` module + EDA report (can reuse your report-writing style from FreshVision)

### Phase 3 — Churn Prediction Module (Week 3-4)
**Goal:** Working, evaluated churn classifier
- [ ] Feature engineering (RFM-style features)
- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest and XGBoost, tune with cross-validation
- [ ] Handle class imbalance (class_weight or SMOTE)
- [ ] Evaluate: ROC-AUC, PR-AUC, confusion matrix
- [ ] Save best model (`churn_model.pkl`)
- **Deliverable:** `churn_module.py` with `train()` and `predict()` functions + saved model + evaluation report

### Phase 4 — Sentiment Analysis Module (Week 5-6)
**Goal:** Working sentiment classifier for reviews/tickets
- [ ] Text cleaning pipeline
- [ ] TF-IDF + Logistic Regression baseline, evaluate
- [ ] Fine-tune DistilBERT (HuggingFace `Trainer` or `pipeline`) on same data, compare
- [ ] Decide which to ship in MVP (baseline is fine if compute-constrained)
- [ ] Save model/pipeline
- **Deliverable:** `sentiment_module.py` + comparison report (TF-IDF vs DistilBERT)

### Phase 5 — Anomaly Detection Module (Week 7)
**Goal:** Working anomaly flagger for business metrics
- [ ] Feature prep on time-series (rolling stats, day-of-week, etc.)
- [ ] Train Isolation Forest, tune contamination parameter
- [ ] Validate against injected synthetic anomalies (precision/recall on known injected points)
- [ ] Save model
- **Deliverable:** `anomaly_module.py` + validation notes

### Phase 6 — Forecasting Module (Week 8-9)
**Goal:** Working revenue/DAU forecaster
- [ ] Chronological train-test split
- [ ] Prophet baseline
- [ ] LSTM model (if time permits) — compare against Prophet and naive baseline
- [ ] Evaluate: MAE, RMSE, MAPE
- **Deliverable:** `forecast_module.py` + comparison plots (actual vs predicted)

### Phase 7 — Risk Aggregation + Explainability (Week 10)
**Goal:** Combine everything into one explainable risk score
- [ ] Design weighted formula: `risk_score = w1*churn_prob + w2*(1-sentiment_score) + w3*anomaly_flag + w4*forecast_deviation`
- [ ] Justify weights (domain reasoning, document assumptions clearly — this is a known simplification, mention it in report)
- [ ] Apply SHAP TreeExplainer on churn model (primary driver), generate summary + force plots
- [ ] Map top SHAP features + anomaly/sentiment flags → plain-language explanation string
- **Deliverable:** `risk_aggregator.py` producing `{risk_score, top_factors, explanation_text}` per company/customer

### Phase 8 — Recommendation Engine (Week 10, parallel with Phase 7)
**Goal:** Simple rule-based action suggestions
- [ ] Define rule table (e.g., high churn + negative sentiment → "prioritize retention outreach"; anomaly on revenue → "investigate payment/billing issue")
- [ ] Implement `recommendation_engine.py` mapping risk factors → action list
- **Deliverable:** Recommendation function tested on 3-4 example scenarios

### Phase 9 — Dashboard Integration (Week 11-12)
**Goal:** Everything visualized in one Streamlit app
- [ ] Multi-page Streamlit app: Overview, Churn, Sentiment, Anomalies, Forecast, Risk & Recommendations
- [ ] Company/customer selector
- [ ] Plot SHAP force plot, sentiment trend, anomaly timeline, forecast chart
- [ ] Display risk score + explanation + recommendations clearly
- **Deliverable:** Working `app.py` — full demo-able Streamlit dashboard

### Phase 10 — Testing, Polish & Documentation (Week 13)
**Goal:** MVP demo-ready
- [ ] End-to-end test with 2-3 synthetic "demo companies" (one healthy, one at-risk, one mixed)
- [ ] Fix bugs, clean UI
- [ ] Write README with setup instructions, architecture diagram, screenshots
- [ ] Prepare short presentation/demo script (for Unstop submission)
- **Deliverable:** Polished MVP + README + demo video/screenshots

### Phase 11 (Post-MVP, Optional) — API & Deployment
- [ ] Wrap modules in FastAPI endpoints
- [ ] Add SQLite/Postgres for storing predictions/history
- [ ] Dockerize the app
- [ ] Basic GitHub Actions CI (lint + test on push)
- **Deliverable:** Dockerized, API-served version — only if time/scope allows post-competition

---

## Part C: Suggested Timeline Summary

| Phase | Duration | Cumulative |
|---|---|---|
| 1. Setup & Data | 1 wk | Week 1 |
| 2. EDA & Preprocessing | 1 wk | Week 2 |
| 3. Churn Module | 2 wk | Week 4 |
| 4. Sentiment Module | 2 wk | Week 6 |
| 5. Anomaly Module | 1 wk | Week 7 |
| 6. Forecasting Module | 2 wk | Week 9 |
| 7-8. Aggregation + Recommendations | 1 wk | Week 10 |
| 9. Dashboard | 2 wk | Week 12 |
| 10. Polish & Docs | 1 wk | Week 13 |

**Total: ~13 weeks (3 months) for a solid MVP**, doable alongside your internship if you dedicate weekend + evening blocks. Agar Unstop deadline isse tight hai, toh Phase 6 (Forecasting) sabse pehle drop kar — MVP still complete lagega churn+sentiment+anomaly+SHAP+dashboard ke saath hi.

---

## Part D: Repo Structure (Reference)

```text
startupshield-ai/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_churn.ipynb
│   ├── 02_eda_sentiment.ipynb
│   └── 03_eda_timeseries.ipynb
├── src/
│   ├── preprocessing.py
│   ├── churn_module.py
│   ├── sentiment_module.py
│   ├── anomaly_module.py
│   ├── forecast_module.py
│   ├── risk_aggregator.py
│   └── recommendation_engine.py
├── models/
│   ├── churn_model.pkl
│   ├── sentiment_model/
│   └── anomaly_model.pkl
├── app/
│   └── app.py  (Streamlit)
├── requirements.txt
└── README.md
```

---

*Jab ready ho phase 1 start karne ke liye, bata — data generator script se shuru kar sakte hain, ya jo dataset finalize kiya hai uski EDA se.*
