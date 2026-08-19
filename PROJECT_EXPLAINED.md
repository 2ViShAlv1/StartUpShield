# StartupShield AI — Ab Tak Kya Hua (Poora Explanation)

*Ye file poore project ka ek jagah, seedha Hinglish walkthrough hai — kya bana, kyu bana, aur result kya nikla. Detail chahiye ho toh har phase ke saath uski asli file link ki hai. Terms samajh na aaye toh `README_Terms_Glossary.md` khol lena.*

**Status: Phase 0-6 complete, Phase 7-11 abhi baaki hain.**

---

## Project Kya Hai (One Line)

Ek dashboard jo startup/SaaS company ke liye ek hi jagah bata de — "customer chhod ke jayega kya, reviews me kya feel kar rahe hain, revenue me kuch ajeeb toh nahi ho raha, aur agla mahina kaisa jayega" — sabko mila ke ek **0-100 Risk Score**.

Chaar alag AI models kaam karte hain, aur ek 5wa module unko jodta hai:

```
Churn Model + Sentiment Model + Anomaly Detector + Forecast Model
                        |
                        v
              Risk Aggregator (Phase 7 — abhi baaki)
                        |
                        v
                  0-100 Risk Score + "kyu" ka explanation
```

---

## Phase 0 — Repo Scaffold

Project ka skeleton banaya — folders (`src/`, `tests/`, `data/`, `models/`, `reports/`), `requirements.txt`, `config/config.yaml`, git setup. Ye foundation hai, koi model nahi bana yahan.

📄 [phase_notes/PHASE_00_REPO_SCAFFOLD.md](phase_notes/PHASE_00_REPO_SCAFFOLD.md)

---

## Phase 1 — Data Foundation

Teen datasets chahiye the: churn data, customer reviews, aur revenue time-series. Real Kaggle/HuggingFace datasets download nahi ho paaye (dependencies missing thi), toh **synthetic fallback data** banaya — code se hi realistic-looking fake data generate kiya (`src/generate_synthetic_data.py`).

Baad mein user ne ek **real Kaggle SaaS churn dataset** (`train.csv`, `test_.csv`) diya — usko anonymize karke (naam/email hataye, ID hash ki) `data/raw/churn.csv` bana diya.

⚠️ **Important caveat**: Sentiment aur time-series data abhi bhi synthetic hai — real nahi.

📄 [phase_notes/PHASE_01_DATA_FOUNDATION.md](phase_notes/PHASE_01_DATA_FOUNDATION.md)

---

## Phase 2 — EDA & Preprocessing

Data ko explore kiya (missing values, class balance, distributions dekhe), aur reusable helper functions banaye (`src/preprocessing.py`):
- Missing values fill karna
- Categorical columns ko one-hot encode karna
- Numeric columns scale karna
- Text clean karna (`clean_text` — ye function Phase 4 me reuse hua)

📄 [phase_notes/PHASE_02_EDA_PREPROCESSING.md](phase_notes/PHASE_02_EDA_PREPROCESSING.md) · [reports/eda_findings.md](reports/eda_findings.md)

---

## Phase 3 — Churn Prediction Model

**Kaam**: customer data dekh ke predict karna — ye customer chhod ke jayega kya (0/1 probability).

7 alag models try kiye (Logistic Regression, Random Forest, XGBoost, LightGBM, etc.), sabko compare kiya. **LightGBM jeeta** — ROC-AUC **0.8535** (target 0.80 se zyada).

⚠️ **Honest caveat**: 5 features me se 3 (`monthly_spend`, `plan_type`, `support_tickets`) real independent signals nahi hain — ye ek hi underlying variable (usage minutes) se derive hui hain. Feature importance dekhते waqt ye yaad rakhna.

📄 [phase_notes/PHASE_03_CHURN_MODEL.md](phase_notes/PHASE_03_CHURN_MODEL.md) · [reports/model_evaluation_churn.md](reports/model_evaluation_churn.md)

---

## Phase 4 — Sentiment Analysis Model

**Kaam**: customer review/text padh ke bataana — positive, negative, ya neutral feel kar raha hai.

TF-IDF + Logistic Regression use kiya (DistilBERT/transformer model try nahi kiya — TF-IDF hi target se zyada achha nikla, aur bahut fast hai). Macro-F1 = **1.0000**.

⚠️ **Honest caveat**: 1.0 score perfect hona suspicious hai — ye isliye kyunki data template-generated hai (real reviews itne clean-cut nahi hote). Real duniya me score kam hoga.

📄 [phase_notes/PHASE_04_SENTIMENT_MODEL.md](phase_notes/PHASE_04_SENTIMENT_MODEL.md) · [reports/model_evaluation_sentiment.md](reports/model_evaluation_sentiment.md)

---

## Phase 5 — Anomaly Detection

**Kaam**: revenue/users/signups ke daily data me "ajeeb" spikes ya drops pakadna — bina bataye ki normal kya hai (unsupervised learning).

Isolation Forest model use kiya, rolling averages aur z-scores jaise features banaye. Injected anomalies (test ke liye jaan-boojh kar dale gaye) me se **5/5 pakde** (Recall = 1.0).

⚠️ **Honest caveat**: Precision sirf 0.15 hai — matlab jitne flag kiye unme se bahut kam sach me anomaly the. Ye `contamination` parameter (3% flag karne ka fixed rule) ka mechanical result hai, model ki kamzori nahi.

📄 [phase_notes/PHASE_05_ANOMALY_DETECTION.md](phase_notes/PHASE_05_ANOMALY_DETECTION.md) · [reports/anomaly_validation.md](reports/anomaly_validation.md)

---

## Phase 6 — Forecasting

**Kaam**: agle 30 din ka revenue predict karna, saath me confidence range bhi ("₹22k-28k ke beech honge, 95% confident").

Prophet (main model), ETS/Holt-Winters (backup), aur seasonal-naive (dependency-free backup) — teeno banaye. Teeno demo companies pe simple "kal jaisa aaj bhi" guess (naive baseline) ko **beat kiya** — 5 se 13 percentage points behtar.

⚠️ **Honest caveat**: Zyada improvement sirf "weekly pattern pakadna" se aaya hai, model ki smartness se kam.

📄 [phase_notes/PHASE_06_FORECASTING.md](phase_notes/PHASE_06_FORECASTING.md) · [reports/model_comparison_forecast.md](reports/model_comparison_forecast.md)

---

## Cross-Phase Cleanup (Phase 5-6 ke beech)

Ek review kiya poore project ka — ye issues fixed kiye:
- **Reproducibility**: `src/train_all.py` bana jo scratch se sab models retrain kar sake (`python -m src.train_all`)
- **Bug fix**: anomaly module me ek row-ordering bug tha jab teen companies ka data jodte the — fix kiya
- Proxy features (Phase 3 wala issue) code + reports me clearly label kiye
- Notebooks execute karke outputs save kiye
- `requirements.txt` ko core aur optional me split kiya

---

## Abhi Baaki Kya Hai (Phase 7-11)

| Phase | Kaam | Status |
|---|---|---|
| **7 — Risk Aggregator + SHAP** | Sab 4 models ko ek weighted formula se combine karna, aur "kyu" ye score aaya wo SHAP se explain karna | ❌ Not started |
| **8 — Recommendation Engine** | Risk score dekh ke actionable suggestion dena ("retention outreach karo") | ❌ Not started |
| **9 — Dashboard** | Streamlit ka 6-page live dashboard | ❌ Not started |
| **10 — Testing & Docs** | End-to-end demo script, final README | ❌ Not started |
| **11 — API/Deployment (optional)** | FastAPI + Docker + CI | ❌ Not started, optional hai |

---

## Numbers — Ek Nazar Me

| Module | Metric | Target | Result |
|---|---|---:|---:|
| Churn | ROC-AUC | > 0.80 | **0.8535** ✅ |
| Sentiment | Macro-F1 | > 0.75 | **1.0000** ✅ (⚠️ synthetic data artifact) |
| Anomaly | Recall | > 0.60 | **1.0000** ✅ |
| Forecast | Beats naive baseline | Yes/No | **3/3 companies** ✅ |

**Tests**: 30/30 pass. **Reproduce karne ka command**: `python -m src.generate_synthetic_data && python -m src.train_all`

---

## Sab Files Kahan Milengi

- **Code**: `src/*.py` (ek file per module)
- **Har phase ka detail**: `phase_notes/PHASE_0X_*.md`
- **Metrics/results**: `reports/*.md`
- **Terms ka matlab**: `README_Terms_Glossary.md`
- **Decisions aur unki wajah**: `DECISIONS.md`
- **Checklist**: `PROGRESS.md`
