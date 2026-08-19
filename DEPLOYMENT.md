# Deploying StartupShield AI

The dashboard is built to deploy to **Streamlit Community Cloud** (free) with no
manual setup steps. This document covers testing it locally first, then deploying.

---

## How deployment works here

`models/*.pkl` is gitignored, so a fresh deploy arrives with **no model artifacts**.
Rather than committing ~11 MB of pickles — which are fragile across library versions
and would silently fail to unpickle on a host running a different scikit-learn — the
app **trains its models on first load, in the deployment environment**:

```
first page load
      ↓
models/ empty?  ──no──►  load and serve
      │ yes
      ▼
data/raw/*.csv present?  ──no──►  generate them
      │ yes
      ▼
train all four modules  (~4 s)
      ↓
cache in st.cache_resource → serve
```

`data/raw/*.csv` **is** committed: the CSVs are small (556 KB total), plain text, and
carry no version fragility, so they act as the reproducible seed. A cold start takes
about 4 seconds and happens once per container, not once per user.

---

## 1. Test locally first

```bash
# from the repo root
source .venv/bin/activate          # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                             # expect 88 passed
streamlit run app/app.py           # opens http://localhost:8501
```

### Verifying the cold-start path

This is the single most important thing to test, because it is what the deployed app
does on its very first request:

```bash
rm -rf models/*.pkl models/sentiment_model
streamlit run app/app.py
```

The first page load should show *"Preparing models…"* for a few seconds, then render a
risk score. If that works locally, it works on Streamlit Cloud.

### Manual test checklist

| # | Step | Expected |
| --- | --- | --- |
| 1 | Open the app | Overview page, **GreenLeaf SaaS**, green badge ≈ **19** |
| 2 | Sidebar → **RedFlag Analytics** | Orange badge ≈ **57**, red 🔴 critical recommendation |
| 3 | Sidebar → **MixedCo** | Green badge ≈ **25** |
| 4 | **Churn** page | Histogram, SHAP bar chart, customer table with progress bars |
| 5 | **Sentiment** page | Sentiment mix bars, weekly positivity trend, flagged negatives |
| 6 | **Anomalies** page | Line chart with **red dots** on flagged days; try the metric dropdown |
| 7 | **Forecast** page | Backtest + orange forward projection with a shaded confidence band |
| 8 | **Risk & Recommendations** | Score breakdown table summing to the headline number |
| 9 | **Upload Your Company** | Download a template, re-upload it → should error (*only 3 rows*) |
| 10 | Upload a real file | Score renders; **Download report (JSON)** works |

Step 9 is worth doing deliberately: it proves validation rejects bad input with a
readable message instead of crashing.

### Generating a realistic test file

```bash
python - <<'PY'
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 150
pd.DataFrame({
    "customer_id": [f"TEST-{i:04d}" for i in range(n)],
    "tenure": rng.integers(1, 36, n),
    "monthly_spend": rng.normal(55, 18, n).round(2).clip(10),
    "usage_frequency": rng.integers(0, 12, n),
    "support_tickets": rng.poisson(1.1, n),
    "plan_type": rng.choice(["freemium", "team", "business"], n),
}).to_csv("test_customers.csv", index=False)

days = 90
rev = 8000 * (0.998 ** np.arange(days)) + rng.normal(0, 200, days)
rev[60:63] *= 0.6                      # inject an incident to see anomaly detection fire
pd.DataFrame({
    "date": pd.date_range("2025-05-01", periods=days).astype(str),
    "revenue": rev.round(2),
    "active_users": (rev / 20).round().astype(int),
    "signups": rng.integers(3, 15, days),
}).to_csv("test_metrics.csv", index=False)
print("wrote test_customers.csv and test_metrics.csv")
PY
```

Upload both on the **Upload Your Company** page. Measured result:

```
RISK 40.0 [Medium]
contributions: Churn 11.9 | Sentiment 0.0 | Anomalies 20.0 | Forecast 8.2
anomalous days flagged: 3
```

Two things worth noticing. The **Anomalies** signal contributes its full 20 points — the
injected incident at day 60 is detected without being told about it. And **Sentiment
contributes 0.0** because no reviews file was uploaded: a missing signal reads as neutral
rather than as risk, which is why the score lands at Medium rather than High. Add a
reviews CSV with negative text and the score climbs.

---

## 2. Deploy to Streamlit Community Cloud

### Prerequisites

- A **public** GitHub repo (Community Cloud cannot read private repos on the free tier)
- A Streamlit account linked to that GitHub account

### Steps

**1. Push the repo.**

```bash
git add -A
git commit -m "Add dashboard, upload flow, and deployment config"
git push origin main
```

Confirm `data/raw/*.csv` is actually in the commit — without it the deploy has no seed
data:

```bash
git ls-files data/raw/
```

**2. Create the app.** Go to <https://share.streamlit.io> → **New app**, then set:

| Field | Value |
| --- | --- |
| Repository | `<your-username>/StartUpShield` |
| Branch | `main` |
| Main file path | `app/app.py` |

**3. Deploy.** First build takes 3–6 minutes (installing Prophet is the slow part).
The first page load then trains the models for ~4 seconds.

### Files that make this work

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `packages.txt` | `build-essential`, needed by Prophet's Stan backend |
| `runtime.txt` | Pins Python 3.12 |
| `.streamlit/config.toml` | Theme, 25 MB upload cap, XSRF protection on |

---

## Troubleshooting

**Build fails on `prophet`.** This is the most likely failure and it is not fatal.
`forecast_module` catches Prophet failures at *fit* time as well as import time and
falls back to statsmodels ETS, which needs no compiled dependency — on the demo data
ETS actually scored marginally better (4.48% vs 4.59% mean MAPE). If Prophet blocks the
build entirely, remove `prophet>=1.1` from `requirements.txt` and set
`forecast.model_type: ets` in `config/config.yaml`.

**App boots but shows no risk score.** Models failed to train. Check the Streamlit Cloud
logs ("Manage app" → logs) and confirm `data/raw/*.csv` is present in the repo.

**"Oh no. Error running app."** Almost always a missing dependency. Compare the traceback
in the logs against `requirements.txt`.

**Slow first load.** Expected — that is the one-time model training. Subsequent loads are
served from `st.cache_resource`.

**Memory limits.** Community Cloud allows ~1 GB. Current usage is comfortably under that;
if you add heavier models later, drop the extra churn comparison artifacts
(`churn_model_random_forest.pkl` is the largest at ~3.9 MB) from `train_all.py`.

---

## Security notes

- Uploaded files are held in memory for the request only. Nothing is written to disk or
  transmitted anywhere; the only output is a JSON report the user chooses to download.
- Upload size is capped at 25 MB (`.streamlit/config.toml`) so a stray large file cannot
  exhaust memory on a shared host.
- All uploads pass through `src/data_validation.py` before reaching any model.
- `.streamlit/secrets.toml` is gitignored. This app needs no secrets or API keys.
- The app is read-only with respect to user data — there is no database and no
  persistence layer, which is also why there is no authentication.
