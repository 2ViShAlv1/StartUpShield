# Phase 6 — Forecasting Module

## What We Have Done

- Implemented `src/forecast_module.py` with the full spec interface:
  `chronological_split`, `train`, `predict`, `evaluate`, `save_model`, `load_model`.
- Split each company's series **chronologically** — last 30 days held out, never shuffled.
- Shipped three backends, each satisfying the same contract:
  - **Prophet** (configured default) with its native 95% uncertainty interval.
  - **ETS / Holt-Winters** (`statsmodels`) with additive trend + additive weekly seasonality.
  - **Seasonal naive** (no dependencies) that repeats the last observed week.
- `predict()` returns `date, forecast, lower_bound, upper_bound` — confidence bounds for every
  future date, which FR4 requires and the code enforces.
- Evaluated MAE, RMSE, MAPE, and interval coverage against a last-value-carried-forward
  naive baseline.
- Fitted one model per company and saved to `models/forecast_model_<company>.pkl`.
- Wired forecast into `src/train_all.py` (`python -m src.train_all --module forecast`).
- Added 9 tests in `tests/test_forecast_module.py`.
- Wrote `reports/model_comparison_forecast.md` with actual-vs-predicted plots in
  `reports/figures/`.

## Current Phase 6 Status

- Phase 6 deliverables are complete.
- Requirement: forecast must beat the naive baseline on MAPE.
- Result: **3/3 companies beat it, on every backend**, by 5.3 to 12.7 percentage points.
- Requirement: confidence bounds on every future date. Result: passed.
- Status: passed.
- LSTM stretch goal: deliberately skipped (see `DECISIONS.md`).

## Model Results

Mean across the three demo companies, 30-day horizon:

| Backend | Mean MAPE | Mean coverage | Mean fit time |
| --- | ---: | ---: | ---: |
| ETS (Holt-Winters) | **4.48%** | 1.00 | 0.18 s |
| Prophet | 4.59% | 0.90 | 0.19 s |
| Seasonal naive | 5.19% | 1.00 | 0.002 s |
| Naive baseline | 13.05% | n/a | — |

**Read this table carefully.** The seasonal-naive model does no modelling at all — it just
repeats last week — and already gets 5.19%. Prophet and ETS add only ~0.6-0.7 pp on top.
So the large win over the naive baseline is mostly **the value of capturing weekly
seasonality**, not proof that Prophet is doing something clever. Say it that way if a judge
asks; claiming the sophisticated model earned all 8.5 points of improvement would be wrong.

Also note Prophet's interval under-covered on GreenLeaf SaaS (0.80 vs a nominal 0.95).

---

## What You Should Study

Ordered roughly by how much it will help you defend this phase.

### 1. Why time series is different (most important)

- **Why you must never random-shuffle a time series.** A random split lets the model train on
  Thursday and test on the Wednesday before it — you leak the future into the past and get a
  beautiful score that means nothing. This is the single most common time-series mistake.
  Understand why `chronological_split` exists and what it protects you from.
- **Backtesting / rolling-origin (walk-forward) validation.** We used one fixed holdout. The
  proper version slides the split forward repeatedly and averages. Know that our single split
  is a simplification, and why more folds would give a more trustworthy number.

### 2. Baselines and honest evaluation

- **Naive baselines**: last-value-carried-forward, seasonal naive, drift. Study *why a
  baseline is mandatory* — "MAPE 4.6%" alone is meaningless until you know what a trivial
  model scores. Our seasonal-naive result is the concrete lesson here.
- **MAE vs RMSE vs MAPE.** MAE is average error in real units. RMSE squares errors so it
  punishes big misses harder. MAPE is a percentage, so it compares across companies with
  different revenue scales — but it **blows up near zero** and penalises over-forecasting
  differently from under-forecasting. Know when MAPE misleads (and look up sMAPE and MASE as
  the fixes).

### 3. Decomposition: trend, seasonality, residual

- Every classical forecaster splits a series into **trend + seasonality + residual**.
- **Additive vs multiplicative** seasonality: additive when the seasonal swing is a fixed
  amount, multiplicative when it grows with the level. We used additive — be able to say why.
- Weekly vs yearly seasonality, and why we disabled yearly (only 365 days — you cannot fit a
  yearly cycle from one year).

### 4. The two models we actually shipped

- **Prophet**: an additive model of trend + seasonality + holidays, with automatic
  changepoint detection for trend shifts. Study changepoints, `interval_width`, and the fact
  that its uncertainty interval comes from simulation, not a closed-form formula.
- **ETS / Holt-Winters**: exponential smoothing with level, trend, and seasonal components.
  Study the `alpha`/`beta`/`gamma` smoothing parameters and what "exponential" means here —
  recent observations weigh more, older ones decay geometrically.
- Then read *why statistical models often beat neural networks on short univariate series*.
  This is exactly why we skipped the LSTM, and it is a strong answer if anyone asks "why no
  deep learning?"

### 5. Confidence / prediction intervals

- The difference between a **confidence interval** (uncertainty about a parameter) and a
  **prediction interval** (uncertainty about a future observation). We produce the latter.
- **Coverage**: if you claim a 95% interval, ~95% of actuals should fall inside it. Ours hit
  0.80 on one company — understand what under-coverage means and why an over-confident
  interval is worse than a wide one.
- Why intervals widen with horizon, and why our ETS path multiplies by `sqrt(step)` (a
  random-walk error-growth assumption). Know that this is an approximation.

### 6. Concepts worth a quick look

- Stationarity, differencing, and the ADF test — the vocabulary ARIMA is built on.
- ARIMA/SARIMA, so you can say what we did *not* use and why.
- Autocorrelation (ACF/PACF) plots for spotting seasonality in the first place.

### Questions you should be able to answer after studying

1. Why can't I use `train_test_split` on time-series data?
2. My MAPE is 4%. Is that good? (Correct answer: *compared to what baseline?*)
3. Why does the interval get wider further into the future?
4. Why did you not use an LSTM?
5. What breaks about all of this when real data has a promotion, an outage, or a holiday?

---

## Useful Files

- `src/forecast_module.py`
- `tests/test_forecast_module.py`
- `src/train_all.py` (the `train_forecast` function)
- `reports/model_comparison_forecast.md`
- `reports/figures/forecast_*.png`
- `models/forecast_model_*.pkl`
- `config/config.yaml` (`forecast.model_type`, `forecast.horizon_days`)
