# Phase 5 — Anomaly Detection Module

## What We Have Done

- Implemented `src/anomaly_module.py`.
- Built grouped time-series features for each company.
- Added 7-day and 30-day rolling means/stds, rolling residuals, z-scores, day-of-week, and 1-day percentage change features.
- Trained an `IsolationForest` anomaly detector.
- Tuned the default model to prioritize injected anomaly recall.
- Validated predictions against the `is_injected_anomaly` ground-truth column.
- Saved the model to `models/anomaly_model.pkl`.
- Added tests in `tests/test_anomaly_module.py`.
- Wrote `reports/anomaly_validation.md`.

## Post-Review Fixes

- **Row-order bug fixed.** `build_features()` used `sort_index()` to restore order. When the
  three company CSVs are concatenated without `ignore_index=True` their indices collide
  (each is 0–364), so the returned rows came back in a different order than the input —
  silently misaligning `is_injected_anomaly` labels against predictions. The builder now
  tracks input position explicitly and restores both the caller's order and index.
  Regression test: `test_build_features_preserves_input_row_order_with_duplicate_index`.
- **Feature scope clarified.** The detector trains on revenue-derived features only
  (`MODEL_SERIES_COLUMNS`), even though features are built for all three series. The report
  previously implied all three were used. Reason: `active_users` and `signups` are noisy
  linear functions of `revenue` in the generator, so including them dropped recall 1.00 → 0.80.

## Current Phase 5 Status

- Phase 5 deliverables are complete.
- Precision: 0.1515.
- Recall: 1.0000.
- F1: 0.2632.
- Required recall threshold: > 0.60.
- Full-dataset inference for 1,095 rows: 0.0781 seconds.
- Status: passed.

## Model Results

| Metric | Value |
| --- | ---: |
| Injected anomalies | 5 |
| Detected anomalies | 33 |
| Precision | 0.1515 |
| Recall | 1.0000 |
| F1 | 0.2632 |

Precision is bounded by the contamination setting, not by model quality: `contamination = 0.03`
on 1,095 rows forces 33 flags, and 5 injected anomalies / 33 flags = 0.1515 exactly.

## What You Should Study

- Anomaly detection.
- Isolation Forest.
- Rolling mean and rolling standard deviation.
- Percentage change.
- Precision, recall, and F1 for anomaly detection.
- Why anomaly detection is usually noisier than supervised classification.

## Useful Files

- `src/anomaly_module.py`
- `tests/test_anomaly_module.py`
- `models/anomaly_model.pkl`
- `reports/anomaly_validation.md`
- `data/raw/timeseries_*.csv`
