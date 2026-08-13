# EDA Findings

## Churn Data

- File: `data/raw/churn.csv`
- Shape: 2,500 rows x 7 columns.
- Missing values: none across all columns.
- Target balance: 1,592 retained customers (63.68%) and 908 churned customers (36.32%).
- This is moderately imbalanced, so Phase 3 should use stratified splits and either `class_weight="balanced"` or SMOTE-style balancing during model training.

### Churn Signals

Numeric correlation with `churn_label`:

| Feature | Correlation |
| --- | ---: |
| `monthly_spend` | -0.4989 |
| `usage_frequency` | -0.4471 |
| `support_tickets` | 0.3297 |
| `tenure` | -0.0254 |

Average feature values by churn label:

| `churn_label` | `tenure` | `monthly_spend` | `usage_frequency` | `support_tickets` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18.66 | 65.14 | 19.42 | 0.09 |
| 1 | 18.10 | 39.69 | 6.81 | 0.36 |

Plan-level churn rate:

| `plan_type` | Churn Rate |
| --- | ---: |
| `basic` | 64.9% |
| `pro` | 15.5% |
| `enterprise` | 6.6% |

Key takeaway: churn is most associated with lower spend, lower usage frequency, more support ticket risk, and the `basic` plan segment. `tenure` is weak in this transformed dataset and should not be relied on alone.

## Sentiment Review Data

- File: `data/raw/reviews.csv`
- Shape: 3,000 rows x 4 columns.
- Missing values: none across all columns.
- Class balance: exactly 1,000 negative, 1,000 neutral, and 1,000 positive reviews.
- Review length is similar across classes, with means around 11 words, so length alone is unlikely to be a useful sentiment signal.

Review length by sentiment:

| Sentiment | Count | Mean Words | Median Words | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `negative` | 1,000 | 10.89 | 9 | 8 | 20 |
| `neutral` | 1,000 | 10.93 | 9 | 8 | 20 |
| `positive` | 1,000 | 11.16 | 9 | 7 | 20 |

Star rating by sentiment:

| Sentiment | Mean Stars | Min | Max |
| --- | ---: | ---: | ---: |
| `negative` | 1.40 | 1 | 2 |
| `neutral` | 3.00 | 3 | 3 |
| `positive` | 4.62 | 4 | 5 |

Top TF-IDF terms by class:

| Sentiment | Top Terms |
| --- | --- |
| `negative` | billing, features, missing, poor experience, failing, failing support, keeps, keeps failing |
| `neutral` | reporting, product, support, support answered, feel, feel slow, slow, workflows |
| `positive` | love, saves, saves time, time, time week, week, feels stable, product |

Key takeaway: the synthetic reviews are clean and balanced, with obvious phrase-level class signals. Phase 4 should still preserve punctuation in `clean_text` because `!` and `?` can carry sentiment.

## Time-Series Data

Files:

- `data/raw/timeseries_greenleaf_saas.csv`
- `data/raw/timeseries_redflag_analytics.csv`
- `data/raw/timeseries_mixedco.csv`

All three files cover 2025-01-01 through 2025-12-31, have 365 rows, and have no missing values.

| Company | Injected Anomalies | Mean Revenue | Min Revenue | Max Revenue | Revenue Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| GreenLeaf SaaS | 0 | 19,087.80 | 11,446.44 | 31,318.31 | +92.80% |
| RedFlag Analytics | 4 | 9,369.16 | 4,898.83 | 17,737.81 | -42.34% |
| MixedCo | 1 | 10,193.16 | 7,774.29 | 12,763.39 | +10.48% |

Key takeaway: the three time-series profiles are intentionally different. GreenLeaf is healthy growth, RedFlag is declining with multiple injected anomalies, and MixedCo is mostly stable with one anomaly. These are useful demo companies for later anomaly, forecast, risk, and dashboard phases.

## Phase 2 Modeling Implications

- Use stratified train-test splitting for churn because the class balance is not equal.
- Keep `monthly_spend`, `usage_frequency`, `support_tickets`, and `plan_type` as high-priority churn features.
- Use one-hot encoding for `plan_type`.
- Scale numeric churn features for linear models such as Logistic Regression.
- Keep tree-based models available for non-linear feature interactions.
- Sentiment modeling can start with TF-IDF plus a linear classifier before using heavier transformer models.
- Time-series modules should evaluate trend and anomaly behavior per company, not only global aggregate metrics.
