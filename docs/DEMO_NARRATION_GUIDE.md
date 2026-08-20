# StartupShield AI — Full Demo Narration & Explanation Guide

**Purpose of this file:** this is the document you keep open on screen while you
record the demo video. It has everything in one place — a deep explanation of the
project (so you actually understand what you're showing), the exact words to say
for every single screen, a cheat-sheet of every number you might quote, answers to
questions a judge might ask, and a glossary. Nothing about the project should be
outside this file.

**How to use it:**
1. Read **Section A** once, fully, before you touch the record button. This is
   where you actually *learn* the project — not memorize lines, understand it.
2. While recording, read from **Section B** — it tells you exactly what to click
   and exactly what to say, screen by screen.
3. Keep **Section C** (numbers) and **Section D** (Q&A) open in a second tab for
   anything you get asked live.

---

## Section A — The Project, Explained In Full

### A1. What problem this solves

A startup founder almost never finds out their company is in trouble at the right
time. They find out from a bad month's revenue report, or from a customer who
finally says "we're cancelling" — by which point the decision was already made
weeks earlier. The warning signs were there the whole time, just scattered across
different places: a support inbox, a reviews page, a billing dashboard, an
analytics tool. Nobody was watching all of them together.

**StartupShield AI's answer:** put four different early-warning signals in one
place, combine them into a single number, and explain *why* that number is what it
is — so the founder acts on a warning, not a post-mortem.

### A2. The architecture, in one picture

```
                     ┌─────────────────────┐
   customer data ───►│   Churn Model        │──┐
                     └─────────────────────┘  │
                     ┌─────────────────────┐  │
   reviews/tickets ─►│   Sentiment Model     │──┤
                     └─────────────────────┘  │      ┌───────────────────┐      ┌──────────────────────┐
                     ┌─────────────────────┐  ├─────►│  Risk Aggregator   │─────►│ Recommendation Engine │
   daily revenue ───►│   Anomaly Detector    │──┤      │ (weighted 0-100)   │      │  (rule-based actions) │
                     └─────────────────────┘  │      └───────────────────┘      └──────────────────────┘
                     ┌─────────────────────┐  │
   daily revenue ───►│   Forecast Model      │──┘
                     └─────────────────────┘
```

Four independent models, each an expert in one thing, feeding into one aggregator
that produces the score, and one recommendation engine that turns the score into
action. This is the whole product.

### A3. The four signal models — what each one actually does

#### 1. Churn Model — "who is about to leave?"

- **Input:** one row per customer — `tenure` (months as a customer),
  `monthly_spend`, `usage_frequency`, `support_tickets`, `plan_type`.
- **Model:** LightGBM (gradient-boosted decision trees) — chosen after comparing
  Logistic Regression, Random Forest, XGBoost, and LightGBM on the same data;
  LightGBM won on both ROC-AUC and PR-AUC.
- **Performance:** **~85% ROC-AUC** (a perfect model scores 100%, a coin flip
  scores 50%). Well above the 80% bar this project set for itself.
- **Explainability:** every churn prediction can be explained with **SHAP**
  (SHapley Additive exPlanations) — it shows exactly which features pushed a
  specific customer's risk up or down, not just a global "spend matters" statement.
- **Honest caveat (say this if asked, don't volunteer it):** the training data
  came from a real Kaggle SaaS dataset that didn't include spend or plan fields
  directly, so those two were derived mathematically from usage minutes. This is
  documented in the code and in `DECISIONS.md` — it doesn't affect the churn
  probability, but it does mean "spend drives churn" shouldn't be over-read from
  the feature importances.

#### 2. Sentiment Model — "how do customers feel?"

- **Input:** free text — reviews, support tickets, feedback.
- **Model:** TF-IDF (turns text into weighted word-frequency vectors) +
  Logistic Regression.
- **Why not a deep learning model (DistilBERT)?** It was considered and
  deliberately skipped — the simpler TF-IDF baseline already cleared the
  required accuracy bar and runs far faster on CPU, which matters for a live
  dashboard. This is a real engineering trade-off, not a shortcut.
- **Performance:** macro-F1 of **1.00** on the test set. **Say this number with
  the caveat attached** — it's a near-perfect score because the training reviews
  were template-generated text for the demo, so this measures "did the model
  learn the templates," not real-world generalization. Volunteering this caveat
  yourself is more convincing than a judge catching it.

#### 3. Anomaly Detector — "did something break?"

- **Input:** the daily revenue series.
- **Model:** Isolation Forest — an algorithm that isolates unusual points by how
  few random splits it takes to separate them from the rest of the data.
- **Features:** rolling z-scores, day-over-day percent change, and rolling
  residuals against a 7-day and 30-day trend — not raw revenue, so it adapts to
  each company's own normal range instead of a fixed threshold.
- **Performance:** catches **100% of injected test anomalies** (recall = 1.0).
  Precision is intentionally lower — the detector is tuned to flag every real
  problem rather than stay quiet, which is the right trade-off for an early-warning
  tool (a missed outage is worse than one extra day to double-check).

#### 4. Forecast Model — "where is revenue headed?"

- **Input:** the daily revenue series (or active users / signups).
- **Model:** Prophet (Facebook's time-series forecasting library) is the primary
  choice, with two automatic fallbacks if it's unavailable: a statsmodels
  Holt-Winters (ETS) model, then a dependency-free seasonal-naive model as a last
  resort — the dashboard never breaks for lack of a forecasting library.
- **Performance:** beats a naive "repeat last week" baseline by **5 to 13
  percentage points on MAPE** (Mean Absolute Percentage Error — lower is
  better) across the three demo companies, with a 95% confidence band on every
  prediction.
- **Requires 30+ days of history** to produce a meaningful backtest; with less,
  the dashboard skips forecasting rather than showing an unreliable number, and
  says so on screen.

### A4. The Risk Aggregator — how the four signals become one number

This is very likely to come up as a question, so understand it properly.

**The formula:**

```
risk_score = 100 × clip(
    0.35 × churn_term
  + 0.20 × sentiment_term
  + 0.20 × anomaly_term
  + 0.25 × forecast_term
, 0, 1)
```

**Where the weights came from:** they are a **hand-picked product judgement**,
set in `config/config.yaml`, not learned by a model. The honest reason: there is
no dataset anywhere of "which real companies actually failed" to train a weighting
model on — nobody has that labelled data. So the weights encode a reasoned
opinion: churn matters most (35%) because a lost customer is the most direct,
irreversible signal; sentiment, anomalies, and forecast are each roughly a fifth
because they're earlier and noisier warnings. This is stated directly in the code
comments and in `DECISIONS.md` rather than hidden — a transparent assumption is
more credible than a black box.

**Why the raw signals had to be rescaled first (the interesting part):** the four
signals don't naturally live on the same scale. Churn probability realistically
spans 0.1–0.9 across companies, but the anomaly ratio tops out near 0.15 and
forecast decline near 0.10. Feeding raw values straight into the weighted sum let
churn dominate the score no matter what the config said — on the first test, the
anomaly and forecast terms together moved the score by under 2 points out of 100,
even though they were assigned 45% of the weight. So each signal is divided by a
**saturation point** first — 10% of days anomalous, or a 10% revenue decline, both
count as "maximum risk" for that signal — so a 20% configured weight actually buys
20 points of real influence.

**Bands:** score < 30 → **Low** (green); 30–60 → **Medium** (orange); ≥ 60 →
**High** (red). These thresholds were left as originally set even when a demo
company scored right at the edge (Medium, not High) — deliberately, to avoid
tuning the bands just to make a demo look more dramatic.

### A5. The Recommendation Engine — from score to action

A rule table, not a model — `recommendation_engine.py` maps combinations of
signals to specific actions using explicit thresholds (e.g. churn probability
above 0.55 is "high," above 0.35 is "moderate"). This is deliberate: a founder
acting on a recommendation should be able to see exactly which number triggered
it, not trust an opaque suggestion. Recommendations are ranked critical → high →
medium → info, and anomaly recommendations name the specific metric that moved.

### A6. Upload Your Company — scoring a real business

This is the feature that turns the project from "a demo with 3 fake companies"
into something usable.

- **What's required:** just a customer list (tenure, spend, usage, tickets,
  plan). Reviews and daily revenue are optional — the score still works without
  them, using fewer signals, and the page says exactly which signal is missing
  and why.
- **What runs on upload:** churn and sentiment reuse the already-trained models
  (pure inference — a new company doesn't have enough labelled history to
  retrain a churn model from scratch, but "low usage + rising tickets precedes
  cancellation" transfers across companies). Anomaly detection and forecasting
  are **fit fresh, live, on the uploaded data** — every company's "normal day"
  is its own, so there's no useful pretrained baseline for that part. The whole
  thing runs in under 5 seconds.
- **Smart Import — the detail worth showing off:** real exports never use this
  project's column names. A Stripe export calls spend "Monthly Recurring
  Revenue," not `monthly_spend`, and doesn't export "tenure" at all — just a
  signup date. `src/smart_import.py` auto-detects what each column means (with a
  visible confidence level per guess, always overridable — it never silently
  guesses wrong), derives tenure from any date column, and can even collapse a
  raw per-ticket support export into a per-customer count. This means someone
  can upload an actual unmodified company export and it still works.
- **Validation:** every uploaded file is checked before it reaches a model, and
  every problem is reported at once in plain language — not one cryptic error
  at a time.
- **Nothing is stored:** scoring happens in-session; the only output is a JSON
  report the user chooses to download.

### A7. Tech stack, and why each piece

| Layer | Choice | Why |
| --- | --- | --- |
| Churn | LightGBM (scikit-learn ecosystem) | Best ROC-AUC/PR-AUC among 4 compared models |
| Sentiment | TF-IDF + Logistic Regression | Cleared the accuracy bar, far faster than a deep model on CPU |
| Anomalies | Isolation Forest | Adapts to each company's own baseline, no fixed threshold |
| Forecasting | Prophet (→ ETS → seasonal-naive fallback chain) | Native confidence intervals, never hard-fails |
| Explainability | SHAP | Per-prediction explanation, not just global importance |
| Dashboard | Streamlit | Fast to build, good caching model for a data app |
| Language | Python, end to end | One language across data, models, and UI |

### A8. Honest limitations (own these, don't wait to be caught)

- **Three demo companies are curated, two are synthetic.** They were built by
  slicing real behavioral data (usage, tickets) — never using model output — so
  the models still have to genuinely discover the risk, but they are demos, not
  real companies. Say this plainly if asked.
- **Sentiment's 1.00 macro-F1 is a template-data artifact**, not proof of
  real-world accuracy — covered in A3.
- **Churn's `monthly_spend`/`plan_type` are derived proxies**, not independent
  source fields — covered in A3.
- **The risk formula is a weighted sum, not a learned model** — covered in A4.
  All of this is documented in the repo's `DECISIONS.md` rather than hidden.
  Volunteering a limitation you already understand and handled well is a sign of
  engineering maturity, not a weakness — treat it that way on camera.

---

## Section B — Full Narration Script (read this while recording)

Timing is a guide, not a stopwatch — you're reading this on screen, so go at a
pace where you sound like you understand it, not like you're racing a clock.

### B0. Opening (≈20s)

**Show:** Overview page, freshly loaded, GreenLeaf SaaS selected (the default).

> "This is StartupShield AI. It's an early-warning dashboard for startups — it
> watches four different signals about a company and turns them into a single
> Risk Score from 0 to 100, with a plain-English explanation of why, and specific
> actions to take. Let me walk through how it works."

### B1. The Overview page (≈45s)

**Show:** Stay on GreenLeaf SaaS, then switch to **RedFlag Analytics** in the
sidebar.

> "Right now I'm looking at GreenLeaf SaaS. Risk Score of 19 out of 100 — Low
> risk, shown in green. Let me switch to a riskier company instead — RedFlag
> Analytics, from the sidebar. The score jumps to 57 — Medium risk, orange. Below
> the score, in plain English, it tells me why: churn risk is elevated, sentiment
> is leaning negative, and there are anomalous days in the revenue data. And right
> here are the top three things I should actually do about it — these aren't
> generic advice, they're generated from which signals are actually driving the
> score."

### B2. The Churn page (≈35s)

**Show:** Click **Churn** in the sidebar.

> "This page is what's behind the churn part of that score. This chart is a SHAP
> explanation — it shows exactly which factors are pushing risk up across the
> customer base, not just a black-box number. And this table ranks the actual
> customers most likely to churn, so it's not just a percentage, it's a list you
> could act on today."

### B3. The Sentiment page (≈30s)

**Show:** Click **Sentiment**.

> "This page reads through every customer review and support message and scores
> it as positive, neutral, or negative. Here's the overall mix, and here's how
> that's trending week over week. And these are the specific negative reviews
> flagged for follow-up — so it's not just 'sentiment is bad,' it's 'here's the
> actual complaint.'"

### B4. The Anomalies page (≈30s)

**Show:** Click **Anomalies**. Point out a flagged day.

> "This is the revenue timeline, and every red dot is a day the model flagged as
> unusual compared to that company's own normal pattern — not a fixed threshold,
> its own baseline. This kind of spike usually means an outage, a billing bug, or
> a pricing change worth investigating."

### B5. The Forecast page (≈30s)

**Show:** Click **Forecast**.

> "This is a 30-day backtest next to a forward projection, with a shaded
> confidence band — so instead of just knowing where revenue is today, I know
> roughly where it's heading, and how confident the model is about that."

### B6. The Risk & Recommendations page (≈25s)

**Show:** Click **Risk & Recommendations**.

> "And this page is the full breakdown behind that single Risk Score — exactly
> how much each of the four signals contributed, in points, so the headline
> number is never a mystery."

### B7. Upload Your Company — the highlight (≈70s)

**Show:** Click **Upload Your Company**. Upload the three files from
`sample_upload/`. If narrating the raw-export version, upload
`sample_upload/raw_exports/stripe_customers.csv` instead and show the
auto-detected column mapping. Click **Score my company**.

> "Everything so far has been a demo company. This page is where it becomes a
> real tool — anyone can upload their own customer list, reviews, and revenue,
> and get a real score. I'll upload a sample company now.
>
> Notice it doesn't ask me to rename any columns — if I upload something straight
> out of Stripe, with completely different column names and no 'tenure' column at
> all, it automatically figures out what each column means and even calculates
> tenure from the signup date. Every guess is shown with a confidence level and I
> can override any of them — it never silently guesses wrong.
>
> I'll click Score my company... and in a few seconds, this brand-new company —
> one the model has never seen before — scores 65 out of 100, High risk, and
> tells me exactly why: churn, negative sentiment, and revenue trending down are
> all contributing. I can download this whole result as a JSON report."

### B8. Closing (≈20s)

**Show:** Back to the Overview page or your face on camera.

> "Four real machine learning models, one explained score, and specific next
> steps — built so a founder finds out their company is at risk from a
> dashboard, not from a bad quarter. Thanks for watching."

---

## Section C — Key numbers cheat sheet (glance at this if you blank on a number)

| Signal | Model | Metric | Verified value |
| --- | --- | --- | --- |
| Churn | LightGBM | ROC-AUC | **~0.85** (85%) |
| Sentiment | TF-IDF + Logistic Regression | Macro-F1 | **1.00** *(synthetic-text caveat — say it yourself)* |
| Anomalies | Isolation Forest | Recall on injected anomalies | **1.00** (100%) |
| Forecast | Prophet | MAPE improvement vs. naive baseline | **5–13 percentage points** |
| Risk score weights | — | Churn / Sentiment / Anomaly / Forecast | **35% / 20% / 20% / 25%** |
| Upload → score latency | — | End to end | **Well under 5 seconds** |

| Demo company | Risk score | Band |
| --- | --- | --- |
| GreenLeaf SaaS | **18.8** | Low (green) |
| MixedCo | **25.3** | Low (green) |
| RedFlag Analytics | **57.4** | Medium (orange) |
| Sample upload company | **65.1** | High (red) |

---

## Section D — If a judge asks a question

Understand the idea, don't memorize the wording.

**"How were the risk score weights decided — 35/20/20/25?"**
> "They're a hand-picked product judgement, not learned by a model — there's no
> dataset of real companies that failed to statistically fit weights against.
> Churn gets the most weight because a lost customer is the most direct signal;
> the other three are earlier, noisier warnings, so they're roughly even. It's a
> documented assumption, not hidden."

**"What if I don't have all this data — reviews, daily revenue, everything?"**
> "Only the customer list is required. Reviews and daily revenue are optional —
> the score still works without them, using fewer signals, and it says exactly
> which one is missing and why, right on screen."

**"How accurate are the models, really?"**
> "Churn is 85% ROC-AUC, solid for this kind of problem. Forecasting beats a
> plain repeat-last-week guess by 5 to 13 percentage points. The three demo
> companies are synthetic, but the models and the scoring math are real — the
> Upload page proves it by scoring genuinely unseen data."

**"Why four separate models instead of one big model?"**
> "Each one is a different kind of problem — tabular customer data, free text,
> a time series for anomalies, a time series for forecasting — and each already
> has a well-understood, well-performing approach on its own. A single model
> trying to do all four would need far more data and would be harder to explain.
> Keeping them separate also means each one can be swapped or improved
> independently."

**"What technology did you use?"**
> "Python end to end. LightGBM for churn, TF-IDF and Logistic Regression for
> sentiment, Isolation Forest for anomalies, Prophet for forecasting — with
> automatic fallbacks if it's not installed — SHAP for explaining predictions,
> and Streamlit for the dashboard."

**"Can this actually scale to real companies?"**
> "That's exactly what the Upload page and the smart column-detection were built
> for — handling a real company's messy export, not just three demo companies.
> The natural next step is direct integrations, connecting straight to Stripe or
> a support tool instead of needing a CSV upload at all."

**"What's next, or what's the business model?"**
> "Right now it's upload-and-score. The next real step is a version that
> connects directly to a company's billing and support tools, refreshes
> automatically, and could be sold as a subscription early-warning system to
> startups or their investors."

**"Are you worried about the sentiment model's 100% F1 score looking too good to
be true?"**
> "It should look suspicious — it's not evidence of real-world accuracy. The
> training reviews were template-generated text for the demo, so that number
> measures how well the model learned the templates, not how it'd perform on
> real customer reviews. It's documented in the repo for exactly that reason."

---

## Section E — Glossary

| Term | Plain meaning |
| --- | --- |
| Churn | A customer cancelling or leaving |
| Sentiment analysis | Reading text to tell if it's positive, negative, or neutral |
| Anomaly detection | Spotting a day where a number looked unusual vs. that company's own normal |
| Forecasting | Predicting future numbers, like next month's revenue |
| SHAP | A method that explains *why* a model made one specific prediction, not just what matters in general |
| Risk Score | The single 0–100 number this project outputs, combining all four signals |
| ROC-AUC | A 0–1 score for how well a model tells two groups apart (e.g. churn vs. not); 0.5 is a coin flip, 1.0 is perfect |
| Macro-F1 | An accuracy measure for classification that treats every class equally, even rare ones |
| MAPE | Mean Absolute Percentage Error — how far off a forecast is, in percent; lower is better |
| LightGBM | A fast, tree-based machine learning algorithm, good with tabular (spreadsheet-like) data |
| TF-IDF | A way of turning text into numbers based on how distinctive each word is |
| Isolation Forest | An algorithm that finds unusual data points by how easily they can be separated from the rest |
| Prophet | A forecasting library (by Meta) built for business time series with seasonality |
| Saturation point | The value at which a signal is treated as "maximum risk" for scoring purposes |
| Smart import | This project's feature that auto-detects what a real company's column names mean |

---

## Section F — Presenting tips

- **Explain before you click**, not after — say what you expect to see, then show
  it. It reads as understanding, not narration.
- **If something loads slowly or looks off**, don't panic-explain — pause, let it
  finish, keep talking about the concept while it loads.
- **Volunteer the honest caveats yourself** (sentiment's F1, the proxy churn
  features, hand-picked weights). It reads as far more credible than a judge
  catching something you didn't mention.
- **You don't need every number memorized** — Section C exists so you can glance,
  not guess.
