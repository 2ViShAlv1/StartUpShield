# Demo Video Script — StartupShield AI

This script is written in plain English, on purpose. Read the primer first so you
actually understand what you're saying — then use the timed script while recording.

---

## Part 1 — Understand the project in 60 seconds

Read this once before anything else. This is the whole project in plain words.

**The problem:** A startup founder usually finds out their company is in trouble
*too late* — when the month's revenue report already looks bad. By then, the
customers who were going to leave have already left.

**The idea:** What if a dashboard could watch the warning signs *before* that
happens, and put them all in one place?

**The four warning signs this project watches:**

1. **Churn** — is a customer about to cancel? (Based on how long they've stayed,
   how much they spend, how often they use the product, and how many support
   tickets they've filed.)
2. **Sentiment** — are customers happy or angry? (Read from their reviews and
   support messages using a text-analysis model.)
3. **Anomalies** — did something weird happen to revenue or usage on a
   particular day? (An outage, a billing bug, a sudden drop.)
4. **Forecast** — where is revenue headed over the next 30 days? (Not just "is
   it bad now," but "is it about to get worse.")

**What ties it together:** Each of those four signals is its own machine learning
model. A fifth piece of code (`risk_aggregator.py`) combines all four into
**one Risk Score from 0 to 100**, explains *why* the score is what it is in plain
English, and suggests 2–3 things to actually do about it.

**The newest feature:** Originally the dashboard only worked for 3 made-up demo
companies. Now there's an **Upload Your Company** page — anyone can upload their
own customer list, reviews, and revenue numbers (even a messy raw export from
Stripe or Zendesk with different column names) and get a real score in seconds.
That's the difference between "a cool demo" and "a tool someone could actually
use."

That's it. That's the whole project. Everything below is just showing this off.

---

## Part 2 — Before you record

- [ ] Run `streamlit run app/app.py` and confirm it opens at `http://localhost:8501`
- [ ] Have `sample_upload/customers.csv`, `reviews.csv`, `daily_metrics.csv` ready
      in a folder you can quickly find in the file picker
- [ ] Close other browser tabs / notifications (nothing distracting on screen)
- [ ] Do one silent practice run through the clicks before you record audio
- [ ] Decide your video length. This script is ~3.5 minutes read at a normal pace.
      Segments marked **[OPTIONAL]** can be cut if you need it shorter.

---

## Part 3 — The Script

Each row: **what to click/show** on the left, **what to say** on the right.
Say it in your own words once you understand it — you don't have to memorize it
word for word, the primer above is what actually matters.

### 🎬 0:00–0:15 — Hook

**Show:** Your face on camera, or the app's Overview page as a title screen.

> "Hi, I'm going to show you StartupShield AI — it's a dashboard that tells a
> startup founder if their company is in danger, before it's too late to fix it."

### 🎬 0:15–0:40 — The problem

**Show:** Stay on camera or a blank slide.

> "Right now, most founders find out something's wrong from a bad month's revenue
> report. But by that point, the customers who were unhappy have already left,
> and you're reading about it after the fact. There's no single place that watches
> your customers, your reviews, and your revenue together, and warns you early."

### 🎬 0:40–1:05 — The solution

**Show:** Stay on camera, or the Overview page.

> "StartupShield AI watches four things at once: how likely your customers are to
> cancel, what they're actually saying in reviews, whether your revenue did
> anything unusual recently, and where your revenue is headed next month. It
> combines all four into one Risk Score out of 100 — and tells you exactly why,
> and what to do about it."

### 🎬 1:05–1:40 — Live demo: the Overview page

**Show:** Open the app. Sidebar → select **RedFlag Analytics**.

> "Let me show you live. I'll pick a company from the sidebar — RedFlag Analytics.
> Right away I get a Risk Score of 57 out of 100 — Medium risk, shown in orange.
> Below that, in plain English, it tells me *why*: churn risk is high, sentiment
> is negative, there are anomalies in the revenue. And here are the top actions I
> should take right now."

**Show:** Switch sidebar → **GreenLeaf SaaS**.

> "If I switch to a healthier company — GreenLeaf SaaS — the score drops to 19,
> Low risk, green. Same model, same math, an honestly different answer."

### 🎬 1:40–2:05 — Live demo: the other pages **[OPTIONAL — trim if short on time]**

**Show:** Click through Churn → Sentiment → Anomalies → Forecast pages quickly.

> "Each number on that score has its own page behind it. Churn shows exactly
> which factors are driving risk up, and which customers are most likely to
> leave. Sentiment shows the mood in customer reviews over time. Anomalies flags
> specific days where something looked wrong — see this red dot? That's a day
> worth investigating. And Forecast predicts the next 30 days of revenue, with a
> confidence range, so I know not just where I am, but where I'm heading."

### 🎬 2:05–2:50 — Live demo: Upload Your Company (the highlight)

**Show:** Click **Upload Your Company** in the sidebar. Upload the three files
from `sample_upload/`. Click **Score my company**.

> "But here's the real point of this project. Every company I just showed you is
> a demo. This page — Upload Your Company — lets *any* business upload their own
> customer list, reviews, and daily revenue, and get a real score in seconds. I'll
> upload a sample company now... and in a few seconds, it scores this brand-new
> company — one the model has never seen before — at 65 out of 100, High risk,
> and tells me exactly why: churn, negative sentiment, and revenue heading down
> are all contributing.
>
> And you don't even have to clean up your file first — if I upload a raw export
> straight from something like Stripe, with completely different column names,
> the app automatically figures out which column means what. That's the
> difference between a demo and something a real founder could actually use."

### 🎬 2:50–3:15 — Why it's technically solid

**Show:** Back to camera, or the Risk & Recommendations page.

> "Under the hood, this is four real machine learning models, not one gimmick.
> The churn model is 85% accurate at telling risky customers apart from safe
> ones. The anomaly detector catches every single unusual day in testing. The
> forecasting model beats a simple guess by up to 13 percentage points. And every
> prediction comes with a plain-English explanation — this isn't a black box that
> just spits out a number."

### 🎬 3:15–3:30 — Close

**Show:** Back to camera.

> "StartupShield AI turns four separate warning signs into one score, one
> explanation, and one next step — so a founder finds out they're at risk from a
> dashboard, not from a bad quarter. Thanks for watching."

---

## Part 4 — If judges ask you a question

You don't need to memorize these — just read them once so the *idea* is in your
head, then answer in your own words.

**"What if I don't have all this data — reviews, daily revenue, everything?"**
> "Only the customer list is required. Reviews and daily revenue are optional —
> the score still works without them, it just uses fewer signals and says so on
> screen. Nothing is forced or faked."

**"How accurate are the models, really?"**
> "The churn model is 85% ROC-AUC, which is solidly good for this kind of
> problem. The forecasting model beats a plain guess by 5 to 13 percentage
> points across the demo companies. The three demo companies are synthetic —
> made up for the demo — but the models and the scoring math are real, and the
> Upload page proves that by scoring on data the model has genuinely never seen."

**"What technology did you use?"**
> "Python end to end. Scikit-learn and LightGBM for churn prediction, a
> TF-IDF text model for sentiment, Isolation Forest for anomaly detection,
> Facebook's Prophet for forecasting, SHAP for explaining the churn predictions,
> and Streamlit for the dashboard itself."

**"Can this actually scale to real companies?"**
> "That's exactly what the Upload page and the smart column-detection were built
> for — a real company's messy export, not just our three demo companies. The
> next real step would be direct integrations, so it connects straight to Stripe
> or Zendesk instead of needing a CSV upload at all."

**"What's the business model / what's next?"**
> "Right now it's a single-user tool you upload data into. The natural next step
> is a version that connects directly to a company's billing and support tools,
> refreshes automatically, and could be sold as a subscription to startups and
> their investors as an early-warning system."

---

## Part 5 — Quick glossary (only if a word trips you up)

| Term | Plain meaning |
| --- | --- |
| **Churn** | A customer cancelling / leaving |
| **Sentiment analysis** | Reading text (reviews) to tell if it's positive or negative |
| **Anomaly detection** | Spotting a day where a number looked unusual compared to normal |
| **Forecasting** | Predicting future numbers (like next month's revenue) |
| **SHAP** | A method that explains *why* a model made a specific prediction |
| **Risk Score** | The single 0–100 number this project outputs, combining all four signals |
| **ROC-AUC** | A 0–1 score for how well a model tells two groups apart (churn vs. not); higher is better |
