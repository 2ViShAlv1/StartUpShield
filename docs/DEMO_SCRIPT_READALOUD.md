# StartupShield AI — Read-Aloud Demo Script

Read this straight through, top to bottom, in your own natural voice. It's
written to be spoken, not summarized — full sentences, no jargon you'd have to
stop and explain.

Lines starting with **>> SHOW** are not spoken — they tell you exactly what
should be visible on screen right before the paragraph under them. Everything
else, read as written, or close to it.

About 830 words — roughly **5.5 to 6 minutes** read at a normal, unhurried
pace. If you need it shorter, skip the paragraph marked *(optional)* (saves
about 20 seconds), or read a little faster than conversational pace, which is
normal for a screen-recorded demo.

*(For the deeper explanation of every model, the numbers behind each claim,
and answers to questions a judge might ask, see
`docs/DEMO_NARRATION_GUIDE.md` — this file is only the spoken part.)*

---

**>> SHOW:** Your face on camera, or the Overview page as a title frame — no
clicking yet.

Hi, today I want to show you StartupShield AI — a dashboard built around one
specific problem. Startup founders usually find out their company is in
trouble too late. They find out from a bad month's revenue report, and by
that point, the customers who were going to leave have already left. There
was never one place watching all the warning signs together. That's what
this project does.

StartupShield AI watches four different things about a company at the same
time. First, how likely each customer is to cancel, based on how long they've
stayed, how much they spend, how often they use the product, and how many
support tickets they've filed. Second, what customers are actually saying —
their reviews and support messages — read through a sentiment model that
tells us if the mood is getting better or worse. Third, whether anything
unusual happened to the company's daily revenue, a spike or a drop that
doesn't match its normal pattern. And fourth, where revenue is heading over
the next thirty days, with a real confidence range, not just a guess.

All four of those get combined into a single Risk Score, from zero to a
hundred. And just as important, the dashboard explains, in plain English,
why the score is what it is, and gives two or three specific things to
actually do about it. Let me show you this live.

**>> SHOW:** Open the app. It lands on the **Overview** page with
**GreenLeaf SaaS** selected by default in the sidebar. Point at the coloured
risk badge under the company name on the left.

Here's the dashboard, looking at a company called GreenLeaf SaaS right now.
Its Risk Score is nineteen out of a hundred — Low risk, shown in green. Let
me switch to a different company from the sidebar.

**>> SHOW:** Click the sidebar dropdown, select **RedFlag Analytics**. Point
at the badge again (now orange), then move to the **"Why this score"** text
on the right, then the small horizontal bar chart below it labelled *"Points
contributed to the score, by signal."*

This is RedFlag Analytics. The score jumps to fifty-seven — Medium risk,
shown in orange. Right below the score, it explains why in plain language:
churn risk is elevated, sentiment is trending negative, and there are
unusual days in the revenue data. And here's exactly how many points each of
the four signals contributed.

**>> SHOW:** Scroll down to the **"Recommended actions"** section — the three
cards, each with a coloured priority icon, a bold action, and a grey caption
underneath explaining the reason.

And right here are the top three recommended actions — these aren't generic
advice, they're generated directly from whichever signals are actually
driving the score up.

**>> SHOW:** Click **Churn** in the sidebar. Point at the **"Risk
distribution"** chart, then **"Top risk factors"** (the SHAP chart), then
scroll to the **"Highest-risk customers"** table. *(optional — skip this
whole paragraph if you're short on time)*

This page sits behind the churn part of that number. This chart explains
exactly which factors are pushing risk up across the customer base — it's
not a black box, every prediction can be explained. And this table lists the
specific customers most likely to cancel, ranked by risk, so it's something
a team could act on today.

**>> SHOW:** Click **Sentiment**. Point at **"Sentiment mix"**, then
**"Positivity trend"**, then scroll to **"Flagged negative reviews"**.

This page reads every review and support message the company has, and
scores each one as positive, neutral, or negative. Here's the overall mix
right now, and here's how that mood has been trending week by week. And
these are the specific negative reviews flagged for follow-up.

**>> SHOW:** Click **Anomalies**. Point at the metric dropdown at the top,
then the line chart — specifically the red dots on it — then scroll to the
**"Flagged days"** table below.

This is the daily revenue timeline, and every red dot marks a day the model
flagged as unusual, compared to this company's own normal pattern, not some
fixed number picked in advance. A day like this usually means an outage, a
billing issue, or a pricing change worth looking into.

**>> SHOW:** Click **Forecast**. Point at the chart — the solid backtest line
on the left, the shaded **95% confidence band** around the projection on the
right, under **"Forward projection."**

And this is a thirty-day forecast — a backtest on one side showing how
accurate the model has actually been, and a forward projection with a
shaded confidence band. So instead of only knowing where revenue is today,
we know roughly where it's heading, and how confident the model is about
that.

**>> SHOW:** Click **Upload Your Company** in the sidebar. Point briefly at
**"1. Download templates"** and **"2. Upload your files"** — the three file
boxes for Customers, Reviews, and Time-series CSVs.

But here's the real point of this whole project. Every company I've shown
you so far has been a demo. This page — Upload Your Company — is where it
becomes something anyone can actually use. Any business can upload their own
customer list, their own reviews, their own daily revenue, and get a real
score in seconds. Let me show you.

**>> SHOW:** Upload the three files from `sample_upload/`. When
**"3. Confirm what your columns mean"** appears, point at one or two of the
dropdowns showing a detected column with its confidence label. Click
**Score my company**.

I'll upload a sample company now. Notice it doesn't ask me to rename a
single column — even if I upload a raw export straight out of something
like Stripe, with completely different column names, and no tenure column
at all, the app automatically figures out what each column means, and even
calculates tenure from the signup date by itself.

**>> SHOW:** After scoring, point at the **"Results"** section — the risk
badge, the explanation text, the four metric tiles, then scroll to
**"Recommended actions."** Finally point at the **"Download this report
(JSON)"** button without necessarily clicking it.

And in just a few seconds — this is a brand-new company the model has never
seen before — it scores sixty-five out of a hundred, High risk, and tells me
exactly why: churn, negative sentiment, and a downward revenue trend are all
contributing. I can download this entire result as a report.

**>> SHOW:** Back to camera, or stay on the Results section.

Under the hood, this is four real machine learning models working together,
not one gimmick pretending to be smart. The churn model is about
eighty-five percent accurate at telling risky customers apart from safe
ones. The anomaly detector caught every single unusual day in testing. The
forecasting model beats a simple guess by up to thirteen percentage points.
And every single prediction, not just the final score, comes with a
plain-English explanation.

**>> SHOW:** Back to camera, or the Overview page.

That's StartupShield AI — four separate warning signs, turned into one
score, one explanation, and one clear next step. So a founder finds out
their company is at risk from a dashboard, not from a bad quarter. Thank you
for watching.
