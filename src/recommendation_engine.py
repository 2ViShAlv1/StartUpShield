"""Recommendation engine for StartupShield AI.

Maps risk signals to concrete actions using an explicit rule table. Rules are
deliberately transparent and inspectable rather than learned -- a founder acting on
a recommendation should be able to see exactly which condition triggered it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Thresholds that define "high"/"low" for each signal. Tuned for the demo data and
# kept in one place so they are easy to review and adjust.
HIGH_CHURN = 0.55
MODERATE_CHURN = 0.35
NEGATIVE_SENTIMENT = 0.45
ANOMALY_PRESENT = 0.03
FORECAST_GAP = 0.08

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def generate_recommendations(assessment: dict, anomaly_data: Any = None) -> list[dict]:
    """Turn a risk assessment into a prioritized list of recommended actions.

    Args:
        assessment: Output of `risk_aggregator.assess_company`.
        anomaly_data: Optional daily frame with `is_anomaly` plus metric columns,
            used to say *which* metric moved rather than just "an anomaly".

    Returns:
        List of {'action', 'reason', 'priority'} dicts, most urgent first.
        Always returns at least one item.
    """
    signals = assessment.get("signals", {})
    churn_prob = _as_float(signals.get("churn_prob"), 0.0)
    positivity = _as_float(signals.get("sentiment_positivity"), 1.0)
    anomaly_ratio = _as_float(signals.get("anomaly_flag_ratio"), 0.0)
    forecast_deviation = _as_float(signals.get("forecast_deviation"), 0.0)

    recommendations: list[dict] = []

    # --- Compound conditions first: they describe the most urgent situations. ---
    if churn_prob > HIGH_CHURN and positivity < NEGATIVE_SENTIMENT:
        recommendations.append(
            {
                "action": "Prioritize retention outreach for the at-risk segment and "
                "review recent negative feedback themes.",
                "reason": f"High churn risk ({churn_prob:.0%}) combined with negative "
                f"customer sentiment ({positivity:.0%} positivity).",
                "priority": "critical",
            }
        )

    if forecast_deviation > FORECAST_GAP and churn_prob > MODERATE_CHURN:
        recommendations.append(
            {
                "action": "Run an immediate pricing and retention review -- revenue "
                "decline is compounding with churn risk.",
                "reason": f"Revenue is {forecast_deviation:.0%} off forecast while churn "
                f"risk sits at {churn_prob:.0%}.",
                "priority": "critical",
            }
        )

    if churn_prob > HIGH_CHURN and not _has_priority(recommendations, "critical"):
        recommendations.append(
            {
                "action": "Launch a targeted win-back campaign for the highest-risk customers.",
                "reason": f"Mean churn probability across the customer base is {churn_prob:.0%}.",
                "priority": "high",
            }
        )
    elif MODERATE_CHURN < churn_prob <= HIGH_CHURN:
        recommendations.append(
            {
                "action": "Schedule check-in calls with low-usage accounts before they lapse.",
                "reason": f"Churn probability is elevated at {churn_prob:.0%}.",
                "priority": "medium",
            }
        )

    if positivity < NEGATIVE_SENTIMENT:
        recommendations.append(
            {
                "action": "Triage recent negative reviews and publish a response plan "
                "for the most common complaint themes.",
                "reason": f"Average review positivity is only {positivity:.0%}.",
                "priority": "high",
            }
        )

    if anomaly_ratio > ANOMALY_PRESENT:
        recommendations.extend(_anomaly_recommendations(anomaly_ratio, anomaly_data))

    if forecast_deviation > FORECAST_GAP and not _has_priority(recommendations, "critical"):
        recommendations.append(
            {
                "action": "Re-forecast the quarter and investigate why actuals are "
                "diverging from the model.",
                "reason": f"Actual revenue is deviating {forecast_deviation:.0%} from forecast.",
                "priority": "medium",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "action": "Healthy -- no immediate action required. Continue monitoring.",
                "reason": "All four risk signals are within normal ranges.",
                "priority": "info",
            }
        )

    recommendations.sort(key=lambda item: PRIORITY_ORDER.get(item["priority"], 99))
    return recommendations


def _anomaly_recommendations(anomaly_ratio: float, anomaly_data: Any) -> list[dict]:
    """Name the metric that actually moved, when the data allows it."""
    dropped_metrics = _recently_dropped_metrics(anomaly_data)

    metric_actions = {
        "revenue": (
            "Investigate the payment and billing pipeline for failed charges.",
            "Revenue showed an unusual drop on recently flagged days.",
        ),
        "active_users": (
            "Check for a product outage or a recent breaking release.",
            "Active users dropped sharply on recently flagged days.",
        ),
        "signups": (
            "Review the signup funnel and any recent marketing or pricing changes.",
            "Signups dropped sharply on recently flagged days.",
        ),
    }

    recommendations = [
        {"action": metric_actions[metric][0], "reason": metric_actions[metric][1], "priority": "high"}
        for metric in dropped_metrics
        if metric in metric_actions
    ]

    if not recommendations:
        recommendations.append(
            {
                "action": "Review flagged anomalous days against your release and "
                "campaign calendar.",
                "reason": f"{anomaly_ratio:.0%} of the last 30 days were flagged as anomalous.",
                "priority": "medium",
            }
        )
    return recommendations


def _recently_dropped_metrics(
    anomaly_data: Any,
    window_days: int = 30,
    drop_threshold: float = -0.10,
) -> list[str]:
    """Which tracked metrics fell on flagged days, versus the window average."""
    if not isinstance(anomaly_data, pd.DataFrame) or anomaly_data.empty:
        return []
    if "is_anomaly" not in anomaly_data.columns:
        return []

    recent = anomaly_data.tail(window_days)
    flagged = recent[recent["is_anomaly"].astype(bool)]
    if flagged.empty:
        return []

    dropped = []
    for metric in ("revenue", "active_users", "signups"):
        if metric not in recent.columns:
            continue
        window_mean = pd.to_numeric(recent[metric], errors="coerce").mean()
        flagged_mean = pd.to_numeric(flagged[metric], errors="coerce").mean()
        if not np.isfinite(window_mean) or not np.isfinite(flagged_mean) or window_mean == 0:
            continue
        if (flagged_mean - window_mean) / abs(window_mean) < drop_threshold:
            dropped.append(metric)
    return dropped


def _has_priority(recommendations: list[dict], priority: str) -> bool:
    """True when a recommendation of the given priority already exists."""
    return any(item["priority"] == priority for item in recommendations)


def _as_float(value: Any, default: float) -> float:
    """Coerce to float, falling back to a default for missing/invalid values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default
