"""Tests for recommendation engine."""

import pandas as pd

from src.recommendation_engine import PRIORITY_ORDER, generate_recommendations


def _assessment(
    churn_prob: float = 0.1,
    sentiment_positivity: float = 0.9,
    anomaly_flag_ratio: float = 0.0,
    forecast_deviation: float = 0.0,
) -> dict:
    return {
        "signals": {
            "churn_prob": churn_prob,
            "sentiment_positivity": sentiment_positivity,
            "anomaly_flag_ratio": anomaly_flag_ratio,
            "forecast_deviation": forecast_deviation,
        }
    }


def _anomaly_frame(dropped_metric: str | None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-12-01", periods=30, freq="D"),
            "revenue": [1000.0] * 30,
            "active_users": [500.0] * 30,
            "signups": [50.0] * 30,
            "is_anomaly": [False] * 28 + [True, True],
        }
    )
    if dropped_metric:
        frame.loc[28:, dropped_metric] = frame[dropped_metric].iloc[0] * 0.4
    return frame


def test_healthy_company_gets_a_single_no_action_recommendation() -> None:
    """A company with all-clear signals should not be handed busywork."""
    recommendations = generate_recommendations(_assessment())

    assert len(recommendations) == 1
    assert recommendations[0]["priority"] == "info"
    assert "Healthy" in recommendations[0]["action"]


def test_high_churn_with_negative_sentiment_is_critical() -> None:
    """The compound at-risk case must surface as the top critical action."""
    recommendations = generate_recommendations(
        _assessment(churn_prob=0.8, sentiment_positivity=0.2)
    )

    assert recommendations[0]["priority"] == "critical"
    assert "retention outreach" in recommendations[0]["action"]
    assert "80%" in recommendations[0]["reason"]


def test_declining_revenue_plus_churn_is_critical() -> None:
    """Compounding revenue decline and churn must escalate."""
    recommendations = generate_recommendations(
        _assessment(churn_prob=0.5, sentiment_positivity=0.7, forecast_deviation=0.2)
    )

    actions = " ".join(item["action"] for item in recommendations)
    assert any(item["priority"] == "critical" for item in recommendations)
    assert "pricing and retention review" in actions


def test_recommendations_are_sorted_by_priority() -> None:
    """Most urgent first -- the dashboard shows only the top three."""
    recommendations = generate_recommendations(
        _assessment(
            churn_prob=0.85,
            sentiment_positivity=0.2,
            anomaly_flag_ratio=0.1,
            forecast_deviation=0.25,
        )
    )

    ranks = [PRIORITY_ORDER[item["priority"]] for item in recommendations]
    assert ranks == sorted(ranks)
    assert len(recommendations) >= 3


def test_anomaly_recommendation_names_the_metric_that_dropped() -> None:
    """Saying which metric moved is the difference between useful and generic."""
    revenue_drop = generate_recommendations(
        _assessment(anomaly_flag_ratio=0.07), _anomaly_frame("revenue")
    )
    users_drop = generate_recommendations(
        _assessment(anomaly_flag_ratio=0.07), _anomaly_frame("active_users")
    )

    assert any("billing" in item["action"] for item in revenue_drop)
    assert any("outage" in item["action"] for item in users_drop)


def test_anomaly_without_detail_falls_back_to_generic_review() -> None:
    """Anomalies with no identifiable dropped metric still get an action."""
    recommendations = generate_recommendations(
        _assessment(anomaly_flag_ratio=0.07), _anomaly_frame(None)
    )

    assert any("flagged anomalous days" in item["action"] for item in recommendations)


def test_every_recommendation_has_the_full_contract() -> None:
    """The dashboard renders all three fields for every item."""
    recommendations = generate_recommendations(
        _assessment(churn_prob=0.6, sentiment_positivity=0.3, anomaly_flag_ratio=0.1)
    )

    for recommendation in recommendations:
        assert set(recommendation) == {"action", "reason", "priority"}
        assert recommendation["priority"] in PRIORITY_ORDER
        assert recommendation["action"].strip()
        assert recommendation["reason"].strip()


def test_missing_signals_do_not_crash_the_engine() -> None:
    """A partially-scored company must still produce a usable list."""
    recommendations = generate_recommendations({"signals": {}})

    assert len(recommendations) >= 1
    assert all("action" in item for item in recommendations)
