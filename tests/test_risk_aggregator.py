"""Tests for risk aggregator."""

import numpy as np
import pandas as pd
import pytest

from src.risk_aggregator import (
    DEFAULT_WEIGHTS,
    assess_company,
    compute_risk_score,
    generate_explanation_text,
    risk_band,
)


def _churn_frame(probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"churn_prob": probabilities})


def _sentiment_frame(scores: list[float], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"score": scores, "label": labels})


def _anomaly_frame(flags: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-12-01", periods=len(flags), freq="D"),
            "is_anomaly": flags,
        }
    )


def _forecast_frame(actuals: list[float], forecasts: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"actual": actuals, "forecast": forecasts})


def test_risk_score_is_bounded_and_monotonic_in_churn() -> None:
    """More churn must never lower the score, and the score stays in [0, 100]."""
    low = compute_risk_score(0.1, 0.9, 0.0, 0.0)
    high = compute_risk_score(0.9, 0.9, 0.0, 0.0)

    assert 0.0 <= low <= 100.0
    assert 0.0 <= high <= 100.0
    assert high > low


def test_risk_score_treats_positivity_as_inverse_risk() -> None:
    """Happy customers must reduce risk, unhappy ones must raise it."""
    happy = compute_risk_score(0.3, 1.0, 0.0, 0.0)
    unhappy = compute_risk_score(0.3, 0.0, 0.0, 0.0)

    assert unhappy > happy
    assert unhappy - happy == pytest.approx(DEFAULT_WEIGHTS["w_sentiment"] * 100, abs=1e-6)


def test_risk_score_never_nan_with_missing_or_invalid_inputs() -> None:
    """Partial data must degrade to sane defaults rather than producing NaN."""
    for score in (
        compute_risk_score(None, None, None, None),
        compute_risk_score(float("nan"), float("nan"), float("nan"), float("nan")),
        compute_risk_score("bad", "worse", None, float("inf")),
    ):
        assert np.isfinite(score)
        assert 0.0 <= score <= 100.0

    # Missing sentiment must not silently look like maximum unhappiness.
    assert compute_risk_score(0.0, None, 0.0, 0.0) == pytest.approx(0.0)


def test_risk_score_clips_out_of_range_signals() -> None:
    """Signals outside [0, 1] are clipped, not allowed to blow past 100."""
    assert compute_risk_score(5.0, -3.0, 9.0, 9.0) == pytest.approx(100.0)


def test_risk_band_thresholds() -> None:
    """Bands map to the documented colour coding."""
    assert risk_band(10.0) == ("Low", "green")
    assert risk_band(45.0) == ("Medium", "orange")
    assert risk_band(85.0) == ("High", "red")


def test_forecast_term_ignores_upside_but_counts_shortfall() -> None:
    """Beating the forecast is good news; missing it is risk."""
    beating = assess_company(
        "Beating",
        _churn_frame([0.2]),
        _sentiment_frame([0.8], ["positive"]),
        _anomaly_frame([False] * 30),
        _forecast_frame([120.0] * 5, [100.0] * 5),
    )
    missing = assess_company(
        "Missing",
        _churn_frame([0.2]),
        _sentiment_frame([0.8], ["positive"]),
        _anomaly_frame([False] * 30),
        _forecast_frame([80.0] * 5, [100.0] * 5),
    )

    assert beating["signals"]["forecast_deviation"] == pytest.approx(0.0)
    assert missing["signals"]["forecast_deviation"] > 0.0
    assert missing["risk_score"] > beating["risk_score"]


def test_sentiment_verdict_uses_score_not_modal_label() -> None:
    """A 'mostly positive by count but low scoring' set must read as negative."""
    sentiment = _sentiment_frame(
        [0.1, 0.1, 0.1, 0.2, 0.9],
        ["negative", "negative", "negative", "negative", "positive"],
    )
    result = assess_company(
        "ScoreDriven",
        _churn_frame([0.3]),
        sentiment,
        _anomaly_frame([False] * 30),
        _forecast_frame([100.0], [100.0]),
    )

    assert "sentiment is mostly negative" in result["explanation_text"]


def test_assess_company_returns_the_dashboard_contract() -> None:
    """The dashboard depends on these exact keys existing."""
    result = assess_company(
        "DemoCo",
        _churn_frame([0.7, 0.6, 0.8]),
        _sentiment_frame([0.2, 0.3], ["negative", "negative"]),
        _anomaly_frame([False] * 27 + [True] * 3),
        _forecast_frame([90.0] * 5, [100.0] * 5),
    )

    assert set(result) >= {
        "company_id",
        "risk_score",
        "risk_label",
        "risk_colour",
        "signals",
        "contributions",
        "top_factors",
        "explanation_text",
    }
    assert result["company_id"] == "DemoCo"
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["explanation_text"].endswith(".")
    assert sum(result["contributions"].values()) == pytest.approx(result["risk_score"], abs=1e-6)


def test_assess_company_handles_completely_empty_inputs() -> None:
    """Empty frames must not raise -- the dashboard renders new companies too."""
    empty = pd.DataFrame()
    result = assess_company("EmptyCo", empty, empty, empty, empty)

    assert np.isfinite(result["risk_score"])
    assert result["risk_score"] == pytest.approx(0.0)
    assert result["top_factors"] == []
    assert isinstance(result["explanation_text"], str)


def test_generate_explanation_text_mentions_every_signal() -> None:
    """The sentence must name the drivers, the sentiment, anomalies, and trend."""
    text = generate_explanation_text(
        {"top_factors": [("numeric__usage_frequency", -0.4), ("numeric__tenure", 0.2)]},
        sentiment_label="negative",
        anomaly_flags=[True, False, True],
        forecast_trend="12% below forecast",
    )

    assert "usage frequency" in text
    assert "negative" in text
    assert "2 anomalous days" in text
    assert "12% below forecast" in text
