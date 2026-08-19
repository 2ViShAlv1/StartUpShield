"""Tests for the uploaded-company scoring path.

This is the onboarding flow: a company the models have never seen uploads its own
data and gets a live score. The cases that matter are the ones a real signup hits --
unseen categorical values, missing optional files, and short histories.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from src import pipeline


@pytest.fixture(autouse=True)
def _quiet_third_party_logs():
    """Prophet fits print progress; keep test output readable."""
    warnings.filterwarnings("ignore")
    for logger_name in ("cmdstanpy", "prophet", "matplotlib"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


@pytest.fixture(scope="module")
def models():
    """Load the trained models once for the whole module."""
    return pipeline.load_models()


def _customers(n_rows: int = 40, plan_types=("starter", "growth")) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "customer_id": [f"C{index}" for index in range(n_rows)],
            "tenure": rng.integers(1, 36, n_rows),
            "monthly_spend": rng.normal(70, 15, n_rows).round(2),
            "usage_frequency": rng.integers(0, 30, n_rows),
            "support_tickets": rng.poisson(0.3, n_rows),
            "plan_type": rng.choice(list(plan_types), n_rows),
        }
    )


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_text": ["Love the product, support is great"] * 8
            + ["Buggy and slow, we are cancelling"] * 6
        }
    )


def _timeseries(n_days: int = 75) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    revenue = 5000 + np.arange(n_days) * 12 + rng.normal(0, 150, n_days)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-06-01", periods=n_days).astype(str),
            "revenue": revenue.round(2),
            "active_users": (revenue / 20).round().astype(int),
            "signups": rng.integers(5, 20, n_days),
        }
    )


def test_scores_a_company_the_models_have_never_seen(models) -> None:
    """The whole point of onboarding: unseen company, unseen plan names, still works."""
    result = pipeline.score_uploaded_company(
        "BrandNewCo", _customers(), _reviews(), _timeseries(), models=models
    )

    assessment = result["assessment"]
    assert 0.0 <= assessment["risk_score"] <= 100.0
    assert assessment["risk_label"] in {"Low", "Medium", "High"}
    assert assessment["company_id"] == "BrandNewCo"
    assert result["recommendations"]
    assert assessment["explanation_text"].endswith(".")


def test_unseen_plan_type_values_do_not_break_the_churn_model(models) -> None:
    """A new company's plan names won't match the training set's -- that's expected.

    The churn pipeline one-hot encodes with handle_unknown="ignore", so this must
    produce real probabilities rather than raising.
    """
    exotic = _customers(plan_types=("platinum", "unobtainium"))
    result = pipeline.score_uploaded_company(
        "ExoticPlans", exotic, _reviews(), _timeseries(), models=models
    )

    churn_probabilities = result["customers"]["churn_prob"]
    assert churn_probabilities.notna().all()
    assert churn_probabilities.between(0.0, 1.0).all()


def test_reviews_and_timeseries_are_optional(models) -> None:
    """Customer data alone must still produce a score -- most signups start there."""
    result = pipeline.score_uploaded_company(
        "CustomersOnly",
        _customers(),
        pd.DataFrame(),
        pd.DataFrame(),
        models=models,
    )

    assessment = result["assessment"]
    assert np.isfinite(assessment["risk_score"])
    # Missing signals must read as neutral, not as maximum risk.
    assert assessment["signals"]["anomaly_flag_ratio"] == 0.0
    assert assessment["signals"]["forecast_deviation"] == 0.0
    assert result["data_quality"]["reviews"] == 0


def test_short_history_skips_forecasting_but_still_scores(models) -> None:
    """Under 30 days can't be forecast; the score must survive and say why."""
    result = pipeline.score_uploaded_company(
        "ShortHistoryCo", _customers(), _reviews(), _timeseries(n_days=20), models=models
    )

    assert result["forecast"].empty
    assert np.isfinite(result["assessment"]["risk_score"])
    assert "30" in result["data_quality"]["forecast_note"]
    # Anomaly detection has a lower bar and should still have run.
    assert not result["anomalies"].empty


def test_forecast_fits_fresh_on_the_uploaded_series(models) -> None:
    """Forecasting must use the uploaded company's own data, not a demo model."""
    result = pipeline.score_uploaded_company(
        "FreshFitCo", _customers(), _reviews(), _timeseries(), models=models
    )

    forecast = result["forecast"]
    assert not forecast.empty
    assert forecast["is_projection"].any(), "expected a forward projection"
    assert (~forecast["is_projection"]).any(), "expected a backtest half"
    # Every row carries a usable confidence band (FR4).
    assert (forecast["lower_bound"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper_bound"]).all()


def test_invalid_customer_data_raises_before_scoring(models) -> None:
    """Bad input must fail loudly with a readable message, not a silent wrong score."""
    with pytest.raises(ValueError, match="Invalid customer data"):
        pipeline.score_uploaded_company(
            "BadCo",
            pd.DataFrame({"tenure": [1, 2, 3]}),
            _reviews(),
            _timeseries(),
            models=models,
        )


def test_result_shape_matches_the_demo_path(models) -> None:
    """Upload results must render in the same UI as demo companies."""
    uploaded = pipeline.score_uploaded_company(
        "ShapeCheck", _customers(), _reviews(), _timeseries(), models=models
    )
    demo = pipeline.score_company("MixedCo", models)

    shared_keys = set(demo) - {"profile"}
    assert shared_keys.issubset(set(uploaded))
    assert set(uploaded["assessment"]) == set(demo["assessment"])
