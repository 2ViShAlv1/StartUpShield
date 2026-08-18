"""Tests for forecast module."""

import numpy as np
import pandas as pd
import pytest

from src.forecast_module import (
    DEFAULT_HORIZON_DAYS,
    chronological_split,
    evaluate,
    load_model,
    naive_baseline_forecast,
    predict,
    save_model,
    train,
)
from src.generate_synthetic_data import generate_company_timeseries


# Prophet fits are slow; the ETS backend exercises the same public contract.
BACKEND = "ets"


def _sample_timeseries(n_days: int = 240, seed: int = 21) -> pd.DataFrame:
    """Return a single company's daily series with no injected anomalies."""
    return generate_company_timeseries(
        company_name="ForecastCo",
        start_date="2025-01-01",
        n_days=n_days,
        base_revenue=12_000,
        growth_rate=0.001,
        n_anomalies=0,
        anomaly_magnitude=0,
        seed=seed,
    )


def test_chronological_split_keeps_time_order_and_does_not_shuffle() -> None:
    """Split must be by time, and train must end before test begins."""
    df = _sample_timeseries()

    train_df, test_df = chronological_split(df, test_days=30)

    assert len(test_df) == 30
    assert len(train_df) == len(df) - 30
    assert train_df["date"].max() < test_df["date"].min()
    assert train_df["date"].is_monotonic_increasing
    assert test_df["date"].is_monotonic_increasing


def test_chronological_split_rejects_too_short_series() -> None:
    """A series shorter than the test window cannot be split."""
    df = _sample_timeseries(n_days=20)

    with pytest.raises(ValueError, match="Need more than"):
        chronological_split(df, test_days=30)


def test_predict_returns_confidence_bounds_for_every_future_date() -> None:
    """FR4: every forecast row must carry a usable confidence interval."""
    train_df, _ = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type=BACKEND, horizon_days=30)

    forecast_df = predict(model, horizon_days=30)

    assert list(forecast_df.columns) == ["date", "forecast", "lower_bound", "upper_bound"]
    assert len(forecast_df) == 30
    assert not forecast_df.isna().any().any()
    assert (forecast_df["lower_bound"] <= forecast_df["forecast"]).all()
    assert (forecast_df["forecast"] <= forecast_df["upper_bound"]).all()
    # Forecast must start the day after training ends and stay daily.
    assert forecast_df["date"].iloc[0] == train_df["date"].max() + pd.Timedelta(days=1)
    assert (forecast_df["date"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_predict_defaults_to_the_models_configured_horizon() -> None:
    """Calling predict() without a horizon uses the trained horizon."""
    train_df, _ = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type=BACKEND, horizon_days=DEFAULT_HORIZON_DAYS)

    assert len(predict(model)) == DEFAULT_HORIZON_DAYS


def test_evaluate_beats_naive_baseline_on_a_trending_series() -> None:
    """Phase 6 definition of done: the model must beat last-value-carried-forward."""
    train_df, test_df = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type=BACKEND, horizon_days=30)

    metrics = evaluate(model, test_df)

    assert set(metrics) >= {"mae", "rmse", "mape", "coverage", "vs_naive_baseline"}
    assert metrics["mape"] < metrics["vs_naive_baseline"]["mape"]
    assert metrics["vs_naive_baseline"]["beats_naive"] is True
    assert metrics["vs_naive_baseline"]["mape_improvement_pp"] > 0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["mae"] > 0 and metrics["rmse"] >= metrics["mae"]


def test_naive_baseline_carries_the_last_training_value_forward() -> None:
    """The baseline is a flat line at the final observed value."""
    train_df, _ = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type=BACKEND, horizon_days=30)

    baseline = naive_baseline_forecast(model, 30)

    assert baseline.shape == (30,)
    assert np.allclose(baseline, train_df["revenue"].iloc[-1])


def test_save_and_load_forecast_model_preserves_predictions(tmp_path) -> None:
    """A reloaded forecaster must produce identical output."""
    train_df, _ = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type=BACKEND, horizon_days=30)
    path = tmp_path / "forecast_model.pkl"

    save_model(model, path)
    loaded_model = load_model(path)

    original = predict(model, horizon_days=30)
    reloaded = predict(loaded_model, horizon_days=30)

    assert loaded_model.resolved_model_type_ == model.resolved_model_type_
    assert np.allclose(original["forecast"], reloaded["forecast"])
    assert np.allclose(original["lower_bound"], reloaded["lower_bound"])
    assert original["date"].equals(reloaded["date"])


def test_train_rejects_unknown_model_type_and_missing_columns() -> None:
    """Bad inputs should fail loudly rather than silently degrade."""
    train_df, _ = chronological_split(_sample_timeseries(), test_days=30)

    with pytest.raises(ValueError, match="model_type must be one of"):
        train(train_df, model_type="lstm")

    with pytest.raises(ValueError, match="Missing forecast column"):
        train(train_df.drop(columns=["revenue"]), model_type=BACKEND)

    with pytest.raises(ValueError, match="horizon_days must be positive"):
        train(train_df, model_type=BACKEND, horizon_days=0)


def test_seasonal_naive_backend_repeats_the_last_weekly_cycle() -> None:
    """The dependency-free fallback must still honour the predict contract."""
    train_df, test_df = chronological_split(_sample_timeseries(), test_days=30)
    model = train(train_df, model_type="seasonal_naive", horizon_days=14)

    forecast_df = predict(model, horizon_days=14)
    metrics = evaluate(model, test_df)

    assert model.resolved_model_type_ == "seasonal_naive"
    assert len(forecast_df) == 14
    assert (forecast_df["lower_bound"] <= forecast_df["upper_bound"]).all()
    assert np.allclose(
        forecast_df["forecast"].to_numpy()[:7],
        train_df["revenue"].to_numpy()[-7:],
    )
    assert metrics["model_type"] == "seasonal_naive"
