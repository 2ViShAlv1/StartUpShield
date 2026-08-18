"""Forecasting module for StartupShield AI.

Forecasts a daily business series (revenue by default) with confidence bounds,
and scores the result against a last-value-carried-forward naive baseline.

Primary model is Prophet. If `prophet` is not installed, the module falls back
to a statsmodels Holt-Winters (ETS) model, and finally to a seasonal-naive
model if statsmodels is unavailable too. The resolved choice is always readable
from `ForecastModel.resolved_model_type_`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/forecast_model.pkl")
DATE_COLUMN = "date"
DEFAULT_TARGET_COLUMN = "revenue"
DEFAULT_HORIZON_DAYS = 30
SEASONAL_PERIOD_DAYS = 7
# 95% interval under a normal residual assumption (used by the ETS/naive paths).
INTERVAL_Z_SCORE = 1.96


class ForecastModel:
    """Fitted forecaster plus everything needed to predict and score it.

    Attributes:
        resolved_model_type_: Which backend actually trained (`prophet`,
            `ets`, or `seasonal_naive`) after fallbacks were applied.
        target_column: Name of the forecast series.
        horizon_days: Default forecast horizon.
        last_train_date: Last date seen during training.
        last_train_value: Final observed value, used by the naive baseline.
        residual_std_: Training residual standard deviation, used to widen
            confidence bounds for the non-Prophet backends.
    """

    def __init__(
        self,
        backend: Any,
        resolved_model_type: str,
        target_column: str,
        horizon_days: int,
        train_df: pd.DataFrame,
        residual_std: float,
        seasonal_tail: np.ndarray | None = None,
    ) -> None:
        self.backend = backend
        self.resolved_model_type_ = resolved_model_type
        self.target_column = target_column
        self.horizon_days = horizon_days
        self.last_train_date = pd.Timestamp(train_df[DATE_COLUMN].iloc[-1])
        self.last_train_value = float(train_df[target_column].iloc[-1])
        self.train_rows_ = len(train_df)
        self.residual_std_ = float(residual_std)
        self.seasonal_tail_ = seasonal_tail

    def future_dates(self, horizon_days: int) -> pd.DatetimeIndex:
        """Return the future date index following the training window."""
        return pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )


def chronological_split(
    df: pd.DataFrame,
    test_days: int = DEFAULT_HORIZON_DAYS,
    date_column: str = DATE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series by time, never by random shuffle.

    Args:
        df: Time-series DataFrame for a single company.
        test_days: Number of trailing days held out for testing.
        date_column: Name of the date column.

    Returns:
        (train_df, test_df), both sorted ascending by date.

    Raises:
        ValueError: If the frame is too short to leave any training data.
    """
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column!r}")
    if test_days <= 0:
        raise ValueError("test_days must be positive")
    if len(df) <= test_days:
        raise ValueError(
            f"Need more than {test_days} rows to split; got {len(df)}"
        )

    sorted_df = df.copy()
    sorted_df[date_column] = pd.to_datetime(sorted_df[date_column])
    sorted_df = sorted_df.sort_values(date_column).reset_index(drop=True)
    return sorted_df.iloc[:-test_days].copy(), sorted_df.iloc[-test_days:].copy()


def train(
    df: pd.DataFrame,
    model_type: str = "prophet",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    seed: int = 42,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> ForecastModel:
    """Fit a forecaster on a single company's daily series.

    Args:
        df: Training rows with a date column and the target column.
        model_type: `prophet`, `ets`, `seasonal_naive`, or `auto`.
        horizon_days: Default horizon stored on the model.
        seed: Random seed (kept for interface parity and reproducibility).
        target_column: Series to forecast.

    Returns:
        Fitted ForecastModel. Check `resolved_model_type_` for the backend that
        was actually used after fallbacks.
    """
    for required_column in (DATE_COLUMN, target_column):
        if required_column not in df.columns:
            raise ValueError(f"Missing forecast column: {required_column!r}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    model_type = model_type.lower()
    if model_type not in {"prophet", "ets", "seasonal_naive", "auto"}:
        raise ValueError(
            "model_type must be one of: prophet, ets, seasonal_naive, auto"
        )

    train_df = df.copy()
    train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN])
    train_df = train_df.sort_values(DATE_COLUMN).reset_index(drop=True)

    np.random.seed(seed)
    seasonal_tail = _seasonal_tail(train_df[target_column].to_numpy(dtype=float))

    if model_type in {"prophet", "auto"}:
        built = _build_prophet(train_df, target_column)
        if built is not None:
            backend, residual_std = built
            return ForecastModel(
                backend, "prophet", target_column, horizon_days, train_df,
                residual_std, seasonal_tail,
            )
        logger.warning("prophet unavailable; falling back to ETS")

    if model_type in {"prophet", "auto", "ets"}:
        built = _build_ets(train_df, target_column)
        if built is not None:
            backend, residual_std = built
            return ForecastModel(
                backend, "ets", target_column, horizon_days, train_df,
                residual_std, seasonal_tail,
            )
        logger.warning("statsmodels unavailable; falling back to seasonal naive")

    residual_std = _seasonal_naive_residual_std(
        train_df[target_column].to_numpy(dtype=float)
    )
    return ForecastModel(
        None, "seasonal_naive", target_column, horizon_days, train_df,
        residual_std, seasonal_tail,
    )


def predict(model: ForecastModel, horizon_days: int | None = None) -> pd.DataFrame:
    """Forecast future values with confidence bounds.

    Args:
        model: Fitted ForecastModel.
        horizon_days: Days to forecast; defaults to the model's horizon.

    Returns:
        DataFrame with columns `date`, `forecast`, `lower_bound`, `upper_bound`.
        Bounds are present for every future date (FR4 requires them).
    """
    horizon_days = horizon_days or model.horizon_days
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    future_dates = model.future_dates(horizon_days)

    if model.resolved_model_type_ == "prophet":
        forecast_df = _predict_prophet(model, future_dates)
    elif model.resolved_model_type_ == "ets":
        forecast_df = _predict_ets(model, future_dates, horizon_days)
    else:
        forecast_df = _predict_seasonal_naive(model, future_dates, horizon_days)

    # Guarantee the FR4 contract even if a backend returns an inverted interval.
    forecast_df["lower_bound"] = np.minimum(
        forecast_df["lower_bound"], forecast_df["forecast"]
    )
    forecast_df["upper_bound"] = np.maximum(
        forecast_df["upper_bound"], forecast_df["forecast"]
    )
    return forecast_df.reset_index(drop=True)


def naive_baseline_forecast(model: ForecastModel, horizon_days: int) -> np.ndarray:
    """Last-value-carried-forward baseline, the bar the model must beat."""
    return np.full(horizon_days, model.last_train_value, dtype=float)


def evaluate(model: ForecastModel, test_df: pd.DataFrame) -> dict:
    """Score the forecast against held-out actuals and the naive baseline.

    Args:
        model: Fitted ForecastModel.
        test_df: Held-out rows with the date and target columns.

    Returns:
        Dictionary with `mae`, `rmse`, `mape`, `coverage`, `model_type`, and a
        `vs_naive_baseline` block containing the baseline metrics, the MAPE
        improvement in percentage points, and a `beats_naive` flag.
    """
    if model.target_column not in test_df.columns:
        raise ValueError(f"Missing target column: {model.target_column!r}")

    actuals = test_df[model.target_column].to_numpy(dtype=float)
    horizon_days = len(actuals)
    forecast_df = predict(model, horizon_days)
    predictions = forecast_df["forecast"].to_numpy(dtype=float)
    baseline = naive_baseline_forecast(model, horizon_days)

    model_metrics = _regression_metrics(actuals, predictions)
    baseline_metrics = _regression_metrics(actuals, baseline)
    within_bounds = (
        (actuals >= forecast_df["lower_bound"].to_numpy(dtype=float))
        & (actuals <= forecast_df["upper_bound"].to_numpy(dtype=float))
    )

    return {
        **model_metrics,
        "coverage": float(within_bounds.mean()),
        "model_type": model.resolved_model_type_,
        "horizon_days": horizon_days,
        "vs_naive_baseline": {
            **baseline_metrics,
            "mape_improvement_pp": float(
                baseline_metrics["mape"] - model_metrics["mape"]
            ),
            "beats_naive": bool(model_metrics["mape"] < baseline_metrics["mape"]),
        },
    }


def save_model(model: ForecastModel, path: str | Path = MODEL_PATH) -> None:
    """Persist a fitted forecaster.

    Prophet objects do not pickle reliably across environments, so the Prophet
    backend is stored via its own JSON serializer and rebuilt on load.
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "resolved_model_type": model.resolved_model_type_,
        "target_column": model.target_column,
        "horizon_days": model.horizon_days,
        "last_train_date": model.last_train_date,
        "last_train_value": model.last_train_value,
        "train_rows": model.train_rows_,
        "residual_std": model.residual_std_,
        "seasonal_tail": model.seasonal_tail_,
        "backend": None,
        "prophet_json": None,
    }

    if model.resolved_model_type_ == "prophet":
        from prophet.serialize import model_to_json

        payload["prophet_json"] = model_to_json(model.backend)
    else:
        payload["backend"] = model.backend

    joblib.dump(payload, model_path)


def load_model(path: str | Path = MODEL_PATH) -> ForecastModel:
    """Load a forecaster saved by `save_model`."""
    payload = joblib.load(Path(path))

    backend = payload["backend"]
    if payload["resolved_model_type"] == "prophet":
        from prophet.serialize import model_from_json

        backend = model_from_json(payload["prophet_json"])

    stub_df = pd.DataFrame(
        {
            DATE_COLUMN: [payload["last_train_date"]],
            payload["target_column"]: [payload["last_train_value"]],
        }
    )
    model = ForecastModel(
        backend,
        payload["resolved_model_type"],
        payload["target_column"],
        payload["horizon_days"],
        stub_df,
        payload["residual_std"],
        payload["seasonal_tail"],
    )
    model.train_rows_ = payload["train_rows"]
    return model


def _regression_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Return MAE, RMSE, and MAPE for one forecast."""
    errors = actuals - predictions
    # Guard against divide-by-zero; the business series is strictly positive.
    safe_actuals = np.where(np.abs(actuals) < 1e-9, np.nan, actuals)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.nanmean(np.abs(errors / safe_actuals)) * 100),
    }


def _seasonal_tail(values: np.ndarray) -> np.ndarray | None:
    """Return the final weekly cycle, used by the seasonal-naive backend."""
    if len(values) < SEASONAL_PERIOD_DAYS:
        return None
    return values[-SEASONAL_PERIOD_DAYS:].copy()


def _seasonal_naive_residual_std(values: np.ndarray) -> float:
    """Residual spread of a 7-day-lag seasonal naive fit on the training data."""
    if len(values) <= SEASONAL_PERIOD_DAYS:
        return float(np.std(values)) if len(values) else 0.0
    residuals = values[SEASONAL_PERIOD_DAYS:] - values[:-SEASONAL_PERIOD_DAYS]
    return float(np.std(residuals))


def _build_prophet(train_df: pd.DataFrame, target_column: str):
    """Fit Prophet when installed; return (model, residual_std) or None."""
    try:
        from prophet import Prophet
    except ImportError:
        return None

    prophet_df = pd.DataFrame(
        {
            "ds": train_df[DATE_COLUMN],
            "y": train_df[target_column].astype(float),
        }
    )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(prophet_df)

    fitted = model.predict(prophet_df[["ds"]])["yhat"].to_numpy(dtype=float)
    residual_std = float(np.std(prophet_df["y"].to_numpy(dtype=float) - fitted))
    return model, residual_std


def _build_ets(train_df: pd.DataFrame, target_column: str):
    """Fit Holt-Winters when statsmodels is installed; else return None."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None

    values = train_df[target_column].astype(float).to_numpy()
    # Additive seasonality needs at least two full weekly cycles.
    use_seasonal = len(values) >= 2 * SEASONAL_PERIOD_DAYS
    fitted_model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=SEASONAL_PERIOD_DAYS if use_seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)

    residual_std = float(np.std(values - fitted_model.fittedvalues))
    return fitted_model, residual_std


def _predict_prophet(model: ForecastModel, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Prophet supplies its own uncertainty interval."""
    forecast = model.backend.predict(pd.DataFrame({"ds": future_dates}))
    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": forecast["yhat"].to_numpy(dtype=float),
            "lower_bound": forecast["yhat_lower"].to_numpy(dtype=float),
            "upper_bound": forecast["yhat_upper"].to_numpy(dtype=float),
        }
    )


def _predict_ets(
    model: ForecastModel,
    future_dates: pd.DatetimeIndex,
    horizon_days: int,
) -> pd.DataFrame:
    """Holt-Winters point forecast with residual-based bounds.

    Holt-Winters has no native interval here, so bounds use the training
    residual spread widened by sqrt(step) to reflect growing uncertainty.
    """
    point_forecast = np.asarray(model.backend.forecast(horizon_days), dtype=float)
    step_scale = np.sqrt(np.arange(1, horizon_days + 1))
    margin = INTERVAL_Z_SCORE * model.residual_std_ * step_scale
    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": point_forecast,
            "lower_bound": point_forecast - margin,
            "upper_bound": point_forecast + margin,
        }
    )


def _predict_seasonal_naive(
    model: ForecastModel,
    future_dates: pd.DatetimeIndex,
    horizon_days: int,
) -> pd.DataFrame:
    """Repeat the last observed weekly cycle when no library is available."""
    if model.seasonal_tail_ is None:
        point_forecast = np.full(horizon_days, model.last_train_value, dtype=float)
    else:
        repeats = int(np.ceil(horizon_days / SEASONAL_PERIOD_DAYS))
        point_forecast = np.tile(model.seasonal_tail_, repeats)[:horizon_days]

    step_scale = np.sqrt(np.arange(1, horizon_days + 1))
    margin = INTERVAL_Z_SCORE * model.residual_std_ * step_scale
    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": point_forecast,
            "lower_bound": point_forecast - margin,
            "upper_bound": point_forecast + margin,
        }
    )
