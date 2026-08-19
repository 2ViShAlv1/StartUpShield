"""End-to-end scoring pipeline for StartupShield AI.

Loads the trained models once, runs all four of them for a demo company, and
returns a single bundle containing every number the dashboard renders.

Training happens in `src/train_all.py`. This module only ever does inference, which
is what keeps per-company scoring fast enough for an interactive dashboard.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    anomaly_module,
    churn_module,
    data_validation,
    demo_data,
    forecast_module,
    recommendation_engine,
    risk_aggregator,
    sentiment_module,
)
from src.generate_synthetic_data import CONFIG_PATH, load_config


MODELS_DIR = Path("models")
FORECAST_EVAL_DAYS = 30


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    """Load every trained model artifact.

    Returns:
        Dict of loaded models. Any model that fails to load is set to None rather
        than raising, so the dashboard can still render the parts that do work.
    """
    models: dict = {"churn": None, "sentiment": None, "anomaly": None, "forecast": {}}

    try:
        models["churn"] = churn_module.load_model(models_dir / "churn_model.pkl")
    except Exception:
        pass

    try:
        models["sentiment"] = sentiment_module.load_model(models_dir / "sentiment_model")
    except Exception:
        pass

    try:
        models["anomaly"] = anomaly_module.load_model(models_dir / "anomaly_model.pkl")
    except Exception:
        pass

    for company_name in demo_data.company_names():
        slug = "_".join(company_name.lower().split())
        try:
            models["forecast"][company_name] = forecast_module.load_model(
                models_dir / f"forecast_model_{slug}.pkl"
            )
        except Exception:
            models["forecast"][company_name] = None

    return models


def score_company(
    company_name: str,
    models: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Run all four models for one company and aggregate into a risk assessment.

    Returns a bundle with the assessment, recommendations, and the per-module
    frames the dashboard charts (customers, reviews, anomalies, forecast).
    """
    models = models if models is not None else load_models()
    config = config if config is not None else load_config(CONFIG_PATH)

    customers = _score_customers(company_name, models.get("churn"))
    reviews = _score_reviews(company_name, models.get("sentiment"))
    anomalies = _score_anomalies(company_name, models.get("anomaly"))
    forecast = _score_forecast(company_name, models.get("forecast", {}).get(company_name))

    assessment = risk_aggregator.assess_company(
        company_id=company_name,
        churn_data=customers,
        sentiment_data=reviews,
        anomaly_data=anomalies,
        forecast_data=forecast,
        weights=config.get("risk_weights"),
        churn_model=models.get("churn"),
    )
    recommendations = recommendation_engine.generate_recommendations(assessment, anomalies)

    return {
        "company_name": company_name,
        "profile": demo_data.company_profile(company_name),
        "assessment": assessment,
        "recommendations": recommendations,
        "customers": customers,
        "reviews": reviews,
        "anomalies": anomalies,
        "forecast": forecast,
    }


def _score_customers(company_name: str, churn_model) -> pd.DataFrame:
    """Attach churn probabilities to a company's customer list."""
    customers = demo_data.load_company_customers(company_name)
    if customers.empty:
        return customers

    features = churn_module.engineer_features(customers)
    scored = pd.concat([customers.reset_index(drop=True), features], axis=1)
    scored = scored.loc[:, ~scored.columns.duplicated()]

    if churn_model is None:
        scored["churn_prob"] = np.nan
        return scored

    scored["churn_prob"] = churn_module.predict(churn_model, features)
    return scored.sort_values("churn_prob", ascending=False).reset_index(drop=True)


def _score_reviews(company_name: str, sentiment_model) -> pd.DataFrame:
    """Attach predicted sentiment labels and positivity scores to reviews."""
    reviews = demo_data.load_company_reviews(company_name)
    if reviews.empty or sentiment_model is None:
        if not reviews.empty:
            reviews["label"] = reviews.get("sentiment_label", "unknown")
            reviews["score"] = np.nan
        return reviews

    predictions = sentiment_module.predict(sentiment_model, list(reviews["review_text"]))
    reviews["label"] = [item["label"] for item in predictions]
    reviews["score"] = [item["score"] for item in predictions]
    return reviews


def _score_anomalies(company_name: str, anomaly_model) -> pd.DataFrame:
    """Flag anomalous days in a company's daily business metrics."""
    timeseries = demo_data.load_company_timeseries(company_name)
    if timeseries.empty:
        return timeseries

    features = anomaly_module.build_features(timeseries)
    features["date"] = pd.to_datetime(features["date"])

    if anomaly_model is None:
        features["is_anomaly"] = False
        features["anomaly_score"] = np.nan
        return features

    labels, scores = anomaly_module.predict(anomaly_model, features)
    features["is_anomaly"] = labels == -1
    features["anomaly_score"] = scores
    return features


def _score_forecast(company_name: str, forecast_model) -> pd.DataFrame:
    """Return the forecast backtest plus a forward projection.

    The saved model was trained up to 30 days before the data ends, so predicting
    60 days yields two useful halves: the first 30 line up with held-out actuals
    (the backtest, which proves the model works), and the next 30 run past the end
    of the data (the forward projection, which is what a founder actually wants).
    Backtest rows carry an `actual`; projection rows leave it NaN.
    """
    timeseries = demo_data.load_company_timeseries(company_name)
    empty_columns = ["date", "actual", "forecast", "lower_bound", "upper_bound", "is_projection"]
    if timeseries.empty or forecast_model is None:
        return pd.DataFrame(columns=empty_columns)

    _, holdout = forecast_module.chronological_split(timeseries, test_days=FORECAST_EVAL_DAYS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = forecast_module.predict(forecast_model, FORECAST_EVAL_DAYS * 2)

    predictions = predictions.reset_index(drop=True)
    predictions["actual"] = np.nan
    predictions.loc[: FORECAST_EVAL_DAYS - 1, "actual"] = holdout["revenue"].to_numpy(dtype=float)
    predictions.loc[: FORECAST_EVAL_DAYS - 1, "date"] = pd.to_datetime(
        holdout["date"].to_numpy()
    )
    predictions["is_projection"] = predictions["actual"].isna()
    return predictions[empty_columns]


def score_uploaded_company(
    company_name: str,
    customers: pd.DataFrame,
    reviews: pd.DataFrame,
    timeseries: pd.DataFrame,
    models: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Score a company from freshly uploaded data -- the real onboarding path.

    Unlike `score_company()`, which reads pre-baked demo CSVs, this takes whatever
    a user just uploaded. Design split, and why:

      * Churn and sentiment reuse the globally trained models (inference only).
        These are customer/text classifiers -- a "will this customer churn" pattern
        learned on one company's history is a reasonable prior for another's, and
        retraining per company would need far more data than a new signup has.
      * Anomaly detection and forecasting are trained FRESH on this company's own
        time series, right here, synchronously. There is no meaningful pretrained
        prior for "what does this specific company's normal day look like" --
        every company's baseline is its own. This mirrors how the demo companies
        already work (one forecast model per company); the only difference is the
        fit happens live instead of ahead of time via `train_all.py`.

    Callers MUST validate with `data_validation` first -- this function assumes the
    required columns exist and raises a clear `ValueError` otherwise rather than
    silently producing a wrong score.

    Returns the same bundle shape as `score_company()`, plus a `data_quality` block
    describing what was actually usable (row counts, whether forecasting ran).
    """
    models = models if models is not None else load_models()
    config = config if config is not None else load_config(CONFIG_PATH)

    customer_errors = data_validation.validate_customers(customers)
    if customer_errors:
        raise ValueError("Invalid customer data: " + "; ".join(customer_errors))

    scored_customers = _score_uploaded_customers(customers, models.get("churn"))
    scored_reviews = _score_uploaded_reviews(reviews, models.get("sentiment"))
    anomalies, forecast, forecast_note = _score_uploaded_timeseries(
        timeseries, config
    )

    assessment = risk_aggregator.assess_company(
        company_id=company_name,
        churn_data=scored_customers,
        sentiment_data=scored_reviews,
        anomaly_data=anomalies,
        forecast_data=forecast,
        weights=config.get("risk_weights"),
        churn_model=models.get("churn"),
    )
    recommendations = recommendation_engine.generate_recommendations(assessment, anomalies)

    return {
        "company_name": company_name,
        "profile": "Uploaded",
        "assessment": assessment,
        "recommendations": recommendations,
        "customers": scored_customers,
        "reviews": scored_reviews,
        "anomalies": anomalies,
        "forecast": forecast,
        "data_quality": {
            "customers": len(customers),
            "reviews": len(reviews) if reviews is not None else 0,
            "timeseries_days": len(timeseries) if timeseries is not None else 0,
            "forecast_note": forecast_note,
        },
    }


def _score_uploaded_customers(customers: pd.DataFrame, churn_model) -> pd.DataFrame:
    """Engineer features and predict churn for an uploaded customer table."""
    features = churn_module.engineer_features(customers)
    scored = pd.concat([customers.reset_index(drop=True), features], axis=1)
    scored = scored.loc[:, ~scored.columns.duplicated()]

    if churn_model is None:
        scored["churn_prob"] = np.nan
        return scored

    scored["churn_prob"] = churn_module.predict(churn_model, features)
    return scored.sort_values("churn_prob", ascending=False).reset_index(drop=True)


def _score_uploaded_reviews(reviews: pd.DataFrame | None, sentiment_model) -> pd.DataFrame:
    """Predict sentiment for an uploaded review table. Optional -- may be empty."""
    if reviews is None or reviews.empty:
        return pd.DataFrame(columns=["review_text", "label", "score"])

    reviews = reviews.copy()
    if sentiment_model is None:
        reviews["label"] = "unknown"
        reviews["score"] = np.nan
        return reviews

    predictions = sentiment_module.predict(sentiment_model, list(reviews["review_text"]))
    reviews["label"] = [item["label"] for item in predictions]
    reviews["score"] = [item["score"] for item in predictions]
    return reviews


def _score_uploaded_timeseries(
    timeseries: pd.DataFrame | None,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fit anomaly detection and (if enough history) forecasting on-the-fly.

    Returns (anomalies, forecast, forecast_note). `forecast` is an empty frame
    when there isn't enough history -- `risk_aggregator` already treats a missing
    forecast signal as neutral (0 contribution) rather than crashing, and the note
    explains that gap to the user instead of hiding it.
    """
    empty_forecast = pd.DataFrame(
        columns=["date", "actual", "forecast", "lower_bound", "upper_bound", "is_projection"]
    )
    if timeseries is None or timeseries.empty:
        return pd.DataFrame(), empty_forecast, "No time-series data uploaded."

    timeseries = timeseries.copy()
    timeseries["date"] = pd.to_datetime(timeseries["date"])
    timeseries = timeseries.sort_values("date").reset_index(drop=True)

    features = anomaly_module.build_features(timeseries)
    anomaly_contamination = config.get("anomaly", {}).get("contamination", 0.03)
    anomaly_model = anomaly_module.train(
        features, contamination=anomaly_contamination, seed=config.get("seed", 42)
    )
    labels, scores = anomaly_module.predict(anomaly_model, features)
    features["is_anomaly"] = labels == -1
    features["anomaly_score"] = scores

    can_forecast, holdout_days, forecast_note = data_validation.forecast_readiness(len(timeseries))
    if not can_forecast:
        return features, empty_forecast, forecast_note

    forecast_model_type = config.get("forecast", {}).get("model_type", "prophet")
    train_df, holdout_df = forecast_module.chronological_split(timeseries, test_days=holdout_days)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_model = forecast_module.train(
            train_df,
            model_type=forecast_model_type,
            horizon_days=holdout_days,
            seed=config.get("seed", 42),
        )
        predictions = forecast_module.predict(forecast_model, holdout_days * 2)

    predictions = predictions.reset_index(drop=True)
    predictions["actual"] = np.nan
    predictions.loc[: holdout_days - 1, "actual"] = holdout_df["revenue"].to_numpy(dtype=float)
    predictions.loc[: holdout_days - 1, "date"] = pd.to_datetime(holdout_df["date"].to_numpy())
    predictions["is_projection"] = predictions["actual"].isna()

    return features, predictions, forecast_note
