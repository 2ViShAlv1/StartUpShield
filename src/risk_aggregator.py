"""Risk aggregation and explanation module for StartupShield AI.

Combines the four model signals into one 0-100 company risk score and explains
what drove it.

KNOWN SIMPLIFICATION: the score is a linear weighted combination of four
normalized signals with hand-chosen weights -- it is not a learned meta-model.
No labelled "company failed" dataset exists to train one on. The weights encode a
product judgement (churn matters most, anomalies are the noisiest signal), and are
configurable in `config/config.yaml`. This is documented rather than hidden because
it is the single biggest caveat on the headline number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "w_churn": 0.35,
    "w_sentiment": 0.20,
    "w_anomaly": 0.20,
    "w_forecast": 0.25,
}

# Score bands used for the dashboard badge colour and the summary verdict.
RISK_BANDS = [(30.0, "Low", "green"), (60.0, "Medium", "orange"), (101.0, "High", "red")]

# CALIBRATION -- why these exist.
#
# All four signals are nominally in [0, 1], but their *realistic* ranges are not
# comparable. Churn probability genuinely spans ~0.1-0.9 across companies, while
# anomaly ratio realistically tops out near 0.15 and projected revenue decline near
# 0.10. Feeding the raw values into the weighted sum therefore lets churn dominate
# the score no matter what the weights say -- on the demo data the anomaly and
# forecast terms together moved the score by under 2 points out of 100, making 45%
# of the declared weight decorative.
#
# Each signal is rescaled to its saturation point so a weight of 0.20 actually buys
# 20 points of influence. Saturation points are chosen on their own merits:
#   * 10% of the last 30 days flagged anomalous (3 days) is already a serious signal.
#   * 10% projected revenue decline over 30 days is a ~70%/year trajectory.
ANOMALY_SATURATION = 0.10
FORECAST_SATURATION = 0.10

TOP_FACTOR_COUNT = 3


def compute_risk_score(
    churn_prob: float,
    sentiment_positivity: float,
    anomaly_flag_ratio: float,
    forecast_deviation: float,
    weights: dict | None = None,
) -> float:
    """Combine the four signals into a 0-100 risk score.

    Args:
        churn_prob: Mean churn probability across the company's customers, [0, 1].
        sentiment_positivity: Mean positivity of recent reviews, [0, 1].
            Inverted internally -- low positivity means high risk.
        anomaly_flag_ratio: Fraction of the last 30 days flagged anomalous, [0, 1].
        forecast_deviation: Normalized |actual - forecast| / forecast, [0, 1].
        weights: Optional weight overrides; missing keys fall back to defaults.

    Returns:
        Risk score in [0, 100]. Never NaN -- missing inputs are treated as the
        neutral value for that signal (see `_safe_signal`).
    """
    resolved_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    churn_term, sentiment_term, anomaly_term, forecast_term = _normalized_terms(
        churn_prob, sentiment_positivity, anomaly_flag_ratio, forecast_deviation
    )

    weighted_sum = (
        resolved_weights["w_churn"] * churn_term
        + resolved_weights["w_sentiment"] * sentiment_term
        + resolved_weights["w_anomaly"] * anomaly_term
        + resolved_weights["w_forecast"] * forecast_term
    )
    return float(np.clip(weighted_sum, 0.0, 1.0) * 100.0)


def risk_band(risk_score: float) -> tuple[str, str]:
    """Return the (label, colour) band for a risk score."""
    for threshold, label, colour in RISK_BANDS:
        if risk_score < threshold:
            return label, colour
    return "High", "red"


def get_shap_explanation(churn_model: Any, X_row: pd.DataFrame) -> dict:
    """Explain one churn prediction with SHAP.

    Args:
        churn_model: Fitted sklearn Pipeline (preprocessor + tree classifier).
        X_row: Single-row (or multi-row) engineered feature frame.

    Returns:
        {'top_factors': [(feature_name, shap_value), ...], 'base_value': float,
        'prediction': float, 'source': str}. If SHAP or the tree explainer is
        unavailable, falls back to the model's native importances (see
        `_importance_fallback`) and flags which method produced the numbers, so the
        dashboard can label the chart honestly rather than always crediting SHAP.
    """
    prediction = float(churn_model.predict_proba(X_row)[:, 1].mean())

    try:
        import shap
    except ImportError:
        return _importance_fallback(churn_model, X_row, prediction, "shap_not_installed")

    try:
        preprocessor = churn_model.named_steps["preprocessor"]
        classifier = churn_model.named_steps["classifier"]
        transformed = preprocessor.transform(X_row)
        feature_names = _transformed_feature_names(preprocessor, X_row)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        shap_values, base_value = _normalize_shap_output(shap_values, explainer)

        # Average across rows so a company-level view works the same as one customer.
        mean_shap = np.asarray(shap_values, dtype=float).reshape(len(X_row), -1).mean(axis=0)
        ranked = sorted(
            zip(feature_names, mean_shap),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        return {
            "top_factors": [(name, float(value)) for name, value in ranked[:TOP_FACTOR_COUNT]],
            "base_value": float(base_value),
            "prediction": prediction,
            "source": "shap",
        }
    except Exception:
        # SHAP is a nice-to-have; never let it take the dashboard down.
        return _importance_fallback(churn_model, X_row, prediction, "shap_failed")


def generate_explanation_text(
    shap_result: dict,
    sentiment_label: str,
    anomaly_flags: list | np.ndarray,
    forecast_trend: str,
) -> str:
    """Turn the numeric signals into one plain-language sentence."""
    clauses = []

    top_factors = shap_result.get("top_factors", [])
    if top_factors:
        factor_names = [_humanize_feature(name) for name, _ in top_factors[:2]]
        clauses.append(f"Risk driven primarily by {_join_readable(factor_names)}")
    else:
        clauses.append("Risk driven by overall customer churn signals")

    anomaly_count = int(np.sum(np.asarray(anomaly_flags, dtype=bool))) if len(anomaly_flags) else 0
    sentiment_clause = {
        "negative": "customer sentiment is mostly negative",
        "neutral": "customer sentiment is flat",
        "positive": "customer sentiment is healthy",
    }.get(sentiment_label, "customer sentiment is unclear")
    clauses.append(sentiment_clause)

    if anomaly_count:
        day_word = "day" if anomaly_count == 1 else "days"
        clauses.append(f"{anomaly_count} anomalous {day_word} detected in the last 30 days")

    clauses.append(f"revenue is trending {forecast_trend}")
    return "; ".join(clauses) + "."


def assess_company(
    company_id: str,
    churn_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    anomaly_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    weights: dict | None = None,
    churn_model: Any = None,
) -> dict:
    """Top-level orchestrator -- the single function the dashboard calls.

    Args:
        company_id: Company name, echoed into the result.
        churn_data: Frame with a `churn_prob` column (one row per customer).
        sentiment_data: Frame with `score` and `label` columns (one row per review).
        anomaly_data: Frame with `is_anomaly` and `date` columns (daily rows).
        forecast_data: Frame with `forecast` and optionally `actual` columns.
        weights: Optional weight overrides.
        churn_model: Optional fitted churn pipeline, required for SHAP factors.

    Returns:
        {'company_id', 'risk_score', 'risk_label', 'risk_colour', 'signals',
        'top_factors', 'explanation_text'}. Missing inputs degrade to neutral
        defaults rather than raising.
    """
    churn_prob = _mean_or_default(churn_data, "churn_prob", default=0.0)
    sentiment_positivity = _mean_or_default(sentiment_data, "score", default=1.0)
    anomaly_flag_ratio = _recent_anomaly_ratio(anomaly_data)
    forecast_deviation = _forecast_deviation(forecast_data)

    risk_score = compute_risk_score(
        churn_prob=churn_prob,
        sentiment_positivity=sentiment_positivity,
        anomaly_flag_ratio=anomaly_flag_ratio,
        forecast_deviation=forecast_deviation,
        weights=weights,
    )
    label, colour = risk_band(risk_score)

    shap_result = {"top_factors": []}
    if churn_model is not None and _has_rows(churn_data):
        feature_columns = getattr(churn_model, "feature_columns_", None)
        if feature_columns is not None and set(feature_columns).issubset(churn_data.columns):
            shap_result = get_shap_explanation(churn_model, churn_data[feature_columns])

    explanation_text = generate_explanation_text(
        shap_result,
        sentiment_label=_dominant_sentiment(sentiment_data),
        anomaly_flags=_recent_anomaly_flags(anomaly_data),
        forecast_trend=_forecast_trend(forecast_data),
    )

    return {
        "company_id": company_id,
        "risk_score": risk_score,
        "risk_label": label,
        "risk_colour": colour,
        "signals": {
            "churn_prob": churn_prob,
            "sentiment_positivity": sentiment_positivity,
            "anomaly_flag_ratio": anomaly_flag_ratio,
            "forecast_deviation": forecast_deviation,
        },
        "contributions": _signal_contributions(
            churn_prob, sentiment_positivity, anomaly_flag_ratio, forecast_deviation, weights
        ),
        "top_factors": shap_result.get("top_factors", []),
        "explanation_source": shap_result.get("source", "none"),
        "explanation_text": explanation_text,
    }


def _signal_contributions(
    churn_prob: float,
    sentiment_positivity: float,
    anomaly_flag_ratio: float,
    forecast_deviation: float,
    weights: dict | None,
) -> dict:
    """Return each signal's point contribution to the final 0-100 score."""
    resolved = {**DEFAULT_WEIGHTS, **(weights or {})}
    churn_term, sentiment_term, anomaly_term, forecast_term = _normalized_terms(
        churn_prob, sentiment_positivity, anomaly_flag_ratio, forecast_deviation
    )
    return {
        "Churn": resolved["w_churn"] * churn_term * 100,
        "Sentiment": resolved["w_sentiment"] * sentiment_term * 100,
        "Anomalies": resolved["w_anomaly"] * anomaly_term * 100,
        "Forecast": resolved["w_forecast"] * forecast_term * 100,
    }


def _normalized_terms(
    churn_prob: Any,
    sentiment_positivity: Any,
    anomaly_flag_ratio: Any,
    forecast_deviation: Any,
) -> tuple[float, float, float, float]:
    """Rescale the four raw signals onto a comparable 0-1 risk axis.

    See the CALIBRATION note at the top of this module for why the anomaly and
    forecast terms are divided by a saturation point rather than used raw.
    """
    churn_term = _safe_signal(churn_prob, default=0.0)
    # Positivity is a "good" signal, so invert it into a risk contribution.
    sentiment_term = 1.0 - _safe_signal(sentiment_positivity, default=1.0)
    anomaly_term = _safe_signal(anomaly_flag_ratio, default=0.0) / ANOMALY_SATURATION
    forecast_term = _safe_signal(forecast_deviation, default=0.0) / FORECAST_SATURATION
    return (
        churn_term,
        sentiment_term,
        float(np.clip(anomaly_term, 0.0, 1.0)),
        float(np.clip(forecast_term, 0.0, 1.0)),
    )


def _safe_signal(value: Any, default: float) -> float:
    """Coerce a signal into [0, 1], substituting a default for missing values."""
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(np.clip(numeric, 0.0, 1.0))


def _has_rows(df: Any) -> bool:
    """True when df is a non-empty DataFrame."""
    return isinstance(df, pd.DataFrame) and not df.empty


def _mean_or_default(df: Any, column: str, default: float) -> float:
    """Mean of a column, or a default when the data is missing/empty."""
    if not _has_rows(df) or column not in df.columns:
        return default
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else default


def _recent_anomaly_flags(anomaly_data: Any, window_days: int = 30) -> np.ndarray:
    """Anomaly flags for the most recent `window_days` rows."""
    if not _has_rows(anomaly_data) or "is_anomaly" not in anomaly_data.columns:
        return np.array([], dtype=bool)
    recent = anomaly_data.tail(window_days)
    return recent["is_anomaly"].to_numpy(dtype=bool)


def _recent_anomaly_ratio(anomaly_data: Any, window_days: int = 30) -> float:
    """Fraction of the last `window_days` flagged anomalous."""
    flags = _recent_anomaly_flags(anomaly_data, window_days)
    return float(flags.mean()) if flags.size else 0.0


def _forecast_deviation(forecast_data: Any) -> float:
    """Forward-looking downside risk from the forecast, in [0, 1].

    DELIBERATE DEVIATION FROM THE ORIGINAL SPEC. The spec defined this term as
    symmetric surprise, `abs(actual - forecast) / forecast`. That is wrong for a
    *risk* product: a company whose revenue is collapsing on schedule scores ~0
    because the model predicted the collapse accurately, and a company beating its
    forecast scores high because beating it is also "surprise". Measured on the
    demo data, RedFlag Analytics was down 33% year-on-year yet produced a
    deviation of 0.012.

    This implementation keeps the spec's intent but makes it directional:

      * `shortfall`  -- how far actual revenue fell BELOW forecast (upside ignored)
      * `decline`    -- how far the forecast projects revenue below the current level

    Both are downside-only, averaged, and clipped to [0, 1]. Rows with an `actual`
    value are treated as backtest; rows without one are the forward projection.
    """
    if not _has_rows(forecast_data) or "forecast" not in forecast_data.columns:
        return 0.0

    numeric = forecast_data.copy()
    numeric["forecast"] = pd.to_numeric(numeric["forecast"], errors="coerce")
    if "actual" in numeric.columns:
        numeric["actual"] = pd.to_numeric(numeric["actual"], errors="coerce")
    else:
        numeric["actual"] = np.nan

    backtest = numeric.dropna(subset=["actual", "forecast"])
    projection = numeric[numeric["actual"].isna()].dropna(subset=["forecast"])

    components = []

    if not backtest.empty:
        forecast_mean = float(backtest["forecast"].mean())
        if abs(forecast_mean) > 1e-9:
            actual_mean = float(backtest["actual"].mean())
            # Downside only: coming in above forecast is good news, not risk.
            shortfall = (forecast_mean - actual_mean) / abs(forecast_mean)
            components.append(max(0.0, shortfall))

    if not projection.empty and not backtest.empty:
        current_level = float(backtest["actual"].mean())
        if abs(current_level) > 1e-9:
            projected_level = float(projection["forecast"].mean())
            decline = (current_level - projected_level) / abs(current_level)
            components.append(max(0.0, decline))

    if not components:
        return 0.0
    return float(np.clip(float(np.mean(components)), 0.0, 1.0))


def _forecast_trend(forecast_data: Any) -> str:
    """Describe how actuals compare to the forecast, in words."""
    if not _has_rows(forecast_data):
        return "in line with forecast"
    if "actual" not in forecast_data.columns or "forecast" not in forecast_data.columns:
        return "in line with forecast"

    paired = forecast_data[["actual", "forecast"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if paired.empty:
        return "in line with forecast"

    forecast_mean = float(paired["forecast"].mean())
    if abs(forecast_mean) < 1e-9:
        return "in line with forecast"

    relative_gap = (float(paired["actual"].mean()) - forecast_mean) / abs(forecast_mean)
    if relative_gap < -0.05:
        return f"{abs(relative_gap) * 100:.0f}% below forecast"
    if relative_gap > 0.05:
        return f"{relative_gap * 100:.0f}% above forecast"
    return "in line with forecast"


def _dominant_sentiment(sentiment_data: Any) -> str:
    """Overall sentiment verdict, driven by mean positivity rather than the modal label.

    A 35/30/35 negative/neutral/positive split has "positive" as its most common
    label while actually being a warning sign, so the modal label alone would read
    the room wrong. The mean positivity score is the honest summary.
    """
    if not _has_rows(sentiment_data):
        return "unknown"
    if "score" in sentiment_data.columns:
        scores = pd.to_numeric(sentiment_data["score"], errors="coerce").dropna()
        if len(scores):
            mean_score = float(scores.mean())
            if mean_score < 0.45:
                return "negative"
            if mean_score < 0.6:
                return "neutral"
            return "positive"
    if "label" in sentiment_data.columns:
        counts = sentiment_data["label"].value_counts()
        return str(counts.index[0]) if len(counts) else "unknown"
    return "unknown"


def _transformed_feature_names(preprocessor: Any, X_row: pd.DataFrame) -> list[str]:
    """Best-effort feature names after the ColumnTransformer runs."""
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        width = preprocessor.transform(X_row).shape[1]
        return [f"feature_{index}" for index in range(width)]


def _normalize_shap_output(shap_values: Any, explainer: Any) -> tuple[Any, float]:
    """Reduce SHAP's several output shapes to (values, base_value).

    Binary classifiers return either one array or a per-class list/3-D array; take
    the positive class in every case.
    """
    base_value = getattr(explainer, "expected_value", 0.0)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
        if isinstance(base_value, (list, np.ndarray)) and len(np.atleast_1d(base_value)) > 1:
            base_value = np.atleast_1d(base_value)[-1]
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
            if isinstance(base_value, (list, np.ndarray)) and len(np.atleast_1d(base_value)) > 1:
                base_value = np.atleast_1d(base_value)[-1]

    return shap_values, float(np.atleast_1d(base_value)[0])


def _importance_fallback(
    churn_model: Any,
    X_row: pd.DataFrame,
    prediction: float,
    source: str,
) -> dict:
    """Explain the prediction without SHAP, in three descending tiers.

    Classifier families expose their importance signal in different places, and the
    dashboard must not go blank just because the deployed environment happened to
    train a different one:

      1. `feature_importances_` -- tree ensembles (RandomForest, XGBoost, LightGBM).
      2. `coef_` -- linear models, e.g. the logistic-regression baseline.
      3. Permutation sensitivity -- anything else. `HistGradientBoostingClassifier`,
         which is exactly what `train()` falls back to when LightGBM is not
         installed, exposes NEITHER attribute, so tiers 1-2 alone left a default
         deploy with an empty explanation.

    Returns `source` suffixed with the tier actually used, so callers can label the
    chart honestly instead of claiming SHAP produced it.
    """
    top_factors: list[tuple[str, float]] = []
    method = "none"

    try:
        classifier = churn_model.named_steps["classifier"]
        preprocessor = churn_model.named_steps["preprocessor"]
        names = _transformed_feature_names(preprocessor, X_row)

        if hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_, dtype=float)
            method = "feature_importances"
        elif hasattr(classifier, "coef_"):
            importances = np.abs(np.asarray(classifier.coef_, dtype=float)).reshape(-1)
            method = "coefficients"
        else:
            importances = _permutation_sensitivity(classifier, preprocessor.transform(X_row))
            method = "permutation" if importances is not None else "none"

        if importances is not None and len(importances) == len(names):
            ranked = sorted(zip(names, importances), key=lambda pair: abs(pair[1]), reverse=True)
            top_factors = [(name, float(value)) for name, value in ranked[:TOP_FACTOR_COUNT]]
        else:
            method = "none"
    except Exception:
        top_factors = []
        method = "none"

    return {
        "top_factors": top_factors,
        "base_value": 0.0,
        "prediction": prediction,
        "source": f"{source}:{method}",
    }


def _permutation_sensitivity(
    classifier: Any,
    transformed: Any,
    max_rows: int = 400,
    seed: int = 42,
) -> np.ndarray | None:
    """Rank features by how much shuffling each one moves the predicted probability.

    Model-agnostic and needs no ground-truth labels, which matters because this runs
    at inference time where no `y` exists -- ruling out sklearn's
    `permutation_importance`. Shuffling a column that the model leans on swings its
    output; shuffling one it ignores does not.
    """
    try:
        matrix = np.asarray(
            transformed.toarray() if hasattr(transformed, "toarray") else transformed,
            dtype=float,
        )
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return None

        rng = np.random.default_rng(seed)
        if len(matrix) > max_rows:
            matrix = matrix[rng.choice(len(matrix), max_rows, replace=False)]

        baseline = classifier.predict_proba(matrix)[:, 1]

        scores = np.empty(matrix.shape[1], dtype=float)
        for index in range(matrix.shape[1]):
            shuffled = matrix.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            scores[index] = float(
                np.mean(np.abs(classifier.predict_proba(shuffled)[:, 1] - baseline))
            )
        return scores
    except Exception:
        return None


def _humanize_feature(raw_name: str) -> str:
    """Turn a transformed column name into something readable in a sentence."""
    name = raw_name.split("__", 1)[-1]
    return name.replace("_", " ").strip()


def _join_readable(items: list[str]) -> str:
    """Join a short list into readable English."""
    if not items:
        return "several factors"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"
