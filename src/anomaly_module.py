"""Anomaly detection module for StartupShield AI."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


MODEL_PATH = Path("models/anomaly_model.pkl")
SERIES_COLUMNS = ["revenue", "active_users", "signups"]
NON_FEATURE_COLUMNS = {"company_name", "date", "is_injected_anomaly"}
ROW_ORDER_COLUMN = "__row_order"

# build_features() derives features for every series in SERIES_COLUMNS, but the
# detector is deliberately trained on revenue only. In the current data
# active_users is a noisy linear function of revenue and signups is a noisy
# linear function of active_users, so the extra series add variance without
# independent signal: training on all three drops injected-anomaly recall from
# 1.00 to 0.80. Widen MODEL_SERIES_COLUMNS once the series are truly independent.
MODEL_SERIES_COLUMNS = ["revenue"]
PREFERRED_MODEL_FEATURES = ["day_of_week"] + [
    f"{column}_{suffix}"
    for column in MODEL_SERIES_COLUMNS
    for suffix in (
        "pct_change_1d",
        "abs_pct_change_1d",
        "rolling_residual_7",
        "rolling_residual_30",
        "zscore_7",
        "zscore_30",
    )
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling time-series features for anomaly detection.

    Args:
        df: DataFrame with date, revenue, active_users, and signups columns.

    Returns:
        DataFrame with original columns plus rolling, percentage-change, and
        calendar features. Feature calculations are grouped by company when a
        company_name column is present. Rows are returned in the caller's input
        order with the caller's index preserved, so anomaly flags taken from the
        input frame stay aligned even when the index has duplicate labels.
    """
    missing_columns = [column for column in ["date", *SERIES_COLUMNS] if column not in df]
    if missing_columns:
        raise ValueError(f"Missing time-series columns: {missing_columns}")

    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"])

    # Concatenating per-company CSVs can produce duplicate index labels, so track
    # input position explicitly instead of relying on the index to restore order.
    original_index = working_df.index
    working_df = working_df.reset_index(drop=True)
    working_df[ROW_ORDER_COLUMN] = np.arange(len(working_df))

    group_columns = ["company_name"] if "company_name" in working_df.columns else []
    if group_columns:
        grouped_frames = [
            _build_company_features(company_df)
            for _, company_df in working_df.sort_values(group_columns + ["date"]).groupby(
                group_columns,
                sort=False,
            )
        ]
        features = pd.concat(grouped_frames)
    else:
        features = _build_company_features(working_df.sort_values("date"))

    features = features.sort_values(ROW_ORDER_COLUMN).drop(columns=ROW_ORDER_COLUMN)
    features.index = original_index
    features["date"] = features["date"].dt.strftime("%Y-%m-%d")
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def train(
    X_train: pd.DataFrame,
    contamination: float = 0.03,
    seed: int = 42,
) -> Pipeline:
    """Train an IsolationForest anomaly detector."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    feature_columns = _select_feature_columns(X_train)
    model = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=500,
                    contamination=contamination,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train[feature_columns])
    model.feature_columns_ = feature_columns
    model.contamination_ = contamination
    return model


def evaluate_against_ground_truth(
    model: Pipeline,
    X: pd.DataFrame,
    y_true_anomaly_flags: pd.Series | np.ndarray,
) -> dict:
    """Evaluate detected anomalies against injected anomaly flags."""
    labels, _ = predict(model, X)
    y_true = np.asarray(y_true_anomaly_flags).astype(bool)
    y_pred = labels == -1
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> None:
    """Save a fitted anomaly detector."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a saved anomaly detector."""
    return joblib.load(path)


def predict(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return IsolationForest labels and anomaly scores.

    Labels use sklearn's convention: -1 means anomaly and 1 means normal.
    Lower decision-function scores indicate more anomalous rows.
    """
    feature_columns = getattr(model, "feature_columns_", _select_feature_columns(X))
    labels = model.predict(X[feature_columns])
    anomaly_scores = model.decision_function(X[feature_columns])
    return labels.astype(int), anomaly_scores.astype(float)


def _build_company_features(company_df: pd.DataFrame) -> pd.DataFrame:
    """Build features for one company ordered by date."""
    features = company_df.copy()
    features["day_of_week"] = features["date"].dt.dayofweek

    for column in SERIES_COLUMNS:
        values = features[column].astype(float)
        lagged_values = values.shift(1)

        for window in (7, 30):
            rolling = lagged_values.rolling(window=window, min_periods=max(3, window // 4))
            mean_column = f"{column}_rolling_mean_{window}"
            std_column = f"{column}_rolling_std_{window}"
            features[mean_column] = rolling.mean()
            features[std_column] = rolling.std()
            features[f"{column}_rolling_residual_{window}"] = (
                values - features[mean_column]
            ) / features[mean_column]
            features[f"{column}_zscore_{window}"] = (
                values - features[mean_column]
            ) / features[std_column]

        pct_change_column = f"{column}_pct_change_1d"
        features[pct_change_column] = values.pct_change()
        features[f"{column}_abs_pct_change_1d"] = features[pct_change_column].abs()

    features["revenue_per_active_user"] = (
        features["revenue"] / features["active_users"].replace(0, np.nan)
    )
    features["signup_rate"] = (
        features["signups"] / features["active_users"].replace(0, np.nan)
    )
    return features


def _select_feature_columns(X: pd.DataFrame) -> list[str]:
    """Select stable numeric model features from a feature DataFrame."""
    preferred_columns = [column for column in PREFERRED_MODEL_FEATURES if column in X.columns]
    if preferred_columns:
        return preferred_columns

    numeric_columns = [
        column
        for column in X.select_dtypes(include=[np.number]).columns
        if column not in NON_FEATURE_COLUMNS and column != ROW_ORDER_COLUMN
    ]
    if not numeric_columns:
        raise ValueError("No numeric anomaly feature columns found")
    return numeric_columns
