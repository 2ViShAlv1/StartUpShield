"""Churn prediction module for StartupShield AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn_label"
BASE_FEATURE_COLUMNS = [
    "tenure",
    "monthly_spend",
    "usage_frequency",
    "support_tickets",
    "plan_type",
]
MODEL_PATH = Path("models/churn_model.pkl")

# The real SaaS source dataset has no spend or plan fields, so
# generate_synthetic_data.load_real_saas_churn_data() derives these two columns
# deterministically from Daily_Usage_Mins (see DECISIONS.md). They are NOT
# independent business signals: any feature-importance or SHAP output must
# report them as usage-minutes proxies, never as pricing or tier effects.
PROXY_FEATURE_COLUMNS = {
    "monthly_spend": "linear transform of Daily_Usage_Mins",
    "plan_type": "binned Daily_Usage_Mins",
    "support_tickets": "keyword flag over last-support-ticket text",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create churn modeling features from the Phase 1 churn schema.

    Args:
        df: Churn DataFrame containing the Phase-compatible raw columns.

    Returns:
        Feature DataFrame ready for preprocessing/modeling.

    Note:
        When the frame comes from the real SaaS source, the columns listed in
        `PROXY_FEATURE_COLUMNS` are derived from a single underlying variable
        and must be interpreted accordingly downstream.
    """
    missing_columns = [column for column in BASE_FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing churn feature columns: {missing_columns}")

    features = df[BASE_FEATURE_COLUMNS].copy()
    features["spend_per_login"] = features["monthly_spend"] / (
        features["usage_frequency"] + 1
    )
    features["tickets_per_tenure"] = features["support_tickets"] / (
        features["tenure"] + 1
    )
    features["usage_per_tenure"] = features["usage_frequency"] / (
        features["tenure"] + 1
    )
    features["low_usage_flag"] = (features["usage_frequency"] <= 4).astype(int)
    features["high_ticket_flag"] = (features["support_tickets"] >= 1).astype(int)
    features["early_customer_flag"] = (features["tenure"] <= 6).astype(int)
    return features


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    model_type: str = "xgboost",
    seed: int = 42,
) -> Pipeline:
    """Train a churn classifier.

    Args:
        X_train: Engineered feature DataFrame.
        y_train: Binary churn labels.
        model_type: `logistic_regression`, `random_forest`, `xgboost`, or
            `lightgbm`. Optional gradient-boosting dependencies fall back to
            sklearn models when unavailable.
        seed: Random seed for reproducible training.

    Returns:
        Fitted sklearn Pipeline.
    """
    model_type = model_type.lower()
    preprocessor = _build_preprocessor(X_train)

    if model_type in {"logistic", "logistic_regression", "baseline"}:
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        resolved_model_type = "logistic_regression"
    elif model_type in {"random_forest", "rf"}:
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        resolved_model_type = "random_forest"
    elif model_type == "xgboost":
        estimator = _build_xgboost_classifier(y_train, seed)
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            resolved_model_type = "random_forest_fallback_for_xgboost"
        else:
            resolved_model_type = "xgboost"
    elif model_type in {"lightgbm", "lgbm"}:
        estimator = _build_lightgbm_classifier(seed)
        if estimator is None:
            estimator = HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=seed,
            )
            resolved_model_type = "hist_gradient_boosting_fallback_for_lightgbm"
        else:
            resolved_model_type = "lightgbm"
    else:
        raise ValueError(
            "model_type must be one of: logistic_regression, random_forest, "
            "xgboost, lightgbm"
        )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    model.fit(X_train, y_train)
    model.resolved_model_type_ = resolved_model_type
    model.feature_columns_ = list(X_train.columns)
    return model


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series | np.ndarray) -> dict:
    """Evaluate a churn classifier.

    Args:
        model: Fitted model with `predict_proba`.
        X_test: Test feature DataFrame.
        y_test: Binary churn labels.

    Returns:
        Dictionary with ROC-AUC, PR-AUC, confusion matrix, and F1.
    """
    probabilities = predict(model, X_test)
    labels = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, labels)),
        "confusion_matrix": confusion_matrix(y_test, labels).tolist(),
    }


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> None:
    """Save a fitted churn model.

    Args:
        model: Fitted sklearn Pipeline.
        path: Destination pickle path.

    Returns:
        None.
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a saved churn model.

    Args:
        path: Saved model path.

    Returns:
        Fitted sklearn Pipeline.
    """
    return joblib.load(path)


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities in the range [0, 1]."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must expose predict_proba")
    probabilities = model.predict_proba(X)[:, 1]
    return np.clip(probabilities, 0.0, 1.0)


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    seed: int = 42,
) -> dict[str, Pipeline]:
    """Train the Phase 3 candidate churn models."""
    return {
        "logistic_regression": train(
            X_train,
            y_train,
            model_type="logistic_regression",
            seed=seed,
        ),
        "random_forest": train(
            X_train,
            y_train,
            model_type="random_forest",
            seed=seed,
        ),
        "xgboost": train(
            X_train,
            y_train,
            model_type="xgboost",
            seed=seed,
        ),
        "lightgbm": train(
            X_train,
            y_train,
            model_type="lightgbm",
            seed=seed,
        ),
    }


def select_best_model(
    candidates: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> tuple[str, Pipeline, dict[str, dict]]:
    """Evaluate candidate models and select the highest ROC-AUC model."""
    metrics = {
        model_name: evaluate(model, X_test, y_test)
        for model_name, model in candidates.items()
    }
    best_name = max(metrics, key=lambda model_name: metrics[model_name]["roc_auc"])
    return best_name, candidates[best_name], metrics


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical columns."""
    numeric_columns = [
        column for column in X.columns if is_numeric_dtype(X[column])
    ]
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def _build_xgboost_classifier(
    y_train: pd.Series | np.ndarray,
    seed: int,
) -> Any | None:
    """Build an XGBoost classifier when the optional dependency is installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    labels = pd.Series(y_train)
    negative_count = max(int((labels == 0).sum()), 1)
    positive_count = max(int((labels == 1).sum()), 1)
    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negative_count / positive_count,
        random_state=seed,
        n_jobs=-1,
    )


def _build_lightgbm_classifier(seed: int) -> Any | None:
    """Build the best Phase 3 LightGBM classifier when installed."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    return LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3,
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
    )
