"""Tests for churn module."""

import numpy as np
import pandas as pd

from src.churn_module import (
    engineer_features,
    evaluate,
    load_model,
    predict,
    save_model,
    train,
)


def _sample_churn_df() -> pd.DataFrame:
    """Return a small learnable churn dataset for unit tests."""
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:03d}" for index in range(12)],
            "tenure": [3, 4, 5, 6, 24, 30, 36, 40, 7, 8, 48, 60],
            "monthly_spend": [25, 28, 30, 32, 80, 90, 95, 100, 34, 36, 105, 110],
            "usage_frequency": [1, 2, 2, 3, 25, 28, 30, 31, 3, 4, 32, 34],
            "support_tickets": [2, 1, 2, 1, 0, 0, 0, 0, 1, 2, 0, 0],
            "plan_type": [
                "basic",
                "basic",
                "basic",
                "basic",
                "pro",
                "pro",
                "enterprise",
                "enterprise",
                "basic",
                "basic",
                "enterprise",
                "enterprise",
            ],
            "churn_label": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
    )


def test_engineer_features_adds_risk_features() -> None:
    """Feature engineering should keep base columns and add useful ratios."""
    features = engineer_features(_sample_churn_df())

    assert "customer_id" not in features.columns
    assert "churn_label" not in features.columns
    assert "spend_per_login" in features.columns
    assert "tickets_per_tenure" in features.columns
    assert "low_usage_flag" in features.columns


def test_predict_returns_probabilities_between_zero_and_one() -> None:
    """Predictions should be one probability per input row."""
    df = _sample_churn_df()
    X = engineer_features(df)
    y = df["churn_label"]

    model = train(X, y, model_type="logistic_regression")
    probabilities = predict(model, X)

    assert len(probabilities) == len(X)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_evaluate_returns_expected_metrics() -> None:
    """Evaluation should expose the Phase 3 metric contract."""
    df = _sample_churn_df()
    X = engineer_features(df)
    y = df["churn_label"]
    model = train(X, y, model_type="random_forest")

    metrics = evaluate(model, X, y)

    assert set(metrics) == {"roc_auc", "pr_auc", "f1", "confusion_matrix"}
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 2


def test_save_and_load_model_preserves_predictions(tmp_path) -> None:
    """Saved and loaded models should produce identical probabilities."""
    df = _sample_churn_df()
    X = engineer_features(df)
    y = df["churn_label"]
    model = train(X, y, model_type="logistic_regression")
    path = tmp_path / "churn_model.pkl"

    save_model(model, path)
    loaded_model = load_model(path)

    assert np.allclose(predict(model, X), predict(loaded_model, X))
