"""Tests for anomaly module."""

import numpy as np
import pandas as pd

from src.anomaly_module import (
    build_features,
    evaluate_against_ground_truth,
    load_model,
    predict,
    save_model,
    train,
)
from src.generate_synthetic_data import generate_company_timeseries


def _sample_timeseries() -> pd.DataFrame:
    """Return a synthetic time-series dataset with known anomaly labels."""
    return pd.concat(
        [
            generate_company_timeseries(
                company_name="HealthyCo",
                start_date="2025-01-01",
                n_days=180,
                base_revenue=13_500,
                growth_rate=0.0015,
                n_anomalies=0,
                anomaly_magnitude=0,
                seed=7,
            ),
            generate_company_timeseries(
                company_name="RiskyCo",
                start_date="2025-01-01",
                n_days=180,
                base_revenue=11_500,
                growth_rate=-0.001,
                n_anomalies=4,
                anomaly_magnitude=0.6,
                seed=8,
            ),
        ],
        ignore_index=True,
    )


def test_build_features_adds_phase_5_timeseries_features() -> None:
    """Feature builder should expose rolling and daily-change signals."""
    features = build_features(_sample_timeseries())

    expected_columns = {
        "day_of_week",
        "revenue_rolling_mean_7",
        "revenue_rolling_std_7",
        "revenue_rolling_mean_30",
        "revenue_rolling_std_30",
        "revenue_pct_change_1d",
        "revenue_zscore_7",
    }

    assert expected_columns.issubset(features.columns)
    assert len(features) == 360
    assert not features[list(expected_columns)].isna().any().any()


def test_build_features_preserves_input_row_order_with_duplicate_index() -> None:
    """Per-company CSVs concatenated without ignore_index must stay aligned.

    Each company frame carries its own 0..n-1 index, so the combined frame has
    duplicate labels. Features must still come back in the caller's row order,
    otherwise anomaly flags taken from the input misalign with predictions.
    """
    company_frames = [
        generate_company_timeseries(
            company_name=name,
            start_date="2025-01-01",
            n_days=120,
            n_anomalies=2,
            seed=seed,
        )
        for name, seed in (("AlphaCo", 11), ("BetaCo", 12), ("GammaCo", 13))
    ]
    duplicated_index_df = pd.concat(company_frames)
    assert duplicated_index_df.index.has_duplicates

    features = build_features(duplicated_index_df)

    assert features["company_name"].tolist() == duplicated_index_df["company_name"].tolist()
    assert np.allclose(features["revenue"], duplicated_index_df["revenue"])
    assert features["is_injected_anomaly"].tolist() == (
        duplicated_index_df["is_injected_anomaly"].tolist()
    )
    assert features.index.equals(duplicated_index_df.index)


def test_train_evaluate_and_predict_anomalies_against_ground_truth() -> None:
    """Detector should recover the majority of injected anomalies."""
    features = build_features(_sample_timeseries())
    model = train(features, contamination=0.03, seed=42)

    metrics = evaluate_against_ground_truth(
        model,
        features,
        features["is_injected_anomaly"],
    )
    labels, anomaly_scores = predict(model, features)

    assert set(metrics) == {"precision", "recall", "f1"}
    assert metrics["recall"] > 0.6
    assert labels.shape == (len(features),)
    assert anomaly_scores.shape == (len(features),)
    assert set(np.unique(labels)).issubset({-1, 1})


def test_save_and_load_anomaly_model_preserves_predictions(tmp_path) -> None:
    """Saved and loaded anomaly models should produce stable outputs."""
    features = build_features(_sample_timeseries())
    model = train(features, contamination=0.03, seed=42)
    path = tmp_path / "anomaly_model.pkl"

    save_model(model, path)
    loaded_model = load_model(path)

    original_labels, original_scores = predict(model, features)
    loaded_labels, loaded_scores = predict(loaded_model, features)

    assert np.array_equal(original_labels, loaded_labels)
    assert np.allclose(original_scores, loaded_scores)
