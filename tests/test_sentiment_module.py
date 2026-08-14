"""Tests for sentiment module."""

import numpy as np

from src.sentiment_module import (
    evaluate,
    load_model,
    predict,
    save_model,
    train_tfidf_baseline,
)


def _sample_reviews() -> tuple[list[str], list[str]]:
    """Return a small learnable sentiment dataset."""
    texts = [
        "Terrible outages and slow support. We are unhappy.",
        "Bad billing problem and confusing product.",
        "Frustrating experience with repeated issues.",
        "It is okay, support answered eventually.",
        "Average product, nothing special but usable.",
        "Neutral update: the dashboard works as expected.",
        "Great experience with fast fixes and clear communication!",
        "Excellent product, reliable and easy to use.",
        "Amazing support and helpful features.",
    ]
    labels = [
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "positive",
        "positive",
        "positive",
    ]
    return texts, labels


def test_tfidf_baseline_trains_and_predicts_sentiment_schema() -> None:
    """Predictions should match the risk aggregator sentiment contract."""
    texts, labels = _sample_reviews()
    model = train_tfidf_baseline(texts, labels, max_features=100)

    predictions = predict(
        model,
        [
            "Excellent support and reliable product.",
            "Awful outages and bad billing.",
            "The product is okay and usable.",
        ],
    )

    assert len(predictions) == 3
    for prediction in predictions:
        assert set(prediction) == {"label", "score"}
        assert prediction["label"] in {"negative", "neutral", "positive"}
        assert 0 <= prediction["score"] <= 1


def test_evaluate_returns_expected_sentiment_metrics() -> None:
    """Evaluation should expose the Phase 4 metric contract."""
    texts, labels = _sample_reviews()
    model = train_tfidf_baseline(texts, labels, max_features=100)

    metrics = evaluate(model, texts, labels)

    assert set(metrics) == {"macro_f1", "accuracy", "per_class_f1"}
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert set(metrics["per_class_f1"]) == {"negative", "neutral", "positive"}


def test_save_and_load_sentiment_model_preserves_predictions(tmp_path) -> None:
    """Saved and loaded sentiment models should produce identical scores."""
    texts, labels = _sample_reviews()
    model = train_tfidf_baseline(texts, labels, max_features=100)
    path = tmp_path / "sentiment_model"

    save_model(model, path)
    loaded_model = load_model(path)

    original_scores = [item["score"] for item in predict(model, texts)]
    loaded_scores = [item["score"] for item in predict(loaded_model, texts)]

    assert np.allclose(original_scores, loaded_scores)
