"""Sentiment analysis module for StartupShield AI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


LABELS = ["negative", "neutral", "positive"]
MODEL_PATH = Path("models/sentiment_model")


def train_tfidf_baseline(
    X_train: list[str] | np.ndarray,
    y_train: list[str] | np.ndarray,
    max_features: int = 5000,
    seed: int = 42,
) -> Pipeline:
    """Train a TF-IDF + Logistic Regression sentiment classifier."""
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    model.resolved_model_type_ = "tfidf_logistic_regression"
    return model


def train_distilbert(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    epochs: int = 2,
) -> Any:
    """Placeholder for optional DistilBERT fine-tuning.

    The project MVP explicitly allows shipping the TF-IDF baseline when
    transformer fine-tuning is unavailable or too slow for the local environment.
    """
    raise NotImplementedError(
        "DistilBERT fine-tuning is optional for the MVP; use train_tfidf_baseline."
    )


def evaluate(model: Pipeline, X_test: list[str] | np.ndarray, y_test: list[str] | np.ndarray) -> dict:
    """Evaluate a sentiment classifier using the Phase 4 metric contract."""
    labels = model.predict(X_test)
    report = classification_report(
        y_test,
        labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_test, labels, average="macro")),
        "accuracy": float(accuracy_score(y_test, labels)),
        "per_class_f1": {
            label: float(report[label]["f1-score"])
            for label in LABELS
        },
    }


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> None:
    """Save a fitted sentiment model into a directory."""
    model_dir = Path(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a saved sentiment model."""
    return joblib.load(Path(path) / "model.pkl")


def predict(model: Pipeline, texts: list[str]) -> list[dict]:
    """Return label and normalized positivity score for each text."""
    labels = model.predict(texts)
    probabilities = _predict_probabilities(model, texts)
    class_to_index = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    results = []
    for row_index, label in enumerate(labels):
        neutral_probability = probabilities[row_index, class_to_index["neutral"]]
        positive_probability = probabilities[row_index, class_to_index["positive"]]
        positivity_score = positive_probability + (0.5 * neutral_probability)
        results.append(
            {
                "label": str(label),
                "score": float(np.clip(positivity_score, 0.0, 1.0)),
            }
        )

    return results


def benchmark_inference(
    model: Pipeline,
    texts: list[str],
) -> dict:
    """Measure batch inference time for report notes."""
    started_at = time.perf_counter()
    predictions = predict(model, texts)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "rows": len(texts),
        "seconds": float(elapsed_seconds),
        "predictions": predictions,
    }


def _predict_probabilities(model: Pipeline, texts: list[str]) -> np.ndarray:
    """Return class probabilities for supported sklearn-style classifiers."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must expose predict_proba")

    probabilities = model.predict_proba(texts)
    missing_classes = sorted(set(LABELS) - set(model.classes_))
    if missing_classes:
        raise ValueError(f"model is missing sentiment classes: {missing_classes}")
    return np.clip(probabilities, 0.0, 1.0)
