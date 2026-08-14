# Phase 4 — Sentiment Analysis Module

## What We Have Done

- Implemented `src/sentiment_module.py`.
- Reused `preprocessing.clean_text` inside the TF-IDF pipeline.
- Trained a TF-IDF + Logistic Regression sentiment baseline.
- Used balanced class weights for the classifier.
- Evaluated accuracy, macro-F1, and per-class F1.
- Saved the chosen model to `models/sentiment_model/model.pkl`.
- Added tests in `tests/test_sentiment_module.py`.
- Wrote `reports/model_evaluation_sentiment.md`.

## Current Phase 4 Status

- Phase 4 deliverables are complete.
- Test macro-F1: 1.0000.
- Required macro-F1 threshold: > 0.75.
- Batch inference for 100 reviews: 0.0082 seconds.
- Status: passed.
- DistilBERT fine-tuning was skipped for the MVP because the TF-IDF baseline exceeds the metric gate and is much lighter for local CPU inference.
- `pytest` can be run with `.venv/bin/python -m pytest`.

## Model Results

| Model | Accuracy | Macro-F1 | 100-review Inference |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 0.0082 sec |
| DistilBERT | Not run | Not run | Not run |

## What You Should Study

- NLP basics.
- Text cleaning.
- Tokenization.
- TF-IDF.
- Multiclass classification.
- Accuracy and macro-F1.
- Difference between classic ML NLP and transformer models.

## Useful Files

- `src/sentiment_module.py`
- `tests/test_sentiment_module.py`
- `models/sentiment_model/model.pkl`
- `reports/model_evaluation_sentiment.md`
