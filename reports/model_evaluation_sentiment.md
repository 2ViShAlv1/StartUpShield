# Sentiment Model Evaluation

## Dataset

- Source file: `data/raw/reviews.csv`
- Rows: 3,000
- Text column: `review_text`
- Target: `sentiment_label`
- Classes:
  - `negative`: 1,000
  - `neutral`: 1,000
  - `positive`: 1,000
- Split: stratified 80/20 train-test split with seed `42`
  - train rows: 2,400
  - test rows: 600

## Text Pipeline

- Text cleaning uses `preprocessing.clean_text`.
- Features use TF-IDF unigrams and bigrams.
- Maximum features: 5,000.
- Classifier: Logistic Regression with balanced class weights.

## Model Comparison

| Model | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | 100-review Inference | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0082 sec | Selected MVP model |
| DistilBERT | Not run | Not run | Not run | Not run | Not run | Not run | Skipped because TF-IDF exceeded the gate and is faster/lighter for CPU inference |

## Selected Model

- Best model: TF-IDF + Logistic Regression
- Saved artifact: `models/sentiment_model/model.pkl`
- Selection reason: exceeds required macro-F1 threshold with very fast CPU inference.
- Prediction contract: `predict()` returns `{"label": <sentiment>, "score": <0-1 positivity score>}` for each input review.

## Acceptance Criteria

- Required macro-F1: > 0.75
- Achieved macro-F1: 1.0000
- Batch inference requirement: 100 reviews in < 5 sec on CPU
- Achieved batch inference: 0.0082 sec for 100 reviews
- Status: passed

## Notes

- The current dataset is template-based and balanced, so the perfect test score should be interpreted as a Phase 4 integration/MVP result rather than proof of broad real-world generalization.
- A transformer comparison can be added later if a larger, messier review dataset is introduced.
