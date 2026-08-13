"""Tests for preprocessing helpers."""

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    encode_categorical,
    handle_missing_values,
    scale_numeric,
    train_test_split_stratified,
)


def test_handle_missing_values_fills_numeric_and_categorical() -> None:
    """Missing numeric and categorical values should be filled."""
    df = pd.DataFrame(
        {
            "tenure": [1.0, None, 5.0],
            "plan_type": ["basic", None, "basic"],
        }
    )

    result = handle_missing_values(df)

    assert result.isna().sum().sum() == 0
    assert result.loc[1, "tenure"] == 3.0
    assert result.loc[1, "plan_type"] == "basic"


def test_handle_missing_values_empty_dataframe_returns_empty_copy() -> None:
    """An empty DataFrame should stay empty and not raise."""
    df = pd.DataFrame(columns=["tenure", "plan_type"])

    result = handle_missing_values(df)

    assert result.empty
    assert list(result.columns) == ["tenure", "plan_type"]


def test_encode_categorical_creates_dummy_columns() -> None:
    """Categorical columns should become one-hot indicator columns."""
    df = pd.DataFrame({"plan_type": ["basic", "pro"], "tenure": [1, 2]})

    result = encode_categorical(df, ["plan_type"])

    assert "plan_type_basic" in result.columns
    assert "plan_type_pro" in result.columns
    assert "plan_type" not in result.columns


def test_encode_categorical_raises_for_missing_columns() -> None:
    """Missing categorical columns should fail with a clear error."""
    df = pd.DataFrame({"tenure": [1, 2]})

    with pytest.raises(ValueError, match="Missing columns"):
        encode_categorical(df, ["plan_type"])


def test_scale_numeric_standardizes_selected_columns() -> None:
    """Standard scaling should center selected numeric columns."""
    df = pd.DataFrame({"tenure": [1.0, 2.0, 3.0], "churn_label": [0, 1, 0]})

    result = scale_numeric(df, ["tenure"])

    assert round(result["tenure"].mean(), 7) == 0
    assert result["churn_label"].tolist() == [0, 1, 0]


def test_scale_numeric_empty_dataframe_returns_empty_copy() -> None:
    """An empty DataFrame should stay empty during scaling."""
    df = pd.DataFrame(columns=["tenure"])

    result = scale_numeric(df, ["tenure"])

    assert result.empty
    assert list(result.columns) == ["tenure"]


def test_clean_text_removes_urls_html_and_extra_spaces() -> None:
    """Text cleaning should normalize noisy review text."""
    text = " GREAT!!! <br> Visit https://example.com now   "

    result = clean_text(text)

    assert result == "great!!! visit now"


def test_train_test_split_stratified_preserves_classes() -> None:
    """Stratified split should keep both classes in train and test sets."""
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0] * 10 + [1] * 10)

    _, _, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)

    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}
    assert len(y_test) == 5
