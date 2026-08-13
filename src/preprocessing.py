"""Reusable preprocessing helpers for StartupShield AI."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing values using a simple column-wise strategy.

    Args:
        df: Input DataFrame.
        strategy: Fill strategy: `median`, `mean`, `mode`, or `drop`.

    Returns:
        DataFrame with missing values handled.
    """
    if df.empty:
        return df.copy()
    if strategy not in {"median", "mean", "mode", "drop"}:
        raise ValueError("strategy must be one of: median, mean, mode, drop")

    cleaned_df = df.copy()
    if strategy == "drop":
        return cleaned_df.dropna().reset_index(drop=True)

    for column in cleaned_df.columns:
        if not cleaned_df[column].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            if strategy == "mean":
                fill_value = cleaned_df[column].mean()
            elif strategy == "mode":
                fill_value = _mode_or_default(cleaned_df[column], 0)
            else:
                fill_value = cleaned_df[column].median()
        else:
            fill_value = _mode_or_default(cleaned_df[column], "unknown")

        cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode selected categorical columns.

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.

    Returns:
        DataFrame with selected categorical columns one-hot encoded.
    """
    _validate_columns(df, columns)
    if df.empty:
        return df.copy()
    return pd.get_dummies(df.copy(), columns=columns, dummy_na=False)


def scale_numeric(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> pd.DataFrame:
    """Scale selected numeric columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to scale.
        method: Scaling method: `standard` or `minmax`.

    Returns:
        DataFrame with scaled numeric columns.
    """
    _validate_columns(df, columns)
    if method not in {"standard", "minmax"}:
        raise ValueError("method must be one of: standard, minmax")
    if df.empty or not columns:
        return df.copy()

    scaled_df = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
    return scaled_df


def clean_text(text: str) -> str:
    """Normalize text while keeping sentiment punctuation.

    Args:
        text: Raw text value.

    Returns:
        Cleaned lowercase text.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def train_test_split_stratified(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Split data into train and test sets while preserving label ratios.

    Args:
        X: Feature matrix or DataFrame.
        y: Target labels.
        test_size: Fraction of rows to place in the test set.
        seed: Random seed for reproducible splitting.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def _mode_or_default(series: pd.Series, default: Any) -> Any:
    """Return a series mode or a fallback value for all-missing columns."""
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return default
    return mode_values.iloc[0]


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when expected columns are absent."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
