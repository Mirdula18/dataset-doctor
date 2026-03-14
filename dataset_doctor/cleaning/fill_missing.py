"""Fill missing values with sensible defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fill_column(series: pd.Series, strategy: str, constant_value: object) -> pd.Series:
    """Fill null values for a single column according to *strategy*."""
    if strategy == "median":
        return series.fillna(series.median())
    if strategy == "mean":
        return series.fillna(series.mean())
    if strategy == "mode":
        mode_values = series.mode()
        if mode_values.empty:
            return series.fillna(constant_value)
        return series.fillna(mode_values.iloc[0])
    if strategy == "constant":
        return series.fillna(constant_value)
    raise ValueError(
        f"Invalid config key: missing_values strategy '{strategy}'. "
        "Allowed values: mean, median, mode, constant"
    )


def fill_missing(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Fill missing values with configurable strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not mutated).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    df = df.copy()
    config = config or {}

    numeric_strategy = str(config.get("numeric_strategy", "median"))
    categorical_strategy = str(config.get("categorical_strategy", "mode"))
    max_missing_threshold = float(config.get("max_missing_threshold", 0.5))
    constant_value = config.get("constant_value")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_ratio = float(df[col].isna().mean())
        if df[col].isna().any() and missing_ratio <= max_missing_threshold:
            fallback_value = 0 if constant_value is None else constant_value
            df[col] = _fill_column(df[col], numeric_strategy, fallback_value)

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        missing_ratio = float(df[col].isna().mean())
        if df[col].isna().any() and missing_ratio <= max_missing_threshold:
            fallback_value = "missing" if constant_value is None else constant_value
            df[col] = _fill_column(df[col], categorical_strategy, fallback_value)

    return df
