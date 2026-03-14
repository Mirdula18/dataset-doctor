"""Handle outliers by clipping to IQR bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def handle_outliers(df: pd.DataFrame, factor: float = 1.5, config: dict | None = None) -> pd.DataFrame:
    """Handle outliers using configurable method and action.

    Parameters
    ----------
    df : pd.DataFrame
    factor : float
        IQR multiplier (default ``1.5``).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled.
    """
    df = df.copy()
    config = config or {}

    method = str(config.get("method", "iqr"))
    action = str(config.get("action", "clip"))
    iqr_factor = float(config.get("iqr_factor", factor))
    zscore_threshold = float(config.get("zscore_threshold", 3.0))

    if method not in {"iqr", "zscore"}:
        raise ValueError("Invalid config key: outliers.method. Allowed values: iqr, zscore")
    if action not in {"clip", "remove"}:
        raise ValueError("Invalid config key: outliers.action. Allowed values: clip, remove")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df

    if action == "remove":
        outlier_rows = pd.Series(False, index=df.index)
    else:
        outlier_rows = None

    for col in numeric_cols:
        series = df[col]

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
        else:
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            lower = mean - zscore_threshold * std
            upper = mean + zscore_threshold * std

        if action == "clip":
            df[col] = series.clip(lower=lower, upper=upper)
        else:
            outlier_rows = outlier_rows | ((series < lower) | (series > upper))

    if action == "remove" and outlier_rows is not None:
        df = df.loc[~outlier_rows].reset_index(drop=True)

    return df
