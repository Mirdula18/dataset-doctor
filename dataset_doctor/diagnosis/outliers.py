"""Detect outliers using the IQR method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutlierResult:
    """Outlier information for a single numeric column."""

    column: str
    outlier_count: int
    lower_bound: float
    upper_bound: float


def detect_outliers(
    df: pd.DataFrame,
    factor: float = 1.5,
) -> list[OutlierResult]:
    """Detect outliers in numeric columns using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    factor : float
        IQR multiplier (default ``1.5``).

    Returns
    -------
    list[OutlierResult]
        One entry per numeric column that contains at least one outlier.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results: list[OutlierResult] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())

        if count > 0:
            results.append(
                OutlierResult(
                    column=str(col),
                    outlier_count=count,
                    lower_bound=round(lower, 4),
                    upper_bound=round(upper, 4),
                )
            )

    return results
