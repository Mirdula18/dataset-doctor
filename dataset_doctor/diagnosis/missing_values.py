"""Detect missing values in a DataFrame."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MissingValueResult:
    """Result of missing-value analysis for a single column."""

    column: str
    missing_count: int
    missing_percentage: float


def detect_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> list[MissingValueResult]:
    """Return columns with missing values above *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Minimum missing-percentage (0–100) to include in the results.
        Default ``0.0`` returns every column that has at least one missing value.

    Returns
    -------
    list[MissingValueResult]
        Sorted descending by missing percentage.
    """
    total_rows = len(df)
    if total_rows == 0:
        return []

    missing_counts: pd.Series = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    results: list[MissingValueResult] = []
    for col, count in missing_counts.items():
        pct = (count / total_rows) * 100
        if pct >= threshold:
            results.append(
                MissingValueResult(
                    column=str(col),
                    missing_count=int(count),
                    missing_percentage=round(pct, 2),
                )
            )

    results.sort(key=lambda r: r.missing_percentage, reverse=True)
    return results
