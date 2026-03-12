"""Detect highly correlated feature pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CorrelationPair:
    """A pair of columns whose absolute correlation exceeds the threshold."""

    column_a: str
    column_b: str
    correlation: float


def detect_high_correlations(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> list[CorrelationPair]:
    """Return pairs of numeric columns with |correlation| >= *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Absolute correlation threshold (default ``0.9``).

    Returns
    -------
    list[CorrelationPair]
        Sorted descending by absolute correlation.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr_matrix = numeric_df.corr()
    results: list[CorrelationPair] = []
    seen: set[tuple[str, str]] = set()

    for i, col_a in enumerate(corr_matrix.columns):
        for j, col_b in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            value = corr_matrix.iloc[i, j]
            if pd.isna(value):
                continue
            if abs(value) >= threshold and (col_a, col_b) not in seen:
                seen.add((col_a, col_b))
                results.append(
                    CorrelationPair(
                        column_a=str(col_a),
                        column_b=str(col_b),
                        correlation=round(float(value), 4),
                    )
                )

    results.sort(key=lambda r: abs(r.correlation), reverse=True)
    return results
