"""Detect constant (zero-variance) columns."""

from __future__ import annotations

import pandas as pd


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return names of columns that have at most one unique value.

    ``NaN`` is **not** counted as a unique value.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    results: list[str] = []
    for col in df.columns:
        try:
            if df[col].nunique(dropna=True) <= 1:
                results.append(str(col))
        except TypeError:
            if df[col].astype(str).nunique(dropna=True) <= 1:
                results.append(str(col))
    return results
