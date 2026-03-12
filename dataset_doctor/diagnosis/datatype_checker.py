"""Detect columns whose dtype may be incorrect."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DatatypeIssue:
    """A column whose inferred dtype appears incorrect."""

    column: str
    current_dtype: str
    suggested_dtype: str
    convertible_ratio: float  # 0-1, fraction of non-null values that convert


def detect_datatype_issues(
    df: pd.DataFrame,
    convertible_threshold: float = 0.8,
) -> list[DatatypeIssue]:
    """Find object columns that could be numeric or datetime.

    Parameters
    ----------
    df : pd.DataFrame
    convertible_threshold : float
        Minimum fraction of non-null values that must successfully convert
        for the column to be flagged (default ``0.8``).

    Returns
    -------
    list[DatatypeIssue]
    """
    results: list[DatatypeIssue] = []

    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        total = len(series)

        # Try numeric conversion.
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().sum()) / total

        if numeric_ratio >= convertible_threshold:
            suggested = "int64" if (numeric.dropna() % 1 == 0).all() else "float64"
            results.append(
                DatatypeIssue(
                    column=str(col),
                    current_dtype=str(df[col].dtype),
                    suggested_dtype=suggested,
                    convertible_ratio=round(numeric_ratio, 4),
                )
            )
            continue

        # Try datetime conversion.
        try:
            dt = pd.to_datetime(series, errors="coerce", format="mixed")
            dt_ratio = float(dt.notna().sum()) / total
            if dt_ratio >= convertible_threshold:
                results.append(
                    DatatypeIssue(
                        column=str(col),
                        current_dtype=str(df[col].dtype),
                        suggested_dtype="datetime64[ns]",
                        convertible_ratio=round(dt_ratio, 4),
                    )
                )
        except Exception:
            pass

    return results
