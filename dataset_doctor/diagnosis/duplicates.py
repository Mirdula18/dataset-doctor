"""Detect duplicate rows and columns."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DuplicateResult:
    """Summary of duplicate detection."""

    duplicate_row_count: int
    duplicate_column_groups: list[list[str]]


def detect_duplicates(df: pd.DataFrame) -> DuplicateResult:
    """Detect duplicate rows and duplicate columns in *df*.

    Duplicate columns are groups of columns whose values are identical
    across every row.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    DuplicateResult
    """
    try:
        dup_row_count = int(df.duplicated().sum())
    except TypeError:
        dup_row_count = int(df.astype(str).duplicated().sum())

    # Detect duplicate column groups by comparing column value hashes.
    seen: dict[int, list[str]] = {}
    for col in df.columns:
        try:
            col_hash = pd.util.hash_pandas_object(df[col]).sum()
        except TypeError:
            col_hash = pd.util.hash_pandas_object(df[col].astype(str)).sum()
        seen.setdefault(col_hash, []).append(str(col))

    dup_col_groups = [group for group in seen.values() if len(group) > 1]

    return DuplicateResult(
        duplicate_row_count=dup_row_count,
        duplicate_column_groups=dup_col_groups,
    )
