"""Remove duplicate rows from a DataFrame."""

from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return df.drop_duplicates().reset_index(drop=True)
