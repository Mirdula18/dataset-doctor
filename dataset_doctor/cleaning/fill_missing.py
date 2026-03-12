"""Fill missing values with sensible defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in-place strategy: median for numeric, mode for categorical.

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

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_values = df[col].mode()
            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])

    return df
