"""Handle outliers by clipping to IQR bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def handle_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Clip numeric column values to IQR-based bounds.

    Parameters
    ----------
    df : pd.DataFrame
    factor : float
        IQR multiplier (default ``1.5``).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers clipped.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
