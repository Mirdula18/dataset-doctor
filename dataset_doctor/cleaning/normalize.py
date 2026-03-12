"""Normalize numeric features using Min-Max scaling."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply Min-Max normalization to all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not mutated).
    scaler : MinMaxScaler, optional
        A pre-fitted scaler. If ``None`` (default), a new scaler is created
        and **fitted** on *df*'s numeric columns before transforming.
        If provided, only ``transform`` is called — useful for applying the
        same scaling to a test set without data leakage.

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler]
        The normalized DataFrame and the (fitted) scaler.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df, scaler or MinMaxScaler()

    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler
