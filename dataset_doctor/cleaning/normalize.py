"""Normalize numeric features using Min-Max scaling."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ScalerType = Union[MinMaxScaler, StandardScaler]


def normalize(
    df: pd.DataFrame,
    scaler: Optional[ScalerType] = None,
    *,
    method: str = "minmax",
    feature_range: tuple[float, float] = (0, 1),
) -> Tuple[pd.DataFrame, Optional[ScalerType]]:
    """Normalize numeric columns using selected strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not mutated).
    scaler : MinMaxScaler | StandardScaler, optional
        A pre-fitted scaler. If ``None`` (default), a new scaler is created
        and **fitted** on *df*'s numeric columns before transforming.
        If provided, only ``transform`` is called — useful for applying the
        same scaling to a test set without data leakage.
    method : str
        Normalization method: ``minmax``, ``standardize``, or ``none``.
    feature_range : tuple[float, float]
        Target range for Min-Max scaling.

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler | StandardScaler | None]
        The normalized DataFrame and the (fitted) scaler.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "none":
        return df, scaler

    if not numeric_cols:
        if scaler is None and method == "minmax":
            return df, MinMaxScaler(feature_range=feature_range)
        if scaler is None and method == "standardize":
            return df, StandardScaler()
        return df, scaler

    if scaler is None:
        if method == "minmax":
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == "standardize":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid config key: normalization.method. Allowed values: minmax, standardize, none")
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler
