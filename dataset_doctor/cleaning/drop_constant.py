"""Drop constant (zero-variance) columns."""

from __future__ import annotations

import pandas as pd

from dataset_doctor.diagnosis.constant_columns import detect_constant_columns


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that have at most one unique value.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    constant_cols = detect_constant_columns(df)
    if constant_cols:
        return df.drop(columns=constant_cols)
    return df.copy()
