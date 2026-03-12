"""Cleaner — applies the automatic cleaning pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataset_doctor.cleaning.remove_duplicates import remove_duplicates
from dataset_doctor.cleaning.fill_missing import fill_missing
from dataset_doctor.cleaning.outlier_handler import handle_outliers
from dataset_doctor.cleaning.drop_constant import drop_constant_columns
from dataset_doctor.cleaning.normalize import normalize
from dataset_doctor.utils.dataframe_loader import DatasetInput, load_dataframe
from dataset_doctor.utils.logging import get_logger

logger = get_logger(__name__)


def clean(
    dataset: DatasetInput,
    *,
    output_path: Optional[str] = None,
    do_normalize: bool = False,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, MinMaxScaler]]:
    """Run the full cleaning pipeline and return the cleaned DataFrame.

    Pipeline order:
        1. Remove duplicate rows
        2. Fill missing values
        3. Handle outliers (IQR clip)
        4. Drop constant columns
        5. Normalize numeric features (optional)

    Parameters
    ----------
    dataset : str | os.PathLike | pd.DataFrame
        File path or DataFrame.
    output_path : str, optional
        If provided, save the cleaned DataFrame as CSV.
    do_normalize : bool
        Whether to apply Min-Max normalization (default ``False``).
    return_scaler : bool
        If ``True``, return a ``(DataFrame, MinMaxScaler)`` tuple instead of
        just the DataFrame. The scaler is fitted on the cleaned data and can
        be reused to transform a held-out test set without data leakage.
        Only meaningful when *do_normalize* is ``True``; when normalization
        is skipped an **unfitted** scaler is returned.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, MinMaxScaler]
        The cleaned DataFrame, or ``(cleaned_df, fitted_scaler)`` when
        *return_scaler* is ``True``.
    """
    df = load_dataframe(dataset)
    logger.info("Cleaning dataset with %d rows and %d columns", len(df), len(df.columns))

    df = remove_duplicates(df)
    logger.info("Step 1/5 — duplicates removed")

    df = fill_missing(df)
    logger.info("Step 2/5 — missing values filled")

    df = handle_outliers(df)
    logger.info("Step 3/5 — outliers clipped")

    df = drop_constant_columns(df)
    logger.info("Step 4/5 — constant columns dropped")

    scaler: Optional[MinMaxScaler] = None
    if do_normalize:
        df, scaler = normalize(df)
        logger.info("Step 5/5 — numeric features normalized")
    else:
        logger.info("Step 5/5 — normalization skipped (use do_normalize=True to enable)")

    if output_path is not None:
        out = Path(output_path)
        df.to_csv(out, index=False)
        logger.info("Cleaned dataset saved to %s", out)

    if return_scaler:
        return df, scaler or MinMaxScaler()

    return df
