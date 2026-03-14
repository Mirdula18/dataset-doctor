"""Cleaner — applies the automatic cleaning pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataset_doctor.cleaning.remove_duplicates import remove_duplicates
from dataset_doctor.cleaning.fill_missing import fill_missing
from dataset_doctor.cleaning.outlier_handler import handle_outliers
from dataset_doctor.cleaning.drop_constant import drop_constant_columns
from dataset_doctor.cleaning.normalize import normalize
from dataset_doctor.utils.dataframe_loader import DatasetInput, load_dataframe
from dataset_doctor.utils.config_loader import load_config
from dataset_doctor.utils.logging import get_logger

logger = get_logger(__name__)


def clean(
    dataset: DatasetInput,
    *,
    output_path: Optional[str] = None,
    output: Optional[str] = None,
    do_normalize: bool = False,
    return_scaler: bool = False,
    config: Optional[Union[str, Path, Mapping[str, Any]]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
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
    output : str, optional
        Backward-compatible alias for *output_path*.
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
    config : str | Path | dict, optional
        Cleaning configuration. If ``None``, sensible defaults are used.
        If a path is provided, the YAML file is loaded.
        If a dict is provided, it is merged with defaults.
    """
    if output_path is None and output is not None:
        output_path = output

    df = load_dataframe(dataset)
    logger.info("Cleaning dataset with %d rows and %d columns", len(df), len(df.columns))

    resolved_config = load_config(config)
    user_provided_config = config is not None

    if resolved_config["duplicates"]["remove"]:
        df = remove_duplicates(df)
        logger.info("Step 1/5 — duplicates removed")
    else:
        logger.info("Step 1/5 — duplicate removal skipped by config")

    df = fill_missing(df, config=resolved_config["missing_values"])
    logger.info("Step 2/5 — missing values filled")

    df = handle_outliers(df, config=resolved_config["outliers"])
    logger.info("Step 3/5 — outliers handled")

    if resolved_config["feature_selection"]["remove_constant_columns"]:
        df = drop_constant_columns(df)
        logger.info("Step 4/5 — constant columns dropped")
    else:
        logger.info("Step 4/5 — constant column removal skipped by config")

    scaler: Optional[Any] = None
    norm_method = str(resolved_config["normalization"]["method"])
    norm_range = tuple(resolved_config["normalization"]["range"])

    if do_normalize and norm_method == "none":
        norm_method = "minmax"

    should_normalize = do_normalize or (user_provided_config and norm_method != "none")

    if should_normalize and norm_method != "none":
        df, scaler = normalize(df, method=norm_method, feature_range=norm_range)
        logger.info("Step 5/5 — numeric features normalized (%s)", norm_method)
    else:
        logger.info("Step 5/5 — normalization skipped (use do_normalize=True to enable)")

    if output_path is not None:
        out = Path(output_path)
        df.to_csv(out, index=False)
        logger.info("Cleaned dataset saved to %s", out)

    if return_scaler:
        return df, scaler or MinMaxScaler()

    return df
