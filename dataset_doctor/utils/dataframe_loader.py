"""Utilities for loading data into pandas DataFrames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd

DatasetInput = Union[str, os.PathLike, pd.DataFrame]


def load_dataframe(dataset: DatasetInput) -> pd.DataFrame:
    """Load a dataset from a file path or return the DataFrame directly.

    Parameters
    ----------
    dataset : str | os.PathLike | pd.DataFrame
        A file path (CSV) or an existing pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file format is unsupported or the input type is invalid.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Supported formats: .csv, .xlsx, .parquet, .json"
        )
