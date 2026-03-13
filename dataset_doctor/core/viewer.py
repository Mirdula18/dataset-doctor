"""Viewer utilities for displaying dataset samples."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from dataset_doctor.utils.dataframe_loader import DatasetInput, load_dataframe
from dataset_doctor.utils.logging import get_logger

logger = get_logger(__name__)


def display_data(
    dataset: DatasetInput,
    *,
    rows: int = 10,
    tail: bool = False,
    columns: Optional[List[str]] = None,
    all_rows: bool = False,
) -> pd.DataFrame:
    """Return a view of the dataset for display purposes.

    Parameters
    ----------
    dataset : str | os.PathLike | pd.DataFrame
        File path or DataFrame to display.
    rows : int
        Number of rows to show when *all_rows* is ``False``.
    tail : bool
        If ``True``, return the last *rows* rows; otherwise return the first.
    columns : list[str], optional
        Optional subset of columns to include.
    all_rows : bool
        If ``True``, return all rows.

    Returns
    -------
    pd.DataFrame
        A DataFrame slice intended for display.
    """
    if rows <= 0:
        raise ValueError("rows must be a positive integer")

    df = load_dataframe(dataset)

    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {', '.join(missing)}")
        df = df[columns]

    if all_rows:
        result = df
    elif tail:
        result = df.tail(rows)
    else:
        result = df.head(rows)

    logger.info(
        "Prepared data view with %d rows and %d columns",
        len(result),
        len(result.columns),
    )

    return result
