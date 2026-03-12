"""Analyzer — runs all diagnosis modules and produces a report."""

from __future__ import annotations

import pandas as pd

from dataset_doctor.core.report import DataQualityReport
from dataset_doctor.diagnosis.missing_values import detect_missing_values
from dataset_doctor.diagnosis.duplicates import detect_duplicates
from dataset_doctor.diagnosis.outliers import detect_outliers
from dataset_doctor.diagnosis.datatype_checker import detect_datatype_issues
from dataset_doctor.diagnosis.correlation_checker import detect_high_correlations
from dataset_doctor.diagnosis.constant_columns import detect_constant_columns
from dataset_doctor.utils.dataframe_loader import DatasetInput, load_dataframe
from dataset_doctor.utils.logging import get_logger

logger = get_logger(__name__)


def analyze(dataset: DatasetInput) -> DataQualityReport:
    """Run every diagnostic check and return a :class:`DataQualityReport`.

    Parameters
    ----------
    dataset : str | os.PathLike | pd.DataFrame
        File path or DataFrame to diagnose.

    Returns
    -------
    DataQualityReport
    """
    df = load_dataframe(dataset)
    logger.info("Analyzing dataset with %d rows and %d columns", len(df), len(df.columns))

    missing = detect_missing_values(df)
    duplicates = detect_duplicates(df)
    outliers = detect_outliers(df)
    dtype_issues = detect_datatype_issues(df)
    correlations = detect_high_correlations(df)
    constant_cols = detect_constant_columns(df)

    report = DataQualityReport(
        row_count=len(df),
        column_count=len(df.columns),
        missing_values=missing,
        duplicates=duplicates,
        outliers=outliers,
        datatype_issues=dtype_issues,
        high_correlations=correlations,
        constant_columns=constant_cols,
    )

    logger.info("Diagnosis complete — %d issue categories found",
                sum([
                    bool(missing),
                    duplicates.duplicate_row_count > 0,
                    bool(outliers),
                    bool(dtype_issues),
                    bool(correlations),
                    bool(constant_cols),
                ]))

    return report
