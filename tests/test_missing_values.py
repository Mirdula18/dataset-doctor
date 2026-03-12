"""Tests for missing-value detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_doctor.diagnosis.missing_values import detect_missing_values


class TestDetectMissingValues:
    def test_no_missing(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = detect_missing_values(df)
        assert result == []

    def test_single_column_missing(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        result = detect_missing_values(df)
        assert len(result) == 1
        assert result[0].column == "a"
        assert result[0].missing_count == 1
        assert result[0].missing_percentage == pytest.approx(33.33, abs=0.01)

    def test_multiple_columns_sorted_descending(self) -> None:
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, 4],
            "b": [np.nan, 2, 3, 4],
        })
        result = detect_missing_values(df)
        assert len(result) == 2
        assert result[0].column == "a"
        assert result[1].column == "b"

    def test_threshold_filters(self) -> None:
        df = pd.DataFrame({
            "a": [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10],  # 10% missing
            "b": [np.nan] * 5 + [1, 2, 3, 4, 5],          # 50% missing
        })
        result = detect_missing_values(df, threshold=20.0)
        assert len(result) == 1
        assert result[0].column == "b"

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = detect_missing_values(df)
        assert result == []

    def test_all_missing(self) -> None:
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = detect_missing_values(df)
        assert len(result) == 1
        assert result[0].missing_percentage == 100.0
