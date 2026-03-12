"""Tests for duplicate detection."""

from __future__ import annotations

import pandas as pd

from dataset_doctor.diagnosis.duplicates import detect_duplicates


class TestDetectDuplicates:
    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = detect_duplicates(df)
        assert result.duplicate_row_count == 0
        assert result.duplicate_column_groups == []

    def test_duplicate_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 1], "b": [4, 5, 4]})
        result = detect_duplicates(df)
        assert result.duplicate_row_count == 1

    def test_multiple_duplicate_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1, 2], "b": [10, 10, 10, 20]})
        result = detect_duplicates(df)
        assert result.duplicate_row_count == 2

    def test_duplicate_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        result = detect_duplicates(df)
        assert len(result.duplicate_column_groups) == 1
        assert set(result.duplicate_column_groups[0]) == {"a", "b"}

    def test_no_duplicate_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = detect_duplicates(df)
        assert result.duplicate_column_groups == []

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = detect_duplicates(df)
        assert result.duplicate_row_count == 0
