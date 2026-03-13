"""Tests for dataset display helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from dataset_doctor.core.viewer import display_data


class TestDisplayData:
    def test_head_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        result = display_data(df, rows=2)
        assert len(result) == 2
        assert result["a"].tolist() == [1, 2]

    def test_tail_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        result = display_data(df, rows=2, tail=True)
        assert result["a"].tolist() == [3, 4]

    def test_column_subset(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = display_data(df, columns=["a", "c"])
        assert result.columns.tolist() == ["a", "c"]

    def test_all_rows_ignores_rows_limit(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = display_data(df, rows=1, all_rows=True)
        assert len(result) == 3

    def test_rows_must_be_positive(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="rows must be a positive integer"):
            display_data(df, rows=0)

    def test_unknown_columns_raise(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown column"):
            display_data(df, columns=["missing_col"])
