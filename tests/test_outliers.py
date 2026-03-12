"""Tests for outlier detection."""

from __future__ import annotations

import pandas as pd
import pytest

from dataset_doctor.diagnosis.outliers import detect_outliers


class TestDetectOutliers:
    def test_no_outliers(self) -> None:
        df = pd.DataFrame({"a": list(range(100))})
        result = detect_outliers(df)
        assert result == []

    def test_detects_outlier(self) -> None:
        values = list(range(100)) + [10_000]
        df = pd.DataFrame({"a": values})
        result = detect_outliers(df)
        assert len(result) == 1
        assert result[0].column == "a"
        assert result[0].outlier_count >= 1

    def test_bounds_are_correct(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        result = detect_outliers(df)
        # tight data, no outliers expected
        assert result == []

    def test_multiple_columns(self) -> None:
        normal = list(range(100))
        with_outlier = list(range(100)) + [9999]
        df = pd.DataFrame({"a": normal + [50], "b": with_outlier})
        result = detect_outliers(df)
        columns = {r.column for r in result}
        assert "b" in columns

    def test_ignores_non_numeric(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = detect_outliers(df)
        assert result == []

    def test_custom_factor(self) -> None:
        values = list(range(50)) + [200]
        df = pd.DataFrame({"a": values})
        strict = detect_outliers(df, factor=0.5)
        lenient = detect_outliers(df, factor=5.0)
        assert len(strict) >= len(lenient)
