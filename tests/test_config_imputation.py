"""Tests for config-driven cleaning behavior."""

from __future__ import annotations

import pandas as pd
import pytest

import dataset_doctor as dd


def test_auto_fix_with_default_config_still_works() -> None:
    df = pd.DataFrame(
        {
            "num": [1.0, None, 3.0],
            "cat": ["a", None, "b"],
        }
    )

    result = dd.auto_fix(df)
    assert len(result) == 3
    assert result["num"].isna().sum() == 0
    assert result["cat"].isna().sum() == 0


def test_yaml_config_overrides_missing_strategy(tmp_path) -> None:
    data_path = tmp_path / "input.csv"
    cfg_path = tmp_path / "config.yaml"

    pd.DataFrame({"x": [1.0, None, 5.0]}).to_csv(data_path, index=False)
    cfg_path.write_text(
        "\n".join(
            [
                "missing_values:",
                "  numeric_strategy: mean",
                "normalization:",
                "  method: none",
            ]
        ),
        encoding="utf-8",
    )

    result = dd.auto_fix(str(data_path), config=str(cfg_path))
    assert result.loc[1, "x"] == pytest.approx(3.0)


def test_invalid_config_raises_helpful_error() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 1000.0]})
    with pytest.raises(ValueError, match="Invalid config key"):
        dd.auto_fix(df, config={"outliers": {"method": "bad_method"}})
