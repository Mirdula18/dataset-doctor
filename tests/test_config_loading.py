"""Tests for YAML config loading and validation."""

from __future__ import annotations

import pytest

from dataset_doctor.utils.config_loader import load_config


def test_load_config_defaults_when_none() -> None:
    config = load_config(None)
    assert config["missing_values"]["numeric_strategy"] == "median"
    assert config["outliers"]["method"] == "iqr"
    assert config["normalization"]["method"] == "minmax"


def test_yaml_overrides_merge_with_defaults(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "missing_values:",
                "  numeric_strategy: mean",
                "outliers:",
                "  method: zscore",
                "  action: remove",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(cfg)
    assert config["missing_values"]["numeric_strategy"] == "mean"
    assert config["missing_values"]["categorical_strategy"] == "mode"
    assert config["outliers"]["method"] == "zscore"
    assert config["outliers"]["action"] == "remove"
    assert config["feature_selection"]["remove_constant_columns"] is True


def test_invalid_config_key_raises() -> None:
    with pytest.raises(ValueError, match="Invalid config key"):
        load_config({"missing_values": {"unsupported": "foo"}})
