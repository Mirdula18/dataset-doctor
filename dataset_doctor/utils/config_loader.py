"""Configuration loading and validation for dataset-doctor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "missing_values": {
        "numeric_strategy": "median",
        "categorical_strategy": "mode",
        "max_missing_threshold": 0.5,
        "constant_value": None,
    },
    "duplicates": {
        "remove": True,
    },
    "outliers": {
        "method": "iqr",
        "action": "clip",
        "iqr_factor": 1.5,
        "zscore_threshold": 3.0,
    },
    "normalization": {
        "method": "minmax",
        "range": [0, 1],
    },
    "feature_selection": {
        "remove_constant_columns": True,
        "correlation_threshold": 0.9,
    },
    "logging": {
        "verbosity": "medium",
    },
}


_ALLOWED_TOP_LEVEL_KEYS = set(DEFAULT_CONFIG.keys())


def _deep_copy_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively copy mapping values into a plain dict."""
    copied: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            copied[key] = _deep_copy_dict(item)
        elif isinstance(item, list):
            copied[key] = list(item)
        else:
            copied[key] = item
    return copied


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* and return *base*."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return dict(base)


def merge_with_defaults(user_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge user config with default values."""
    merged = _deep_copy_dict(DEFAULT_CONFIG)
    if not user_config:
        return merged
    return _deep_merge(merged, user_config)


def _raise_invalid_key(path: str, allowed: list[str]) -> None:
    joined = ", ".join(allowed)
    raise ValueError(f"Invalid config key: {path}. Allowed values: {joined}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate structure and allowed values for config data."""
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping/dictionary.")

    for key, value in config.items():
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            _raise_invalid_key(key, sorted(_ALLOWED_TOP_LEVEL_KEYS))
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid config value for '{key}': expected a mapping.")

    missing_values = config.get("missing_values", {})
    allowed_missing_keys = {
        "numeric_strategy",
        "categorical_strategy",
        "max_missing_threshold",
        "constant_value",
    }
    for key in missing_values:
        if key not in allowed_missing_keys:
            _raise_invalid_key(f"missing_values.{key}", sorted(allowed_missing_keys))

    allowed_missing_strategies = ["constant", "mean", "median", "mode"]
    for strategy_key in ["numeric_strategy", "categorical_strategy"]:
        strategy = missing_values.get(strategy_key)
        if strategy is not None and strategy not in allowed_missing_strategies:
            _raise_invalid_key(
                f"missing_values.{strategy_key}",
                allowed_missing_strategies,
            )

    threshold = missing_values.get("max_missing_threshold")
    if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1):
        raise ValueError(
            "Invalid config key: missing_values.max_missing_threshold. Allowed values: float between 0 and 1"
        )

    duplicates = config.get("duplicates", {})
    allowed_duplicates_keys = {"remove"}
    for key in duplicates:
        if key not in allowed_duplicates_keys:
            _raise_invalid_key(f"duplicates.{key}", sorted(allowed_duplicates_keys))
    if "remove" in duplicates and not isinstance(duplicates["remove"], bool):
        raise ValueError("Invalid config key: duplicates.remove. Allowed values: true, false")

    outliers = config.get("outliers", {})
    allowed_outlier_keys = {"method", "action", "iqr_factor", "zscore_threshold"}
    for key in outliers:
        if key not in allowed_outlier_keys:
            _raise_invalid_key(f"outliers.{key}", sorted(allowed_outlier_keys))

    if "method" in outliers and outliers["method"] not in ["iqr", "zscore"]:
        _raise_invalid_key("outliers.method", ["iqr", "zscore"])
    if "action" in outliers and outliers["action"] not in ["clip", "remove"]:
        _raise_invalid_key("outliers.action", ["clip", "remove"])
    if "iqr_factor" in outliers and (
        not isinstance(outliers["iqr_factor"], (int, float)) or outliers["iqr_factor"] <= 0
    ):
        raise ValueError("Invalid config key: outliers.iqr_factor. Allowed values: positive float")
    if "zscore_threshold" in outliers and (
        not isinstance(outliers["zscore_threshold"], (int, float)) or outliers["zscore_threshold"] <= 0
    ):
        raise ValueError("Invalid config key: outliers.zscore_threshold. Allowed values: positive float")

    normalization = config.get("normalization", {})
    allowed_normalization_keys = {"method", "range"}
    for key in normalization:
        if key not in allowed_normalization_keys:
            _raise_invalid_key(f"normalization.{key}", sorted(allowed_normalization_keys))
    if "method" in normalization and normalization["method"] not in ["minmax", "none", "standardize"]:
        _raise_invalid_key("normalization.method", ["minmax", "standardize", "none"])
    if "range" in normalization:
        range_value = normalization["range"]
        if (
            not isinstance(range_value, list)
            or len(range_value) != 2
            or not all(isinstance(x, (int, float)) for x in range_value)
            or range_value[0] >= range_value[1]
        ):
            raise ValueError(
                "Invalid config key: normalization.range. Allowed values: [min, max] where min < max"
            )

    feature_selection = config.get("feature_selection", {})
    allowed_feature_keys = {"remove_constant_columns", "correlation_threshold"}
    for key in feature_selection:
        if key not in allowed_feature_keys:
            _raise_invalid_key(f"feature_selection.{key}", sorted(allowed_feature_keys))
    if "remove_constant_columns" in feature_selection and not isinstance(
        feature_selection["remove_constant_columns"],
        bool,
    ):
        raise ValueError(
            "Invalid config key: feature_selection.remove_constant_columns. Allowed values: true, false"
        )
    if "correlation_threshold" in feature_selection and (
        not isinstance(feature_selection["correlation_threshold"], (int, float))
        or feature_selection["correlation_threshold"] < 0
        or feature_selection["correlation_threshold"] > 1
    ):
        raise ValueError(
            "Invalid config key: feature_selection.correlation_threshold. Allowed values: float between 0 and 1"
        )

    logging_cfg = config.get("logging", {})
    allowed_logging_keys = {"verbosity"}
    for key in logging_cfg:
        if key not in allowed_logging_keys:
            _raise_invalid_key(f"logging.{key}", sorted(allowed_logging_keys))
    if "verbosity" in logging_cfg and logging_cfg["verbosity"] not in ["low", "medium", "high"]:
        _raise_invalid_key("logging.verbosity", ["low", "medium", "high"])


def load_config(config_path: Union[None, str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    """Load, validate, and merge config with defaults.

    Parameters
    ----------
    config_path : None | str | Path | Mapping[str, Any]
        ``None`` -> defaults.
        ``str``/``Path`` -> YAML file path.
        ``dict`` -> already loaded user config.
    """
    if config_path is None:
        return _deep_copy_dict(DEFAULT_CONFIG)

    user_config: Mapping[str, Any]
    if isinstance(config_path, Mapping):
        user_config = config_path
    else:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("Configuration file must contain a YAML mapping.")
        user_config = loaded

    validate_config(user_config)
    merged = merge_with_defaults(user_config)
    validate_config(merged)
    return merged