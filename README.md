# dataset-doctor

![PyPI](https://img.shields.io/pypi/v/dataset-doctor)
![Python Versions](https://img.shields.io/pypi/pyversions/dataset-doctor)
![License](https://img.shields.io/pypi/l/dataset-doctor)
![CI](https://img.shields.io/github/actions/workflow/status/dataset-doctor/dataset-doctor/ci.yml?branch=main)

Automatic dataset diagnosis and cleaning for machine learning and data science.

## Project Overview

dataset-doctor is an open-source Python package for detecting and fixing common data quality issues in tabular datasets.

The project exists to reduce repetitive preprocessing work before feature engineering and model training. In many ML projects, teams lose time on the same recurring tasks: checking missing values, dropping duplicates, handling outliers, and standardizing numeric columns. dataset-doctor wraps these tasks into a clear diagnosis report and a configurable cleaning pipeline.

This improves ML workflow reliability by making dataset checks explicit, reproducible, and easy to automate in scripts or CI pipelines.

## Features

- Multi-check dataset diagnosis with structured report output
- Missing-value analysis and configurable imputation strategies
- Duplicate row and duplicate column detection
- Outlier detection and handling (IQR or z-score)
- Constant-column removal
- Optional normalization (min-max or standardization)
- YAML-based preprocessing configuration
- User-friendly command-line interface
- Python API for notebook and pipeline integration

## Installation

Install from PyPI:

```bash
pip install dataset-doctor
```

Install from source:

```bash
git clone https://github.com/Mirdula18/dataset-doctor.git
cd dataset-doctor
pip install .
```

Development install:

```bash
pip install -e ".[dev]"
```

## Usage

### CLI Usage

Diagnose:

```bash
dataset-doctor diagnose data.csv
```

Full report alias:

```bash
dataset-doctor report data.csv
```

Clean dataset:

```bash
dataset-doctor clean data.csv
```

Clean with output path:

```bash
dataset-doctor clean data.csv --output cleaned.csv
```

Clean with normalization:

```bash
dataset-doctor clean data.csv --normalize
```

Clean with config:

```bash
dataset-doctor clean data.csv --config dataset_doctor_config.yaml
```

Generate default config file:

```bash
dataset-doctor init-config
```

Display rows:

```bash
dataset-doctor display data.csv --rows 10
```

Show rows (alias):

```bash
dataset-doctor show data.csv --tail --rows 20 --columns age,salary,city
```

### Python API

```python
import dataset_doctor as dd

# Diagnose a dataset
report = dd.diagnose("data.csv")
print(report.summary())

# Auto-clean with defaults
clean_df = dd.auto_fix("data.csv")

# Auto-clean with options
clean_df = dd.auto_fix(
    "data.csv",
    output_path="cleaned.csv",
    do_normalize=True,
)

# Auto-clean with YAML config
clean_df = dd.auto_fix("data.csv", config="dataset_doctor_config.yaml")

# Display data preview
preview = dd.display_data("data.csv", rows=5)
print(preview)
```

```bash
import dataset_doctor as dd

dd.diagnose("data.csv")
dd.auto_fix("data.csv")
dd.auto_fix("data.csv", output_path="cleaned.csv")
dd.auto_fix("data.csv", output="cleaned.csv")
dd.auto_fix("data.csv", do_normalize=True)
dd.auto_fix("data.csv", return_scaler=True)
dd.auto_fix("data.csv", config="dataset_doctor_config.yaml")
dd.auto_fix("data.csv", config={"missing_values": {"numeric_strategy": "mean"}})
dd.display_data("data.csv")
dd.display_data("data.csv", rows=10)
dd.display_data("data.csv", tail=True)
dd.display_data("data.csv", columns=["col1", "col2"])
dd.display_data("data.csv", all_rows=True)

report = dd.diagnose("data.csv")
report.summary()
report.to_dict()
report.print_report()
```

## Architecture Overview

High-level package structure:

```text
dataset_doctor/
  diagnosis/
  cleaning/
  core/
  utils/
```

Responsibilities:

- diagnosis
  - Contains issue detectors (missing values, duplicates, outliers, data types, correlations, constant columns).
- cleaning
  - Contains data-fixing transforms (imputation, duplicate removal, outlier handling, normalization, constant-column dropping).
- core
  - Orchestrates diagnosis and cleaning workflows, assembles report output, and exposes end-user pipeline behavior.
- utils
  - Shared helpers for logging, DataFrame loading, and YAML configuration loading/validation.

## Configuration System

dataset-doctor supports YAML-based configuration to control preprocessing behavior without code changes.

Generate default config:

```bash
dataset-doctor init-config
```

Use config from CLI:

```bash
dataset-doctor clean data.csv --config dataset_doctor_config.yaml
```

Use config from Python:

```python
import dataset_doctor as dd

clean_df = dd.auto_fix("data.csv", config="dataset_doctor_config.yaml")
```

Example configuration:

```yaml
missing_values:
  numeric_strategy: median
  categorical_strategy: mode
  max_missing_threshold: 0.4

duplicates:
  remove: true

outliers:
  method: iqr
  action: clip

normalization:
  method: minmax
  range: [0, 1]

feature_selection:
  remove_constant_columns: true
  correlation_threshold: 0.9

logging:
  verbosity: medium
```

## Example Workflow

Typical ML data preparation flow:

1. Collect raw dataset
2. Run diagnosis with dataset-doctor
3. Review summary and issue categories
4. Auto-clean with defaults or custom YAML config
5. Export cleaned dataset
6. Train and evaluate ML model

```text
raw_data.csv -> diagnose -> report -> auto_fix(config) -> cleaned_data.csv -> model training
```

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests and documentation updates
4. Open a pull request with a clear description of scope and rationale

Please keep pull requests focused and include reproducible examples for bug fixes.

## Development Setup

Editable install with dev extras:

```bash
pip install -e ".[dev]"
```

Alternative workflow if your fork includes a dev requirements file:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=dataset_doctor
```

## Roadmap

Planned and potential enhancements:

- HTML diagnosis reports for sharing and archival
- Feature engineering helpers (encoding and transformations)
- Dataset health scoring framework
- Visualization dashboard for issue exploration
- Extended schema and expectation checks

## License

MIT License. See LICENSE for full text.
