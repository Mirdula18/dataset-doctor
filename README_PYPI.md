# dataset-doctor

## Automatic Dataset Diagnosis and Cleaning for Machine Learning

dataset-doctor helps you quickly identify and fix common dataset quality problems before model training.

It is built for:
- Data scientists preparing tabular data for experiments
- ML engineers standardizing preprocessing workflows
- Beginners who want safer defaults for dataset cleaning

Instead of manually writing repeated preprocessing code, you can diagnose data issues and run an automatic cleaning pipeline from either Python or the command line.

## Features

- Dataset diagnosis with a readable summary report
- Missing value detection and imputation
- Duplicate row detection and removal
- Outlier detection and handling
- Constant column detection and removal
- Optional normalization for numeric columns
- YAML-based configuration system for preprocessing behavior
- CLI commands for diagnosis, cleaning, display, and config generation

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

## Quick Example

```python
import dataset_doctor as dd

report = dd.diagnose("data.csv")
print(report.summary())

clean_df = dd.auto_fix("data.csv")
```

## CLI Usage

Diagnose a dataset:

```bash
dataset-doctor diagnose data.csv
```

Print report output (alias of diagnose):

```bash
dataset-doctor report data.csv
```

Clean a dataset:

```bash
dataset-doctor clean data.csv
```

Clean and write output file:

```bash
dataset-doctor clean data.csv --output cleaned.csv
```

Enable normalization from CLI:

```bash
dataset-doctor clean data.csv --normalize
```

Clean using a YAML config file:

```bash
dataset-doctor clean data.csv --config dataset_doctor_config.yaml
```

Generate a default config file:

```bash
dataset-doctor init-config
```

Display rows:

```bash
dataset-doctor display data.csv --rows 10
```

Show rows (alias of display):

```bash
dataset-doctor show data.csv --tail --rows 20 --columns age,salary
```

## Python API

Main API entry points:
- dd.diagnose(dataset)
- dd.auto_fix(dataset, ...)
- dd.display_data(dataset, ...)

## Example Usecase

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

Example:

```python
import dataset_doctor as dd

# Diagnose
report = dd.diagnose("data.csv")
print(report.summary())

# Auto-clean with options
clean_df = dd.auto_fix(
    "data.csv",
    output_path="cleaned.csv",
    do_normalize=True,
)
```

## Configuration System

Use a YAML file to customize preprocessing behavior.

CLI example:

```bash
dataset-doctor clean data.csv --config config.yaml
```

Python example:

```python
import dataset_doctor as dd

clean_df = dd.auto_fix("data.csv", config="config.yaml")
```

Example config:

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

## Output Example

```text
## DATASET DIAGNOSIS REPORT

Rows: 10000
Columns: 12

### Issues Detected

Missing Values:
- age (12.0%)
- salary (4.0%)

Duplicate Rows:
- 18 rows

Outliers:
- transaction_amount (42 values)

Constant Columns:
- user_flag

Highly Correlated Columns:
- income vs salary (0.97)
```

## License

MIT License. See LICENSE for details.
