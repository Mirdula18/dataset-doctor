# dataset-doctor

**Automatically diagnose and clean messy datasets for machine learning and data science.**

`dataset-doctor` analyzes your data and detects common quality issues — missing values, duplicates, outliers, type mismatches, constant columns, and highly correlated features — then recommends (or automatically applies) fixes so you can get to modeling faster.

---

## Features

- **Missing-value detection** — counts and percentages per column
- **Duplicate detection** — duplicate rows *and* duplicate columns
- **Outlier detection** — IQR-based flagging for numeric columns
- **Data-type checking** — finds string columns that should be numeric or datetime
- **Correlation analysis** — flags highly correlated feature pairs (|r| ≥ 0.9)
- **Constant-column detection** — identifies zero-variance columns
- **Automatic cleaning pipeline** — one-call fix: dedup → fill → clip → drop → normalize
- **Rich CLI** — terminal commands with formatted output
- **Extensible architecture** — modular diagnosis and cleaning components

---

## Installation

```bash
pip install dataset-doctor
```

Or install from source:

```bash
git clone https://github.com/dataset-doctor/dataset-doctor.git
cd dataset-doctor
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Python Usage

### Diagnose a dataset

```python
import dataset_doctor as dd

# From a file path
report = dd.diagnose("data.csv")
print(report.summary())

# From a pandas DataFrame
import pandas as pd

df = pd.read_csv("data.csv")
report = dd.diagnose(df)
report.print_report()          # rich-formatted output
print(report.to_dict())        # machine-readable dict
```

### Auto-fix a dataset

```python
import dataset_doctor as dd

# Clean and get back a DataFrame
clean_df = dd.auto_fix("data.csv")

# Clean with options
clean_df = dd.auto_fix(
    "data.csv",
    output_path="cleaned.csv",   # save to disk
    do_normalize=True,            # apply Min-Max scaling
)

# Preview rows/columns for quick inspection
preview_df = dd.display_data("data.csv", rows=5)
print(preview_df)
```

### Example diagnosis output

```
## DATASET DIAGNOSIS REPORT

Rows: 10,000
Columns: 12

### Issues Detected

**Missing Values**
  - age (12.0%)
  - salary (4.0%)

**Duplicate Rows**
  - 18 rows

**Outliers**
  - transaction_amount (42 values)

**Constant Columns**
  - user_flag

**Highly Correlated Columns**
  - income vs salary (0.97)

### Recommended Fixes

  ✔ Fill missing values (median / mode)
  ✔ Remove duplicate rows
  ✔ Handle outliers (clip to IQR bounds)
  ✔ Drop constant columns
  ✔ Review or drop highly correlated features
```

---

## CLI Usage

`dataset-doctor` ships with a command-line interface powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).

```bash
# Diagnose a dataset
dataset-doctor diagnose data.csv

# Print the full report (alias for diagnose)
dataset-doctor report data.csv

# Clean a dataset
dataset-doctor clean data.csv --output cleaned.csv

# Clean with normalization
dataset-doctor clean data.csv --output cleaned.csv --normalize

# Display first 10 rows (default)
dataset-doctor display data.csv

# Display selected columns and last 20 rows
dataset-doctor show data.csv --tail --rows 20 --columns age,salary,city
```

---

## Architecture

```
dataset_doctor/
├── __init__.py              # Public API: diagnose(), auto_fix()
├── cli.py                   # Typer CLI application
│
├── core/
│   ├── analyzer.py          # Orchestrates all diagnosis modules
│   ├── cleaner.py           # Orchestrates the cleaning pipeline
│   ├── viewer.py            # Produces DataFrame views for display
│   └── report.py            # DataQualityReport dataclass
│
├── diagnosis/
│   ├── missing_values.py    # Missing-value detection
│   ├── duplicates.py        # Duplicate row & column detection
│   ├── outliers.py          # IQR-based outlier detection
│   ├── datatype_checker.py  # Dtype mismatch detection
│   ├── correlation_checker.py  # High-correlation detection
│   └── constant_columns.py  # Constant-column detection
│
├── cleaning/
│   ├── fill_missing.py      # Median / mode imputation
│   ├── remove_duplicates.py # Drop duplicate rows
│   ├── normalize.py         # Min-Max scaling
│   ├── outlier_handler.py   # IQR clipping
│   └── drop_constant.py     # Drop zero-variance columns
│
└── utils/
    ├── dataframe_loader.py  # CSV / DataFrame input handling
    └── logging.py           # Logging configuration
```

**Design principles:**

| Principle | How it's applied |
|-----------|-----------------|
| Single Responsibility | Each diagnosis / cleaning task is its own module |
| Open/Closed | Add new checks by creating a module — no existing code changes |
| Dependency Inversion | Core engines depend on abstractions, not concrete I/O |
| Vectorized ops | All pandas operations avoid Python-level loops |

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=dataset_doctor
```

---


## Requirements

- Python ≥ 3.9
- pandas ≥ 1.5
- numpy ≥ 1.23
- scikit-learn ≥ 1.1
- rich ≥ 12.0
- typer ≥ 0.9

---

## License

MIT — see [LICENSE](LICENSE) for details.
