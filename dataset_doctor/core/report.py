"""DataQualityReport — structured output of the diagnosis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataset_doctor.diagnosis.missing_values import MissingValueResult
from dataset_doctor.diagnosis.duplicates import DuplicateResult
from dataset_doctor.diagnosis.outliers import OutlierResult
from dataset_doctor.diagnosis.datatype_checker import DatatypeIssue
from dataset_doctor.diagnosis.correlation_checker import CorrelationPair


@dataclass
class DataQualityReport:
    """Aggregated data-quality report returned by :func:`dataset_doctor.diagnose`."""

    row_count: int
    column_count: int
    missing_values: list[MissingValueResult] = field(default_factory=list)
    duplicates: DuplicateResult = field(
        default_factory=lambda: DuplicateResult(0, [])
    )
    outliers: list[OutlierResult] = field(default_factory=list)
    datatype_issues: list[DatatypeIssue] = field(default_factory=list)
    high_correlations: list[CorrelationPair] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable Markdown summary of the report."""
        lines: list[str] = []
        lines.append("## DATASET DIAGNOSIS REPORT\n")
        lines.append(f"Rows: {self.row_count:,}")
        lines.append(f"Columns: {self.column_count:,}")
        lines.append("")

        has_issues = False

        # Missing values
        if self.missing_values:
            has_issues = True
            lines.append("### Issues Detected\n")
            lines.append("**Missing Values**")
            for mv in self.missing_values:
                lines.append(f"  - {mv.column} ({mv.missing_percentage}%)")
            lines.append("")

        # Duplicate rows
        if self.duplicates.duplicate_row_count > 0:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Duplicate Rows**")
            lines.append(f"  - {self.duplicates.duplicate_row_count} rows")
            lines.append("")

        # Duplicate columns
        if self.duplicates.duplicate_column_groups:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Duplicate Columns**")
            for group in self.duplicates.duplicate_column_groups:
                lines.append(f"  - {' vs '.join(group)}")
            lines.append("")

        # Outliers
        if self.outliers:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Outliers**")
            for o in self.outliers:
                lines.append(f"  - {o.column} ({o.outlier_count} values)")
            lines.append("")

        # Datatype issues
        if self.datatype_issues:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Datatype Issues**")
            for dt in self.datatype_issues:
                lines.append(
                    f"  - {dt.column}: {dt.current_dtype} -> {dt.suggested_dtype}"
                )
            lines.append("")

        # High correlations
        if self.high_correlations:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Highly Correlated Columns**")
            for cp in self.high_correlations:
                lines.append(
                    f"  - {cp.column_a} vs {cp.column_b} ({cp.correlation})"
                )
            lines.append("")

        # Constant columns
        if self.constant_columns:
            if not has_issues:
                lines.append("### Issues Detected\n")
                has_issues = True
            lines.append("**Constant Columns**")
            for col in self.constant_columns:
                lines.append(f"  - {col}")
            lines.append("")

        if not has_issues:
            lines.append("No issues detected. Dataset looks clean!")
            lines.append("")

        # Recommendations
        recommendations = self._build_recommendations()
        if recommendations:
            lines.append("### Recommended Fixes\n")
            for rec in recommendations:
                lines.append(f"  \u2714 {rec}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a plain dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_values": [
                {
                    "column": m.column,
                    "missing_count": m.missing_count,
                    "missing_percentage": m.missing_percentage,
                }
                for m in self.missing_values
            ],
            "duplicate_row_count": self.duplicates.duplicate_row_count,
            "duplicate_column_groups": self.duplicates.duplicate_column_groups,
            "outliers": [
                {
                    "column": o.column,
                    "outlier_count": o.outlier_count,
                    "lower_bound": o.lower_bound,
                    "upper_bound": o.upper_bound,
                }
                for o in self.outliers
            ],
            "datatype_issues": [
                {
                    "column": d.column,
                    "current_dtype": d.current_dtype,
                    "suggested_dtype": d.suggested_dtype,
                }
                for d in self.datatype_issues
            ],
            "high_correlations": [
                {
                    "column_a": c.column_a,
                    "column_b": c.column_b,
                    "correlation": c.correlation,
                }
                for c in self.high_correlations
            ],
            "constant_columns": self.constant_columns,
        }

    def print_report(self) -> None:
        """Print the summary to stdout using *rich* if available."""
        try:
            from rich.console import Console
            from rich.markdown import Markdown

            console = Console()
            console.print(Markdown(self.summary()))
        except ImportError:
            print(self.summary())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_recommendations(self) -> list[str]:
        recs: list[str] = []
        if self.missing_values:
            recs.append("Fill missing values (median / mode)")
        if self.duplicates.duplicate_row_count > 0:
            recs.append("Remove duplicate rows")
        if self.outliers:
            recs.append("Handle outliers (clip to IQR bounds)")
        if self.constant_columns:
            recs.append("Drop constant columns")
        if self.high_correlations:
            recs.append("Review or drop highly correlated features")
        if self.datatype_issues:
            recs.append("Fix incorrect data types")
        return recs
