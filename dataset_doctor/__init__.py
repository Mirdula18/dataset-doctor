"""dataset-doctor — Automatically diagnose and clean messy datasets."""

from __future__ import annotations

from dataset_doctor.core.analyzer import analyze as diagnose
from dataset_doctor.core.cleaner import clean as auto_fix
from dataset_doctor.core.report import DataQualityReport
from dataset_doctor.core.viewer import display_data

__version__ = "0.1.0"
__all__ = ["diagnose", "auto_fix", "display_data", "DataQualityReport"]
