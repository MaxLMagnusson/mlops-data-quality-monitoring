"""
Validation Engine — Evidently AI Data Drift & Quality Checks
=============================================================
The mathematical brain of the pipeline. Compares incoming data batches
against the baseline reference using statistical tests to detect:

1. Data Drift: Distribution shifts in numeric/categorical features
   - Kolmogorov-Smirnov test for continuous variables
   - Chi-squared test for categorical variables
   - Per-column p-value thresholds

2. Data Quality: Schema and value integrity checks
   - Column type validation
   - Missing value thresholds
   - Value range enforcement
   - Duplicate detection
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.test_suite import TestSuite
from evidently.pipeline.column_mapping import ColumnMapping

logger = logging.getLogger(__name__)


# Column classification for Evidently
NUMERICAL_FEATURES = [
    "bbox_width",
    "bbox_height",
    "bbox_length",
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_w",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "num_lidar_pts",
    "num_radar_pts",
]

CATEGORICAL_FEATURES = [
    "category",
    "visibility",
    "sensor_channel",
]


@dataclass
class ValidationResult:
    """
    Structured result of a data validation run.

    Contains drift detection outcomes, data quality test results,
    and severity classification for downstream routing.
    """

    # Overall status
    passed: bool
    status: str  # 'SUCCESS' or 'DRIFT_DETECTED'

    # Drift metrics
    drift_detected: bool
    drift_score: float  # 0.0 - 1.0 (fraction of drifted columns)
    drifted_columns: list[str] = field(default_factory=list)
    drift_details: dict[str, Any] = field(default_factory=dict)

    # Data quality test results
    total_tests: int = 0
    failed_tests: int = 0
    test_results: list[dict[str, Any]] = field(default_factory=list)

    # Severity classification
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL

    # Report paths
    drift_report_html: Optional[str] = None
    quality_report_html: Optional[str] = None

    # Execution metadata
    execution_time_seconds: float = 0.0
    records_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/Postgres storage."""
        return {
            "passed": self.passed,
            "status": self.status,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "drifted_columns": self.drifted_columns,
            "total_tests": self.total_tests,
            "failed_tests": self.failed_tests,
            "severity": self.severity,
            "execution_time_seconds": self.execution_time_seconds,
            "records_processed": self.records_processed,
        }


def _classify_severity(drift_score: float, failed_tests: int, total_tests: int) -> str:
    """
    Classify the severity of detected issues.

    Severity levels:
    - LOW:      < 10% columns drifted, < 5% tests failed
    - MEDIUM:   10-30% columns drifted, 5-15% tests failed
    - HIGH:     30-60% columns drifted, 15-40% tests failed
    - CRITICAL: > 60% columns drifted, or > 40% tests failed
    """
    test_fail_rate = failed_tests / max(total_tests, 1)

    if drift_score > 0.6 or test_fail_rate > 0.4:
        return "CRITICAL"
    elif drift_score > 0.3 or test_fail_rate > 0.15:
        return "HIGH"
    elif drift_score > 0.1 or test_fail_rate > 0.05:
        return "MEDIUM"
    else:
        return "LOW"


def _get_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    """
    Create Evidently ColumnMapping based on available columns.

    Filters the predefined feature lists to only include columns
    actually present in the DataFrame.
    """
    mapping = ColumnMapping()
    mapping.numerical_features = [
        col for col in NUMERICAL_FEATURES if col in df.columns
    ]
    mapping.categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in df.columns
    ]
    return mapping


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[Report, dict[str, Any]]:
    """
    Run Evidently DataDrift report comparing current vs. reference data.

    Returns the Report object and a parsed summary dictionary.
    """
    column_mapping = _get_column_mapping(reference_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report_dict = report.as_dict()

    # Parse drift results
    drift_info = {}
    drifted_columns = []

    for metric_result in report_dict.get("metrics", []):
        result_data = metric_result.get("result", {})
        if "drift_by_columns" in result_data:
            for col_name, col_data in result_data["drift_by_columns"].items():
                drift_info[col_name] = {
                    "drifted": col_data.get("drift_detected", False),
                    "p_value": col_data.get("drift_score", None),
                    "stattest": col_data.get("stattest_name", "unknown"),
                }
                if col_data.get("drift_detected", False):
                    drifted_columns.append(col_name)

    # Calculate overall drift score
    total_cols = max(len(drift_info), 1)
    drift_score = len(drifted_columns) / total_cols

    summary = {
        "drift_score": drift_score,
        "drifted_columns": drifted_columns,
        "total_columns_tested": total_cols,
        "per_column": drift_info,
    }

    return report, summary


def run_data_quality_tests(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[TestSuite, dict[str, Any]]:
    """
    Run Evidently DataQuality test suite.

    Checks for:
    - Schema consistency (column types, names)
    - Missing value thresholds
    - Value range violations
    - Duplicate rows
    """
    column_mapping = _get_column_mapping(reference_df)

    suite = TestSuite(tests=[DataQualityTestPreset()])
    suite.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    results_dict = suite.as_dict()

    # Parse test results
    test_results = []
    failed_count = 0
    total_count = 0

    for test in results_dict.get("tests", []):
        total_count += 1
        status = test.get("status", "UNKNOWN")
        test_info = {
            "name": test.get("name", "unknown"),
            "status": status,
            "description": test.get("description", ""),
        }
        test_results.append(test_info)

        if status == "FAIL":
            failed_count += 1

    summary = {
        "total_tests": total_count,
        "failed_tests": failed_count,
        "passed_tests": total_count - failed_count,
        "test_results": test_results,
    }

    return suite, summary


def validate_batch(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> ValidationResult:
    """
    Run full validation pipeline on a data batch.

    Executes both drift detection and data quality tests, then
    classifies severity and determines pass/fail status.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline reference dataset (clean data).
    current_df : pd.DataFrame
        Incoming batch to validate.

    Returns
    -------
    ValidationResult
        Comprehensive validation outcome with all metrics.
    """
    start_time = time.time()

    # --- Step 1: Data Drift Detection ---
    logger.info("Running data drift detection...")
    try:
        drift_report, drift_summary = run_drift_report(reference_df, current_df)
        drift_html = drift_report.get_html()
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return ValidationResult(
            passed=False,
            status="ERROR",
            drift_detected=False,
            drift_score=0.0,
            severity="CRITICAL",
            execution_time_seconds=time.time() - start_time,
            records_processed=len(current_df),
        )

    # --- Step 2: Data Quality Tests ---
    logger.info("Running data quality tests...")
    try:
        quality_suite, quality_summary = run_data_quality_tests(
            reference_df, current_df
        )
        quality_html = quality_suite.get_html()
    except Exception as e:
        logger.error(f"Data quality tests failed: {e}")
        quality_summary = {"total_tests": 0, "failed_tests": 0, "test_results": []}
        quality_html = None

    # --- Step 3: Classify severity ---
    drift_score = drift_summary["drift_score"]
    failed_tests = quality_summary["failed_tests"]
    total_tests = quality_summary["total_tests"]

    severity = _classify_severity(drift_score, failed_tests, total_tests)

    # Pass/fail decision: fail if ANY drift detected OR any quality test fails
    drift_detected = drift_score > 0.0
    quality_passed = failed_tests == 0
    overall_passed = not drift_detected and quality_passed

    execution_time = time.time() - start_time

    result = ValidationResult(
        passed=overall_passed,
        status="SUCCESS" if overall_passed else "DRIFT_DETECTED",
        drift_detected=drift_detected,
        drift_score=drift_score,
        drifted_columns=drift_summary["drifted_columns"],
        drift_details=drift_summary.get("per_column", {}),
        total_tests=total_tests,
        failed_tests=failed_tests,
        test_results=quality_summary["test_results"],
        severity=severity,
        drift_report_html=drift_html,
        quality_report_html=quality_html,
        execution_time_seconds=round(execution_time, 2),
        records_processed=len(current_df),
    )

    logger.info(
        f"Validation complete: status={result.status} "
        f"drift_score={result.drift_score:.3f} "
        f"failed_tests={result.failed_tests}/{result.total_tests} "
        f"severity={result.severity} "
        f"time={result.execution_time_seconds:.1f}s"
    )

    return result
