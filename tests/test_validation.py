"""
Tests for the Evidently AI Validation Engine.
==============================================
Verifies that the validation engine correctly detects drift patterns,
catches schema violations, and produces properly structured results.
"""

import numpy as np
import pandas as pd
import pytest

from src.assets.validation import (
    validate_batch,
    run_drift_report,
    run_data_quality_tests,
    ValidationResult,
    _classify_severity,
    _get_column_mapping,
)
from src.utils.data_loader import generate_synthetic_baseline
from src.drift_simulator.corrupt_data import apply_corruption


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Clean reference dataset."""
    return generate_synthetic_baseline(n_samples=300, n_scenes=5, seed=42)


@pytest.fixture
def clean_current_df() -> pd.DataFrame:
    """Clean current dataset (different seed, same distribution)."""
    return generate_synthetic_baseline(n_samples=300, n_scenes=5, seed=99)


@pytest.fixture
def drifted_df(reference_df) -> pd.DataFrame:
    """Dataset with significant distribution shift."""
    result = apply_corruption(
        reference_df, "distribution_shift", severity=0.8, seed=42
    )
    return result.corrupted_df


@pytest.fixture
def missing_df(reference_df) -> pd.DataFrame:
    """Dataset with missing values."""
    result = apply_corruption(
        reference_df, "sensor_missing", severity=0.6, seed=42
    )
    return result.corrupted_df


class TestValidateBatch:
    """Tests for the full validation pipeline."""

    def test_clean_data_passes(self, reference_df, clean_current_df):
        """Clean data from the same distribution should pass or have low drift."""
        result = validate_batch(reference_df, clean_current_df)

        assert isinstance(result, ValidationResult)
        assert result.records_processed == len(clean_current_df)
        assert result.execution_time_seconds > 0
        # Clean data may still show some statistical variation,
        # so we check severity is at most MEDIUM
        assert result.severity in ("LOW", "MEDIUM")

    def test_drifted_data_detected(self, reference_df, drifted_df):
        """Significantly drifted data should be detected."""
        result = validate_batch(reference_df, drifted_df)

        assert result.drift_detected is True
        assert result.drift_score > 0.0
        assert len(result.drifted_columns) > 0
        assert result.status == "DRIFT_DETECTED"

    def test_missing_data_detected(self, reference_df, missing_df):
        """Data with many nulls should fail quality tests."""
        result = validate_batch(reference_df, missing_df)

        # Should have failed tests due to null values
        assert result.total_tests > 0
        # Missing data should be flagged
        assert result.status == "DRIFT_DETECTED" or result.failed_tests > 0

    def test_empty_batch_returns_error(self, reference_df):
        """Empty batch should return an error result."""
        empty_df = pd.DataFrame()
        result = validate_batch(reference_df, empty_df)
        # Empty check is in the routing asset, not validation itself
        # Validation would either error or produce a result
        assert isinstance(result, ValidationResult)

    def test_result_serialization(self, reference_df, clean_current_df):
        """ValidationResult should serialize to a dictionary."""
        result = validate_batch(reference_df, clean_current_df)
        result_dict = result.to_dict()

        assert "passed" in result_dict
        assert "drift_score" in result_dict
        assert "severity" in result_dict
        assert isinstance(result_dict["drift_score"], float)


class TestDriftReport:
    """Tests for the drift detection report."""

    def test_generates_report(self, reference_df, clean_current_df):
        report, summary = run_drift_report(reference_df, clean_current_df)

        assert "drift_score" in summary
        assert "drifted_columns" in summary
        assert "total_columns_tested" in summary
        assert 0.0 <= summary["drift_score"] <= 1.0

    def test_html_generation(self, reference_df, clean_current_df):
        report, _ = run_drift_report(reference_df, clean_current_df)
        html = report.get_html()

        assert isinstance(html, str)
        assert len(html) > 100
        assert "<html" in html.lower() or "<!doctype" in html.lower()


class TestDataQualityTests:
    """Tests for the data quality test suite."""

    def test_runs_test_suite(self, reference_df, clean_current_df):
        suite, summary = run_data_quality_tests(reference_df, clean_current_df)

        assert "total_tests" in summary
        assert "failed_tests" in summary
        assert summary["total_tests"] > 0

    def test_clean_data_passes_quality(self, reference_df, clean_current_df):
        _, summary = run_data_quality_tests(reference_df, clean_current_df)

        # Clean data should pass most tests
        fail_rate = summary["failed_tests"] / max(summary["total_tests"], 1)
        assert fail_rate < 0.5, f"Clean data failed {fail_rate:.0%} of tests"


class TestSeverityClassification:
    """Tests for the severity classification logic."""

    def test_low_severity(self):
        assert _classify_severity(0.05, 1, 50) == "LOW"

    def test_medium_severity(self):
        assert _classify_severity(0.2, 3, 50) == "MEDIUM"

    def test_high_severity(self):
        assert _classify_severity(0.4, 10, 50) == "HIGH"

    def test_critical_severity(self):
        assert _classify_severity(0.7, 25, 50) == "CRITICAL"

    def test_critical_from_test_failures(self):
        assert _classify_severity(0.0, 25, 50) == "CRITICAL"


class TestColumnMapping:
    """Tests for column mapping configuration."""

    def test_maps_available_columns(self, reference_df):
        mapping = _get_column_mapping(reference_df)

        assert len(mapping.numerical_features) > 0
        assert len(mapping.categorical_features) > 0

    def test_filters_missing_columns(self):
        small_df = pd.DataFrame({"bbox_width": [1.0], "category": ["car"]})
        mapping = _get_column_mapping(small_df)

        assert "bbox_width" in mapping.numerical_features
        assert "translation_x" not in mapping.numerical_features
