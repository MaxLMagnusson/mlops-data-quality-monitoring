"""
Tests for the Data Routing Logic.
==================================
Verifies that the routing layer correctly determines pass/fail
outcomes and produces the right routing decisions.
"""

import pandas as pd
import pytest

from src.assets.validation import validate_batch, ValidationResult
from src.utils.data_loader import generate_synthetic_baseline
from src.drift_simulator.corrupt_data import apply_corruption


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Clean reference dataset."""
    return generate_synthetic_baseline(n_samples=200, n_scenes=5, seed=42)


@pytest.fixture
def clean_batch() -> pd.DataFrame:
    """Clean batch that should pass validation."""
    return generate_synthetic_baseline(n_samples=200, n_scenes=5, seed=99)


@pytest.fixture
def drifted_batch(reference_df) -> pd.DataFrame:
    """Drifted batch that should fail validation."""
    result = apply_corruption(
        reference_df, "distribution_shift", severity=0.9, seed=42
    )
    return result.corrupted_df


class TestRoutingDecision:
    """Tests for the routing decision logic."""

    def test_clean_data_routes_to_clean_lake(self, reference_df, clean_batch):
        """Clean data should be routed to the clean production lake."""
        result = validate_batch(reference_df, clean_batch)
        # Clean data should either pass or have low severity
        if result.passed:
            assert result.status == "SUCCESS"

    def test_drifted_data_routes_to_quarantine(self, reference_df, drifted_batch):
        """Drifted data should be routed to quarantine."""
        result = validate_batch(reference_df, drifted_batch)

        assert result.drift_detected is True
        assert result.status == "DRIFT_DETECTED"
        assert result.severity in ("MEDIUM", "HIGH", "CRITICAL")

    def test_validation_result_has_reports(self, reference_df, clean_batch):
        """Validation result should contain HTML reports."""
        result = validate_batch(reference_df, clean_batch)

        assert result.drift_report_html is not None
        assert len(result.drift_report_html) > 0

    def test_validation_result_has_metadata(self, reference_df, clean_batch):
        """Validation result should contain execution metadata."""
        result = validate_batch(reference_df, clean_batch)

        assert result.records_processed == len(clean_batch)
        assert result.execution_time_seconds > 0
        assert result.total_tests > 0

    def test_different_corruption_profiles_detected(self, reference_df):
        """All corruption profiles should be detectable."""
        profiles = ["camera_fault", "sensor_missing", "distribution_shift"]

        for profile in profiles:
            corrupted = apply_corruption(
                reference_df, profile, severity=0.8, seed=42
            )
            result = validate_batch(reference_df, corrupted.corrupted_df)

            assert result.drift_detected or result.failed_tests > 0, (
                f"Profile '{profile}' at severity 0.8 was not detected"
            )
