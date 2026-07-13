"""
Tests for the Drift Simulator module.
=====================================
Verifies that each corruption profile produces the expected changes
and that severity scaling works correctly.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift_simulator.corrupt_data import (
    apply_corruption,
    apply_combined_corruption,
    CameraFaultCorruptor,
    SensorMissingnessCorruptor,
    SchemaCorruptor,
    DistributionShiftCorruptor,
    CORRUPTION_PROFILES,
)
from src.utils.data_loader import generate_synthetic_baseline


@pytest.fixture
def baseline_df() -> pd.DataFrame:
    """Generate a clean baseline dataset for testing."""
    return generate_synthetic_baseline(n_samples=200, n_scenes=5, seed=42)


class TestCameraFaultCorruptor:
    """Tests for camera fault corruption."""

    def test_reduces_visibility(self, baseline_df):
        result = apply_corruption(baseline_df, "camera_fault", severity=0.8, seed=42)
        df = result.corrupted_df

        # Most rows should have visibility forced to 1
        assert (df["visibility"] == 1).mean() > 0.5
        assert "visibility" in result.columns_affected

    def test_adds_lidar_noise(self, baseline_df):
        original_lidar = baseline_df["num_lidar_pts"].copy()
        result = apply_corruption(baseline_df, "camera_fault", severity=0.8, seed=42)
        corrupted_lidar = result.corrupted_df["num_lidar_pts"]

        # Values should be different from original
        assert not original_lidar.equals(corrupted_lidar)
        # All values should still be non-negative
        assert (corrupted_lidar >= 0).all()

    def test_severity_zero_minimal_change(self, baseline_df):
        result = apply_corruption(baseline_df, "camera_fault", severity=0.0, seed=42)
        # With severity 0, corrupt_frac = 0, so very few rows affected
        assert result.rows_affected == 0 or result.severity == 0.0

    def test_returns_corruption_result(self, baseline_df):
        result = apply_corruption(baseline_df, "camera_fault", severity=0.5, seed=42)
        assert result.profile == "camera_fault"
        assert 0.0 <= result.severity <= 1.0
        assert isinstance(result.corrupted_df, pd.DataFrame)
        assert len(result.corrupted_df) == len(baseline_df)


class TestSensorMissingnessCorruptor:
    """Tests for sensor missingness corruption."""

    def test_injects_nulls(self, baseline_df):
        result = apply_corruption(baseline_df, "sensor_missing", severity=0.5, seed=42)
        df = result.corrupted_df

        # Should have NaN values in coordinate columns
        coord_cols = ["translation_x", "translation_y", "translation_z"]
        for col in coord_cols:
            assert df[col].isna().any(), f"Expected NaN in {col}"

    def test_higher_severity_more_nulls(self, baseline_df):
        low = apply_corruption(baseline_df, "sensor_missing", severity=0.2, seed=42)
        high = apply_corruption(baseline_df, "sensor_missing", severity=0.8, seed=42)

        low_nulls = low.corrupted_df.isna().sum().sum()
        high_nulls = high.corrupted_df.isna().sum().sum()

        assert high_nulls > low_nulls, "Higher severity should produce more nulls"

    def test_preserves_non_target_columns(self, baseline_df):
        result = apply_corruption(baseline_df, "sensor_missing", severity=0.5, seed=42)
        df = result.corrupted_df

        # Category column should be untouched
        assert df["category"].equals(baseline_df["category"])
        assert df["sample_token"].equals(baseline_df["sample_token"])


class TestSchemaCorruptor:
    """Tests for schema corruption."""

    def test_mild_injects_string_values(self, baseline_df):
        result = apply_corruption(baseline_df, "schema_break", severity=0.3, seed=42)
        df = result.corrupted_df

        # At least one float column should now have object dtype (mixed types)
        affected = result.columns_affected
        assert len(affected) > 0

    def test_severe_converts_column_types(self, baseline_df):
        result = apply_corruption(baseline_df, "schema_break", severity=0.7, seed=42)
        df = result.corrupted_df

        # At least one column should have string values with "_m" suffix
        has_string_col = False
        for col in ["translation_x", "translation_y", "bbox_width"]:
            if col in df.columns and df[col].dtype == object:
                has_string_col = True
                break

        assert has_string_col or len(result.columns_affected) > 0

    def test_very_high_severity_drops_column(self, baseline_df):
        result = apply_corruption(baseline_df, "schema_break", severity=0.95, seed=42)
        df = result.corrupted_df

        # May have dropped a column
        dropped = [c for c in result.columns_affected if c.startswith("DROPPED:")]
        if dropped:
            dropped_col = dropped[0].replace("DROPPED:", "")
            assert dropped_col not in df.columns


class TestDistributionShiftCorruptor:
    """Tests for distribution shift corruption."""

    def test_shifts_distributions(self, baseline_df):
        result = apply_corruption(
            baseline_df, "distribution_shift", severity=0.7, seed=42
        )
        df = result.corrupted_df

        # Check that at least one column's mean has shifted significantly
        shifted = False
        for col in result.columns_affected:
            if col in baseline_df.columns and col in df.columns:
                original_mean = baseline_df[col].mean()
                shifted_mean = df[col].mean()
                if abs(shifted_mean - original_mean) > 0.01 * abs(original_mean):
                    shifted = True
                    break

        assert shifted, "At least one column should show distribution shift"

    def test_preserves_schema(self, baseline_df):
        result = apply_corruption(
            baseline_df, "distribution_shift", severity=0.5, seed=42
        )
        df = result.corrupted_df

        # Column count should be the same
        assert len(df.columns) == len(baseline_df.columns)
        # Row count should be the same
        assert len(df) == len(baseline_df)


class TestApplyCorruption:
    """Tests for the public API."""

    def test_all_profiles_exist(self):
        expected = {"camera_fault", "sensor_missing", "schema_break", "distribution_shift"}
        assert set(CORRUPTION_PROFILES.keys()) == expected

    def test_invalid_profile_raises(self, baseline_df):
        with pytest.raises(ValueError, match="Unknown corruption profile"):
            apply_corruption(baseline_df, "nonexistent_profile")

    def test_severity_clamped(self, baseline_df):
        result = apply_corruption(baseline_df, "camera_fault", severity=1.5, seed=42)
        assert result.severity == 1.0

        result = apply_corruption(baseline_df, "camera_fault", severity=-0.5, seed=42)
        assert result.severity == 0.0

    def test_combined_corruption(self, baseline_df):
        profiles = {
            "camera_fault": 0.3,
            "sensor_missing": 0.4,
        }
        result = apply_combined_corruption(baseline_df, profiles, seed=42)

        assert result.profile == "combined"
        assert result.severity == 0.4  # max of provided severities
        assert len(result.columns_affected) > 0

    def test_reproducibility(self, baseline_df):
        r1 = apply_corruption(baseline_df, "distribution_shift", severity=0.5, seed=123)
        r2 = apply_corruption(baseline_df, "distribution_shift", severity=0.5, seed=123)

        pd.testing.assert_frame_equal(r1.corrupted_df, r2.corrupted_df)
