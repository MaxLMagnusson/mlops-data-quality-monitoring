"""
Drift Simulator — Data Corruption Engine
==========================================
Simulates real-world data quality issues in autonomous driving datasets.
Each corruptor takes a severity parameter (0.0–1.0) controlling the
intensity of the corruption.

Corruption Profiles:
- CameraFaultCorruptor:       Degrades image quality (brightness, noise)
- SensorMissingnessCorruptor: Drops coordinate values, introduces NaN
- SchemaCorruptor:            Injects type mismatches (float → string)
- DistributionShiftCorruptor: Applies scaling/offset to numeric features

Usage:
    from src.drift_simulator.corrupt_data import apply_corruption

    corrupted_df = apply_corruption(
        df=baseline_df,
        profile="camera_fault",
        severity=0.7,
        seed=42,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class CorruptionResult:
    """Result of a corruption operation with metadata for logging."""

    corrupted_df: pd.DataFrame
    profile: str
    severity: float
    rows_affected: int
    columns_affected: list[str]
    corruption_details: dict = field(default_factory=dict)

    @property
    def corruption_rate(self) -> float:
        """Fraction of total values that were corrupted."""
        total = self.corrupted_df.shape[0] * len(self.columns_affected)
        return self.rows_affected / max(total, 1)


class CameraFaultCorruptor:
    """
    Simulates camera sensor faults by degrading image metadata.

    In a real pipeline, this would modify actual image pixel arrays.
    For metadata-level simulation, it corrupts brightness/exposure-related
    features and adds noise to pixel-intensity proxy columns.
    """

    @staticmethod
    def corrupt(
        df: pd.DataFrame,
        severity: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> CorruptionResult:
        """
        Apply camera fault corruption to the dataset.

        Simulates:
        - Reduced visibility scores (sensor degradation)
        - Noisy LiDAR/RADAR point counts (interference)
        - Corrupted timestamp values (clock drift)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to corrupt.
        severity : float
            Corruption intensity (0.0 = no change, 1.0 = maximum corruption).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        rng = rng or np.random.default_rng()
        result = df.copy()
        affected_cols = []
        rows_affected = 0

        # Fraction of rows to corrupt
        corrupt_frac = min(severity * 0.8, 0.95)
        mask = rng.random(len(result)) < corrupt_frac

        # Reduce visibility (simulate fogged/dirty camera)
        if "visibility" in result.columns:
            result.loc[mask, "visibility"] = 1  # Force worst visibility
            affected_cols.append("visibility")
            rows_affected += mask.sum()

        # Add noise to LiDAR point counts (sensor interference)
        if "num_lidar_pts" in result.columns:
            noise_scale = int(200 * severity)
            noise = rng.integers(-noise_scale, noise_scale, size=mask.sum())
            result.loc[mask, "num_lidar_pts"] = (
                result.loc[mask, "num_lidar_pts"] + noise
            ).clip(0)
            affected_cols.append("num_lidar_pts")

        # Corrupt RADAR points similarly
        if "num_radar_pts" in result.columns:
            noise = rng.integers(-int(20 * severity), int(20 * severity), size=mask.sum())
            result.loc[mask, "num_radar_pts"] = (
                result.loc[mask, "num_radar_pts"] + noise
            ).clip(0)
            affected_cols.append("num_radar_pts")

        return CorruptionResult(
            corrupted_df=result,
            profile="camera_fault",
            severity=severity,
            rows_affected=int(rows_affected),
            columns_affected=affected_cols,
            corruption_details={
                "corrupt_fraction": corrupt_frac,
                "visibility_forced_to": 1,
                "lidar_noise_scale": int(200 * severity),
            },
        )


class CameraImageCorruptor:
    """
    Corrupts actual image files by reducing brightness/contrast and adding noise.

    This operates on image arrays using OpenCV, simulating physical camera faults
    like dirty lenses, exposure errors, or electrical noise.
    """

    @staticmethod
    def corrupt_image(
        image_path: str,
        output_path: str,
        severity: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Corrupt a single image file.

        Parameters
        ----------
        image_path : str
            Path to input image.
        output_path : str
            Path to save corrupted image.
        severity : float
            Corruption intensity.
        rng : np.random.Generator, optional
            Random generator.

        Returns
        -------
        dict
            Corruption metadata.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python-headless required for image corruption")

        rng = rng or np.random.default_rng()
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Reduce brightness
        brightness_factor = max(0.1, 1.0 - severity * 0.8)
        img = (img.astype(np.float32) * brightness_factor).clip(0, 255).astype(np.uint8)

        # Add Gaussian noise
        noise_std = severity * 60
        noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

        # Reduce contrast
        contrast_factor = max(0.2, 1.0 - severity * 0.6)
        mean = img.mean()
        img = ((img.astype(np.float32) - mean) * contrast_factor + mean).clip(0, 255).astype(np.uint8)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)

        return {
            "brightness_factor": brightness_factor,
            "noise_std": noise_std,
            "contrast_factor": contrast_factor,
        }


class SensorMissingnessCorruptor:
    """
    Simulates sensor data loss by randomly dropping values.

    This creates NaN/null values in coordinate columns and sensor readings,
    mimicking intermittent sensor failures or data transmission issues.
    """

    @staticmethod
    def corrupt(
        df: pd.DataFrame,
        severity: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> CorruptionResult:
        """
        Apply sensor missingness corruption.

        Targets coordinate and measurement columns with random NaN injection.
        """
        rng = rng or np.random.default_rng()
        result = df.copy()

        # Columns that can have missing values from sensor failures
        target_cols = [
            "translation_x", "translation_y", "translation_z",
            "bbox_width", "bbox_height", "bbox_length",
            "num_lidar_pts", "num_radar_pts",
        ]
        target_cols = [c for c in target_cols if c in result.columns]

        drop_frac = min(severity * 0.6, 0.8)
        rows_affected = 0
        affected_cols = []

        for col in target_cols:
            mask = rng.random(len(result)) < drop_frac
            # Convert int columns to float to support NaN
            if result[col].dtype in ("int64", "int32"):
                result[col] = result[col].astype("float64")
            result.loc[mask, col] = np.nan
            rows_affected += mask.sum()
            affected_cols.append(col)

        return CorruptionResult(
            corrupted_df=result,
            profile="sensor_missing",
            severity=severity,
            rows_affected=int(rows_affected),
            columns_affected=affected_cols,
            corruption_details={
                "drop_fraction": drop_frac,
                "null_injection_columns": affected_cols,
            },
        )


class SchemaCorruptor:
    """
    Injects schema violations into the dataset.

    Simulates upstream ETL bugs where column types change unexpectedly
    (e.g., float coordinates become strings, new unexpected columns appear,
    or required columns are missing entirely).
    """

    @staticmethod
    def corrupt(
        df: pd.DataFrame,
        severity: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> CorruptionResult:
        """
        Apply schema corruption.

        Low severity: inject a few string values into numeric columns.
        High severity: convert entire columns to strings, drop columns.
        """
        rng = rng or np.random.default_rng()
        result = df.copy()
        affected_cols = []
        rows_affected = 0

        float_cols = [
            "translation_x", "translation_y", "translation_z",
            "bbox_width", "bbox_height", "bbox_length",
        ]
        float_cols = [c for c in float_cols if c in result.columns]

        if severity < 0.5:
            # Mild: inject string values into some numeric cells
            n_corrupted_cols = max(1, int(len(float_cols) * severity))
            cols_to_corrupt = rng.choice(
                float_cols, size=n_corrupted_cols, replace=False
            ).tolist()

            for col in cols_to_corrupt:
                # Convert to object dtype so we can mix types
                result[col] = result[col].astype(object)
                corrupt_mask = rng.random(len(result)) < (severity * 0.3)
                corrupt_values = rng.choice(
                    ["N/A", "null", "error", "-", "???", "NaN_str"],
                    size=corrupt_mask.sum(),
                )
                result.loc[corrupt_mask, col] = corrupt_values
                rows_affected += corrupt_mask.sum()
                affected_cols.append(col)
        else:
            # Severe: convert entire numeric columns to string
            n_cols = max(1, int(len(float_cols) * severity * 0.5))
            cols_to_break = rng.choice(
                float_cols, size=min(n_cols, len(float_cols)), replace=False
            ).tolist()

            for col in cols_to_break:
                result[col] = result[col].astype(str) + "_m"  # e.g., "1.5_m"
                rows_affected += len(result)
                affected_cols.append(col)

            # Very high severity: drop a column entirely
            if severity > 0.8 and float_cols:
                drop_col = rng.choice(float_cols)
                if drop_col not in cols_to_break:
                    result = result.drop(columns=[drop_col])
                    affected_cols.append(f"DROPPED:{drop_col}")

        return CorruptionResult(
            corrupted_df=result,
            profile="schema_break",
            severity=severity,
            rows_affected=int(rows_affected),
            columns_affected=affected_cols,
            corruption_details={
                "type": "mixed_types" if severity < 0.5 else "type_conversion",
            },
        )


class DistributionShiftCorruptor:
    """
    Simulates gradual distribution drift in numeric features.

    Applies scaling factors and offsets to numeric columns, simulating
    scenarios like:
    - Sensor calibration drift over time
    - Environmental changes (different geography, weather)
    - Upstream data processing changes
    """

    @staticmethod
    def corrupt(
        df: pd.DataFrame,
        severity: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> CorruptionResult:
        """
        Apply distribution shift corruption.

        Applies multiplicative scaling and additive offset to numeric features.
        The shift is designed to be detectable by statistical tests (KS, etc.)
        while remaining plausible.
        """
        rng = rng or np.random.default_rng()
        result = df.copy()

        numeric_cols = [
            "bbox_width", "bbox_height", "bbox_length",
            "translation_x", "translation_y", "translation_z",
            "num_lidar_pts", "num_radar_pts",
        ]
        numeric_cols = [c for c in numeric_cols if c in result.columns]

        # Number of columns to shift
        n_shifted = max(1, int(len(numeric_cols) * (0.3 + 0.7 * severity)))
        cols_to_shift = rng.choice(
            numeric_cols, size=min(n_shifted, len(numeric_cols)), replace=False
        ).tolist()

        affected_cols = []
        rows_affected = 0

        for col in cols_to_shift:
            col_std = result[col].std()
            col_mean = result[col].mean()

            # Scale factor: 1.0 (no change) → up to 2.0x or 0.5x at max severity
            scale = 1.0 + rng.uniform(-severity * 0.5, severity * 0.8)

            # Offset: 0 → up to 2 standard deviations at max severity
            offset = rng.uniform(-1, 1) * severity * col_std * 1.5

            result[col] = result[col] * scale + offset

            # For integer columns, round back
            if col in ("num_lidar_pts", "num_radar_pts", "visibility"):
                result[col] = result[col].round().clip(0).astype("int64")

            affected_cols.append(col)
            rows_affected += len(result)

        return CorruptionResult(
            corrupted_df=result,
            profile="distribution_shift",
            severity=severity,
            rows_affected=int(rows_affected),
            columns_affected=affected_cols,
            corruption_details={
                "shifted_columns": cols_to_shift,
                "n_shifted": len(cols_to_shift),
            },
        )


# ============================================================================
# Public API
# ============================================================================

CORRUPTION_PROFILES = {
    "camera_fault": CameraFaultCorruptor.corrupt,
    "sensor_missing": SensorMissingnessCorruptor.corrupt,
    "schema_break": SchemaCorruptor.corrupt,
    "distribution_shift": DistributionShiftCorruptor.corrupt,
}


def apply_corruption(
    df: pd.DataFrame,
    profile: str,
    severity: float = 0.5,
    seed: int = 42,
) -> CorruptionResult:
    """
    Apply a named corruption profile to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline data.
    profile : str
        Name of corruption profile. One of:
        'camera_fault', 'sensor_missing', 'schema_break', 'distribution_shift'
    severity : float
        Intensity of corruption (0.0 = no change, 1.0 = maximum).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CorruptionResult
        Corrupted data with metadata about what was changed.

    Raises
    ------
    ValueError
        If profile name is not recognized.
    """
    if profile not in CORRUPTION_PROFILES:
        raise ValueError(
            f"Unknown corruption profile: '{profile}'. "
            f"Choose from: {list(CORRUPTION_PROFILES.keys())}"
        )

    severity = max(0.0, min(1.0, severity))  # Clamp to [0, 1]
    rng = np.random.default_rng(seed)

    corruptor = CORRUPTION_PROFILES[profile]
    return corruptor(df, severity=severity, rng=rng)


def apply_combined_corruption(
    df: pd.DataFrame,
    profiles: dict[str, float],
    seed: int = 42,
) -> CorruptionResult:
    """
    Apply multiple corruption profiles sequentially.

    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline data.
    profiles : dict[str, float]
        Mapping of profile name → severity.
        Example: {"camera_fault": 0.3, "sensor_missing": 0.5}
    seed : int
        Random seed.

    Returns
    -------
    CorruptionResult
        Final corrupted data with combined metadata.
    """
    result_df = df.copy()
    all_affected_cols = []
    total_rows_affected = 0

    for profile, severity in profiles.items():
        result = apply_corruption(result_df, profile, severity, seed)
        result_df = result.corrupted_df
        all_affected_cols.extend(result.columns_affected)
        total_rows_affected += result.rows_affected

    return CorruptionResult(
        corrupted_df=result_df,
        profile="combined",
        severity=max(profiles.values()),
        rows_affected=total_rows_affected,
        columns_affected=list(set(all_affected_cols)),
        corruption_details={"profiles": profiles},
    )
