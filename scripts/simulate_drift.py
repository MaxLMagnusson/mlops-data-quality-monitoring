"""
Drift Simulator CLI - Inject Data Quality Issues
==================================================
Generates a corrupted data batch and uploads it to the MinIO
incoming-data bucket to trigger the Dagster pipeline.

Usage:
    # Simulate camera sensor fault (medium severity)
    python scripts/simulate_drift.py --profile camera_fault --severity 0.5

    # Simulate missing sensor data (high severity)
    python scripts/simulate_drift.py --profile sensor_missing --severity 0.8

    # Simulate schema violations
    python scripts/simulate_drift.py --profile schema_break --severity 0.6

    # Simulate gradual distribution drift
    python scripts/simulate_drift.py --profile distribution_shift --severity 0.4

    # Combined corruption
    python scripts/simulate_drift.py --profile combined --severity 0.5

    # Clean batch (no corruption - should pass validation)
    python scripts/simulate_drift.py --profile none --severity 0.0
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift_simulator.corrupt_data import (
    apply_combined_corruption,
    apply_corruption,
)
from src.resources.minio_resource import MinIOResource
from src.utils.data_loader import generate_synthetic_baseline

PROFILE_DESCRIPTIONS = {
    "camera_fault": "🎥 Camera Fault - Degrades visibility, adds sensor noise",
    "sensor_missing": "📡 Sensor Missing - Drops coordinate values, introduces NaN",
    "schema_break": "🔧 Schema Break - Injects type mismatches (float → string)",
    "distribution_shift": "📈 Distribution Shift - Scales/offsets numeric features",
    "combined": "💥 Combined - Multiple corruption types applied together",
    "none": "✅ Clean - No corruption (should pass validation)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate data drift by generating corrupted batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:25s} {v}" for k, v in PROFILE_DESCRIPTIONS.items()),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="distribution_shift",
        choices=list(PROFILE_DESCRIPTIONS.keys()),
        help="Corruption profile to apply (default: distribution_shift)",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
        help="Corruption severity 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of records in the batch (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: based on current time)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        help="MinIO endpoint URL",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default=os.getenv("MINIO_ROOT_USER", "minioadmin"),
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate corrupted data but don't upload to MinIO",
    )
    args = parser.parse_args()

    seed = args.seed or int(time.time()) % 100_000
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  Data Drift Simulator")
    print("=" * 60)
    print(f"\n  Profile:  {PROFILE_DESCRIPTIONS[args.profile]}")
    print(f"  Severity: {args.severity}")
    print(f"  Samples:  {args.samples}")
    print(f"  Seed:     {seed}")

    # Step 1: Generate clean data (using different seed than baseline)
    print("\n📊 Generating clean batch data...")
    clean_df = generate_synthetic_baseline(
        n_samples=args.samples,
        n_scenes=10,
        seed=seed + 100,  # Different seed to get different-but-similar data
    )

    # Step 2: Apply corruption
    if args.profile == "none":
        corrupted_df = clean_df
        print("   → No corruption applied (clean batch)")
        rows_affected = 0
        cols_affected = []
    elif args.profile == "combined":
        print(f"\n🔨 Applying combined corruption (severity={args.severity})...")
        profiles = {
            "camera_fault": args.severity * 0.6,
            "sensor_missing": args.severity * 0.4,
            "distribution_shift": args.severity * 0.8,
        }
        result = apply_combined_corruption(clean_df, profiles, seed=seed)
        corrupted_df = result.corrupted_df
        rows_affected = result.rows_affected
        cols_affected = result.columns_affected
    else:
        print(f"\n🔨 Applying {args.profile} corruption (severity={args.severity})...")
        result = apply_corruption(
            clean_df,
            profile=args.profile,
            severity=args.severity,
            seed=seed,
        )
        corrupted_df = result.corrupted_df
        rows_affected = result.rows_affected
        cols_affected = result.columns_affected

    print(f"   → Rows affected: {rows_affected}")
    print(f"   → Columns affected: {cols_affected}")
    print(f"   → Output shape: {corrupted_df.shape}")

    # Step 3: Upload to MinIO
    if args.dry_run:
        print("\n🏃 Dry run - skipping MinIO upload.")
        # Save locally instead
        output_path = f"data/synthetic/corrupted_batch_{timestamp}.parquet"
        os.makedirs("data/synthetic", exist_ok=True)
        corrupted_df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"   → Saved locally to {output_path}")
    else:
        batch_key = f"batch_{args.profile}_{timestamp}.parquet"
        print(f"\n☁️  Uploading to MinIO ({args.endpoint})...")
        minio = MinIOResource(
            endpoint_url=args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
        )
        uri = minio.upload_dataframe(
            df=corrupted_df,
            bucket="incoming-data",
            key=batch_key,
            format="parquet",
        )
        print(f"   ✅ Uploaded to {uri}")
        print("\n   The Dagster sensor will detect this file and trigger validation.")
        print("   Check the Dagster UI at http://localhost:3000")

    print(f"\n{'=' * 60}")
    print("  Simulation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
