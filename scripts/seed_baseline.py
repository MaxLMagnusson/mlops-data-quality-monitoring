"""
Seed Baseline — Upload Reference Dataset to MinIO
===================================================
Generates the synthetic baseline dataset and uploads it to MinIO's
baseline bucket. Run this after `docker compose up` to initialize
the reference data.

Usage:
    python scripts/seed_baseline.py
    python scripts/seed_baseline.py --samples 2000 --endpoint http://localhost:9000
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.resources.minio_resource import MinIOResource
from src.utils.data_loader import generate_synthetic_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Generate and upload baseline dataset to MinIO."
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of annotation samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--scenes", type=int, default=10,
        help="Number of scenes (default: 10)",
    )
    parser.add_argument(
        "--endpoint", type=str,
        default=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        help="MinIO endpoint URL",
    )
    parser.add_argument(
        "--access-key", type=str,
        default=os.getenv("MINIO_ROOT_USER", "minioadmin"),
    )
    parser.add_argument(
        "--secret-key", type=str,
        default=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Baseline Data Seeder")
    print("=" * 60)

    # Generate synthetic data
    print(f"\n📊 Generating {args.samples} synthetic annotations...")
    start = time.time()
    df = generate_synthetic_baseline(
        n_samples=args.samples,
        n_scenes=args.scenes,
        output_dir="data/synthetic",
    )
    gen_time = time.time() - start
    print(f"   ✅ Generated in {gen_time:.1f}s")
    print(f"   → {len(df)} records, {df['category'].nunique()} categories")
    print(f"   → Columns: {list(df.columns)}")

    # Upload to MinIO
    print(f"\n☁️  Uploading to MinIO ({args.endpoint})...")
    minio = MinIOResource(
        endpoint_url=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )

    # Ensure bucket exists
    minio.create_bucket_if_not_exists("baseline")

    # Upload
    uri = minio.upload_dataframe(
        df=df,
        bucket="baseline",
        key="baseline_metadata.parquet",
        format="parquet",
    )
    print(f"   ✅ Uploaded to {uri}")

    # Also save locally for reference
    print("\n💾 Local copy saved to data/synthetic/baseline_metadata.parquet")
    print(f"\n{'=' * 60}")
    print("  Baseline seeded successfully!")
    print("  You can now run the drift simulator to test the pipeline.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
