"""
Ingestion Assets - Data Loading from MinIO
============================================
Dagster assets for detecting and loading incoming data batches
from the MinIO object storage data lake.

Includes:
- Sensor to detect new data in the incoming-data bucket
- Asset to load the baseline reference dataset
- Asset to load incoming batch data
"""


import logging

import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetKey,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    asset,
    sensor,
)

from src.resources.minio_resource import MinIOResource

logger = logging.getLogger(__name__)

# Bucket configuration
BASELINE_BUCKET = "baseline"
BASELINE_KEY = "baseline_metadata.parquet"
INCOMING_BUCKET = "incoming-data"


@asset(
    group_name="ingestion",
    description="Reference baseline dataset loaded from MinIO for drift comparison.",
    compute_kind="MinIO",
)
def baseline_data(
    context: AssetExecutionContext,
    minio: MinIOResource,
) -> pd.DataFrame:
    """
    Load the baseline (reference) dataset from MinIO.

    This is the clean, untouched dataset that represents the expected
    data distribution. All incoming batches are compared against this.
    """
    context.log.info(f"Loading baseline from s3://{BASELINE_BUCKET}/{BASELINE_KEY}")

    df = minio.download_dataframe(
        bucket=BASELINE_BUCKET,
        key=BASELINE_KEY,
        format="parquet",
    )

    context.log.info(
        f"Baseline loaded: {len(df)} records, "
        f"{len(df.columns)} columns, "
        f"categories: {df['category'].nunique() if 'category' in df.columns else 'N/A'}"
    )

    return df


@asset(
    group_name="ingestion",
    description="Incoming data batch loaded from MinIO for validation.",
    compute_kind="MinIO",
)
def incoming_batch(
    context: AssetExecutionContext,
    minio: MinIOResource,
) -> pd.DataFrame:
    """
    Load the most recent incoming data batch from MinIO.

    Scans the incoming-data bucket for Parquet files and loads
    the most recently uploaded one for validation.
    """
    context.log.info(f"Scanning s3://{INCOMING_BUCKET}/ for new batches...")

    objects = minio.list_objects(INCOMING_BUCKET)

    if not objects:
        context.log.warning("No objects found in incoming-data bucket.")
        return pd.DataFrame()

    # Find the most recent parquet file
    parquet_objects = [obj for obj in objects if obj["Key"].endswith(".parquet")]

    if not parquet_objects:
        context.log.warning("No Parquet files found in incoming-data bucket.")
        return pd.DataFrame()

    # Sort by last modified, get most recent
    latest = sorted(parquet_objects, key=lambda x: x["LastModified"], reverse=True)[0]
    latest_key = latest["Key"]

    context.log.info(f"Loading batch: s3://{INCOMING_BUCKET}/{latest_key}")

    df = minio.download_dataframe(
        bucket=INCOMING_BUCKET,
        key=latest_key,
        format="parquet",
    )

    context.log.info(f"Batch loaded: {len(df)} records from {latest_key}")

    # Store the batch key in metadata for downstream assets
    context.add_output_metadata(
        {
            "batch_key": latest_key,
            "num_records": len(df),
            "num_columns": len(df.columns),
        }
    )

    return df


@sensor(
    asset_selection=[AssetKey("incoming_batch")],
    minimum_interval_seconds=30,
    description="Polls MinIO incoming-data bucket for new batch files.",
)
def incoming_data_sensor(
    context: SensorEvaluationContext,
    minio: MinIOResource,
) -> SensorResult | SkipReason:
    """
    Sensor that watches MinIO for new data uploads.

    Triggers a pipeline run when a new Parquet file appears in the
    incoming-data bucket. Uses a cursor to track which files have
    already been processed.
    """
    objects = minio.list_objects(INCOMING_BUCKET)
    parquet_objects = [obj for obj in objects if obj["Key"].endswith(".parquet")]

    if not parquet_objects:
        return SkipReason("No Parquet files found in incoming-data bucket.")

    # Get cursor (last processed file key)
    last_processed = context.cursor or ""

    # Find new files
    new_files = [obj for obj in parquet_objects if obj["Key"] > last_processed]

    if not new_files:
        return SkipReason("No new files since last check.")

    # Sort by key and process the newest
    new_files.sort(key=lambda x: x["Key"])
    latest = new_files[-1]

    context.log.info(f"New batch detected: {latest['Key']}")

    return SensorResult(
        run_requests=[RunRequest(run_key=latest["Key"])],
        cursor=latest["Key"],
    )
