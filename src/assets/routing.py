"""
Routing Assets — Conditional Data Routing Based on Validation
==============================================================
Implements the pass/fail routing logic:

- PASSED:  Move data to /clean-production-lake, log SUCCESS
- FAILED:  Move data to /quarantine, log DRIFT_DETECTED with severity

Also saves Evidently HTML reports to the /reports bucket in MinIO.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetIn,
    MetadataValue,
    asset,
)

from src.assets.validation import validate_batch
from src.resources.minio_resource import MinIOResource
from src.resources.postgres_resource import PostgresResource

logger = logging.getLogger(__name__)

# Bucket configuration
INCOMING_BUCKET = "incoming-data"
CLEAN_BUCKET = "clean-production-lake"
QUARANTINE_BUCKET = "quarantine"
REPORTS_BUCKET = "reports"


@asset(
    group_name="validation",
    ins={
        "baseline_data": AssetIn(key="baseline_data"),
        "incoming_batch": AssetIn(key="incoming_batch"),
    },
    description="Runs Evidently AI validation on the incoming batch vs baseline.",
    compute_kind="Evidently",
)
def validation_result(
    context: AssetExecutionContext,
    baseline_data: pd.DataFrame,
    incoming_batch: pd.DataFrame,
) -> dict:
    """
    Run the full Evidently validation pipeline.

    Compares the incoming batch against the baseline reference data
    using both drift detection and data quality tests.
    """
    if incoming_batch.empty:
        context.log.warning("Incoming batch is empty — skipping validation.")
        return {"passed": False, "status": "ERROR", "error": "Empty batch"}

    context.log.info(
        f"Validating batch ({len(incoming_batch)} records) "
        f"against baseline ({len(baseline_data)} records)..."
    )

    result = validate_batch(
        reference_df=baseline_data,
        current_df=incoming_batch,
    )

    context.log.info(
        f"Validation result: status={result.status}, "
        f"drift_score={result.drift_score:.3f}, "
        f"severity={result.severity}"
    )

    # Add rich metadata for Dagster UI
    context.add_output_metadata(
        {
            "status": result.status,
            "drift_score": result.drift_score,
            "drifted_columns": str(result.drifted_columns),
            "severity": result.severity,
            "failed_tests": f"{result.failed_tests}/{result.total_tests}",
            "execution_time": f"{result.execution_time_seconds:.1f}s",
        }
    )

    # Store the HTML reports and result data for the routing step
    return {
        "passed": result.passed,
        "status": result.status,
        "drift_score": result.drift_score,
        "drifted_columns": result.drifted_columns,
        "severity": result.severity,
        "failed_tests": result.failed_tests,
        "total_tests": result.total_tests,
        "execution_time_seconds": result.execution_time_seconds,
        "records_processed": result.records_processed,
        "drift_report_html": result.drift_report_html,
        "quality_report_html": result.quality_report_html,
        "test_results": result.test_results,
    }


@asset(
    group_name="routing",
    ins={"validation_result": AssetIn(key="validation_result")},
    description="Routes data to clean lake or quarantine based on validation results.",
    compute_kind="Python",
)
def route_data(
    context: AssetExecutionContext,
    validation_result: dict,
    minio: MinIOResource,
    postgres: PostgresResource,
) -> dict:
    """
    Route data based on validation results and log the outcome.

    - If validation PASSED: move data to clean-production-lake
    - If validation FAILED: move data to quarantine
    - Always: save reports to MinIO and log to PostgreSQL
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    passed = validation_result["passed"]
    status = validation_result["status"]

    # --- Step 1: Find the batch to route ---
    objects = minio.list_objects(INCOMING_BUCKET)
    parquet_objects = [obj for obj in objects if obj["Key"].endswith(".parquet")]

    if not parquet_objects:
        context.log.warning("No parquet files to route in incoming-data bucket.")
        batch_name = f"unknown_batch_{timestamp}"
    else:
        latest = sorted(parquet_objects, key=lambda x: x["LastModified"], reverse=True)[0]
        batch_name = latest["Key"]

    # --- Step 2: Save Evidently reports to MinIO ---
    report_paths = {}

    drift_html = validation_result.get("drift_report_html")
    if drift_html:
        drift_key = f"drift_report_{timestamp}.html"
        report_uri = minio.upload_html(drift_html, REPORTS_BUCKET, drift_key)
        report_paths["drift_report"] = report_uri
        context.log.info(f"Saved drift report to {report_uri}")

    quality_html = validation_result.get("quality_report_html")
    if quality_html:
        quality_key = f"quality_report_{timestamp}.html"
        report_uri = minio.upload_html(quality_html, REPORTS_BUCKET, quality_key)
        report_paths["quality_report"] = report_uri
        context.log.info(f"Saved quality report to {report_uri}")

    # --- Step 3: Route data ---
    if passed:
        # Move to clean production lake
        dest_prefix = f"validated/{timestamp}/"
        context.log.info(f"✅ PASSED — Moving data to s3://{CLEAN_BUCKET}/{dest_prefix}")
        if parquet_objects:
            moved = minio.move_prefix(INCOMING_BUCKET, "", CLEAN_BUCKET, dest_prefix)
            context.log.info(f"Moved {len(moved)} objects to clean lake.")
    else:
        # Move to quarantine
        dest_prefix = f"quarantined/{timestamp}/"
        context.log.info(f"❌ FAILED — Moving data to s3://{QUARANTINE_BUCKET}/{dest_prefix}")
        if parquet_objects:
            moved = minio.move_prefix(INCOMING_BUCKET, "", QUARANTINE_BUCKET, dest_prefix)
            context.log.info(f"Moved {len(moved)} objects to quarantine.")

    # --- Step 4: Log to PostgreSQL ---
    run_id = postgres.log_pipeline_run(
        batch_name=batch_name,
        status=status,
        drift_score=validation_result.get("drift_score"),
        failed_tests=validation_result.get("failed_tests", 0),
        total_tests=validation_result.get("total_tests", 0),
        severity=validation_result.get("severity"),
        evidently_report_path=report_paths.get("drift_report"),
        records_processed=validation_result.get("records_processed"),
        execution_time_seconds=validation_result.get("execution_time_seconds"),
        details={
            "drifted_columns": validation_result.get("drifted_columns", []),
            "report_paths": report_paths,
        },
    )

    context.log.info(f"Logged pipeline run: {run_id}")

    # Add metadata for Dagster UI
    context.add_output_metadata(
        {
            "run_id": run_id,
            "status": status,
            "destination": CLEAN_BUCKET if passed else QUARANTINE_BUCKET,
            "drift_report": MetadataValue.url(report_paths.get("drift_report", "")),
        }
    )

    return {
        "run_id": run_id,
        "status": status,
        "destination": CLEAN_BUCKET if passed else QUARANTINE_BUCKET,
        "report_paths": report_paths,
    }
