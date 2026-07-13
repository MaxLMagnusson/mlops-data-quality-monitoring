"""
MinIO Resource — S3-compatible Object Storage Client
=====================================================
Dagster ConfigurableResource wrapping boto3 for MinIO interactions.
Handles bucket management, file uploads/downloads, and object operations
across the data lake buckets.

Buckets:
- incoming-data:         New batch data awaiting validation
- clean-production-lake: Validated data that passed quality checks
- quarantine:            Data that failed validation (drift detected)
- reports:               Evidently HTML/JSON reports
- baseline:              Reference dataset for comparison
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Optional

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

logger = logging.getLogger(__name__)


class MinIOResource(ConfigurableResource):
    """Dagster resource for interacting with MinIO object storage."""

    endpoint_url: str = "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    region_name: str = "us-east-1"

    def _get_client(self) -> boto3.client:
        """Create a boto3 S3 client configured for MinIO."""
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name=self.region_name,
        )

    def create_bucket_if_not_exists(self, bucket: str) -> None:
        """Create a bucket if it doesn't already exist."""
        client = self._get_client()
        try:
            client.head_bucket(Bucket=bucket)
            logger.info(f"Bucket '{bucket}' already exists.")
        except ClientError:
            client.create_bucket(Bucket=bucket)
            logger.info(f"Created bucket '{bucket}'.")

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str,
        format: str = "parquet",
    ) -> str:
        """
        Upload a pandas DataFrame to MinIO.

        Parameters
        ----------
        df : pd.DataFrame
            Data to upload.
        bucket : str
            Target bucket name.
        key : str
            Object key (path within bucket).
        format : str
            Serialization format ('parquet' or 'csv').

        Returns
        -------
        str
            The S3 URI of the uploaded object.
        """
        client = self._get_client()
        buffer = io.BytesIO()

        if format == "parquet":
            df.to_parquet(buffer, index=False, engine="pyarrow")
        elif format == "csv":
            buffer = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported format: {format}")

        buffer.seek(0)
        client.upload_fileobj(buffer, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded DataFrame ({len(df)} rows) to {uri}")
        return uri

    def download_dataframe(
        self,
        bucket: str,
        key: str,
        format: str = "parquet",
    ) -> pd.DataFrame:
        """
        Download a DataFrame from MinIO.

        Parameters
        ----------
        bucket : str
            Source bucket name.
        key : str
            Object key.
        format : str
            File format ('parquet' or 'csv').

        Returns
        -------
        pd.DataFrame
        """
        client = self._get_client()
        buffer = io.BytesIO()
        client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        if format == "parquet":
            return pd.read_parquet(buffer, engine="pyarrow")
        elif format == "csv":
            return pd.read_csv(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def upload_file(self, local_path: str, bucket: str, key: str) -> str:
        """Upload a local file to MinIO."""
        client = self._get_client()
        client.upload_file(local_path, bucket, key)
        uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded {local_path} to {uri}")
        return uri

    def download_file(self, bucket: str, key: str, local_path: str) -> str:
        """Download a file from MinIO to a local path."""
        client = self._get_client()
        client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
        return local_path

    def upload_json(
        self, data: dict[str, Any], bucket: str, key: str
    ) -> str:
        """Upload a JSON object to MinIO."""
        client = self._get_client()
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded JSON to {uri}")
        return uri

    def upload_html(self, html_content: str, bucket: str, key: str) -> str:
        """Upload HTML content (e.g., Evidently report) to MinIO."""
        client = self._get_client()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=html_content.encode("utf-8"),
            ContentType="text/html",
        )
        uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded HTML report to {uri}")
        return uri

    def list_objects(
        self, bucket: str, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """
        List objects in a bucket, optionally filtered by prefix.

        Returns
        -------
        list[dict]
            List of objects with 'Key', 'Size', 'LastModified'.
        """
        client = self._get_client()
        try:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return response.get("Contents", [])
        except ClientError as e:
            logger.error(f"Error listing objects in {bucket}/{prefix}: {e}")
            return []

    def move_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> str:
        """Move an object between buckets (copy + delete)."""
        client = self._get_client()

        # Copy to destination
        copy_source = {"Bucket": source_bucket, "Key": source_key}
        client.copy_object(
            Bucket=dest_bucket,
            Key=dest_key,
            CopySource=copy_source,
        )

        # Delete from source
        client.delete_object(Bucket=source_bucket, Key=source_key)

        dest_uri = f"s3://{dest_bucket}/{dest_key}"
        logger.info(
            f"Moved s3://{source_bucket}/{source_key} → {dest_uri}"
        )
        return dest_uri

    def move_prefix(
        self,
        source_bucket: str,
        source_prefix: str,
        dest_bucket: str,
        dest_prefix: str,
    ) -> list[str]:
        """Move all objects under a prefix to another location."""
        objects = self.list_objects(source_bucket, source_prefix)
        moved = []

        for obj in objects:
            source_key = obj["Key"]
            # Replace the source prefix with dest prefix
            relative_key = source_key[len(source_prefix):]
            dest_key = dest_prefix + relative_key
            self.move_object(source_bucket, source_key, dest_bucket, dest_key)
            moved.append(f"s3://{dest_bucket}/{dest_key}")

        return moved

    def object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists in a bucket."""
        client = self._get_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
