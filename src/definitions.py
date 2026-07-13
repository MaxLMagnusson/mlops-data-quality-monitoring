"""
Dagster Definitions — Pipeline Entry Point
============================================
Central configuration that registers all assets, sensors, and resources
with the Dagster framework. This is the module loaded by the Dagster
gRPC code server.

Asset Dependency Graph:
    baseline_data ──┐
                    ├──→ validation_result ──→ route_data
    incoming_batch ─┘
"""

import os

from dagster import Definitions

from src.assets.ingestion import baseline_data, incoming_batch, incoming_data_sensor
from src.assets.routing import validation_result, route_data
from src.resources.minio_resource import MinIOResource
from src.resources.postgres_resource import PostgresResource


defs = Definitions(
    assets=[
        baseline_data,
        incoming_batch,
        validation_result,
        route_data,
    ],
    sensors=[
        incoming_data_sensor,
    ],
    resources={
        "minio": MinIOResource(
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        ),
        "postgres": PostgresResource(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "dagster"),
            password=os.getenv("POSTGRES_PASSWORD", "dagster_password"),
            dbname=os.getenv("POSTGRES_DB", "dagster_db"),
        ),
    },
)
