"""
PostgreSQL Resource - Pipeline Run Logger
==========================================
Dagster ConfigurableResource for logging pipeline run metadata to PostgreSQL.
Stores validation results, drift scores, and execution metrics for the
monitoring dashboard.
"""


import json
import logging
import uuid
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from dagster import ConfigurableResource

logger = logging.getLogger(__name__)


class PostgresResource(ConfigurableResource):
    """Dagster resource for logging pipeline runs to PostgreSQL."""

    host: str = "postgres"
    port: int = 5432
    user: str = "dagster"
    password: str = "dagster_password"
    dbname: str = "dagster_db"

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )

    def log_pipeline_run(
        self,
        batch_name: str,
        status: str,
        drift_score: Optional[float] = None,
        failed_tests: int = 0,
        total_tests: int = 0,
        severity: Optional[str] = None,
        evidently_report_path: Optional[str] = None,
        records_processed: Optional[int] = None,
        execution_time_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Log a pipeline validation run to the database.

        Parameters
        ----------
        batch_name : str
            Name/identifier of the data batch.
        status : str
            One of: 'SUCCESS', 'DRIFT_DETECTED', 'ERROR'.
        drift_score : float, optional
            Overall drift score (0.0–1.0).
        failed_tests : int
            Number of failed Evidently tests.
        total_tests : int
            Total number of Evidently tests run.
        severity : str, optional
            Drift severity: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
        evidently_report_path : str, optional
            S3 URI of the Evidently HTML report.
        records_processed : int, optional
            Number of records in the batch.
        execution_time_seconds : float, optional
            Pipeline execution time.
        details : dict, optional
            Additional metadata as JSON.

        Returns
        -------
        str
            Generated run_id.
        """
        run_id = str(uuid.uuid4())[:12]

        query = """
            INSERT INTO monitoring.pipeline_runs (
                run_id, batch_name, status, drift_score, failed_tests,
                total_tests, severity, evidently_report_path,
                records_processed, execution_time_seconds, details
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        run_id,
                        batch_name,
                        status,
                        drift_score,
                        failed_tests,
                        total_tests,
                        severity,
                        evidently_report_path,
                        records_processed,
                        execution_time_seconds,
                        json.dumps(details) if details else None,
                    ),
                )
            conn.commit()

        logger.info(
            f"Logged pipeline run: id={run_id} batch={batch_name} "
            f"status={status} drift_score={drift_score}"
        )
        return run_id

    def get_run_history(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Query pipeline run history.

        Parameters
        ----------
        limit : int
            Maximum number of runs to return.
        status_filter : str, optional
            Filter by status ('SUCCESS', 'DRIFT_DETECTED', 'ERROR').

        Returns
        -------
        list[dict]
            List of run records.
        """
        query = """
            SELECT * FROM monitoring.pipeline_runs
            WHERE 1=1
        """
        params: list[Any] = []

        if status_filter:
            query += " AND status = %s"
            params.append(status_filter)

        query += " ORDER BY run_timestamp DESC LIMIT %s"
        params.append(limit)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [dict(row) for row in rows]

    def get_run_summary(self) -> list[dict[str, Any]]:
        """
        Get aggregated daily run summary from the view.

        Returns
        -------
        list[dict]
            Daily summary with pass/fail counts and average metrics.
        """
        query = "SELECT * FROM monitoring.run_summary LIMIT 30"

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return [dict(row) for row in rows]

    def health_check(self) -> bool:
        """Check if the database is reachable."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
