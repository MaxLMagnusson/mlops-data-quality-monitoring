-- =============================================================================
-- MLops Data Quality Monitoring - PostgreSQL Schema
-- =============================================================================
-- Dagster uses its own internal tables; this schema is for our pipeline metadata.

CREATE SCHEMA IF NOT EXISTS monitoring;

-- History of every pipeline validation run
CREATE TABLE IF NOT EXISTS monitoring.pipeline_runs (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64)   NOT NULL UNIQUE,
    run_timestamp   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    batch_name      VARCHAR(255)  NOT NULL,
    status          VARCHAR(32)   NOT NULL CHECK (status IN ('SUCCESS', 'DRIFT_DETECTED', 'ERROR')),
    drift_score     FLOAT,
    failed_tests    INTEGER       DEFAULT 0,
    total_tests     INTEGER       DEFAULT 0,
    severity        VARCHAR(16)   CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', NULL)),
    evidently_report_path VARCHAR(512),
    records_processed     INTEGER,
    execution_time_seconds FLOAT,
    details         JSONB,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Index for fast lookups by status and time
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON monitoring.pipeline_runs (status);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_timestamp ON monitoring.pipeline_runs (run_timestamp DESC);

-- Summary view for dashboard queries
CREATE OR REPLACE VIEW monitoring.run_summary AS
SELECT
    date_trunc('day', run_timestamp) AS run_date,
    COUNT(*)                         AS total_runs,
    COUNT(*) FILTER (WHERE status = 'SUCCESS')        AS passed,
    COUNT(*) FILTER (WHERE status = 'DRIFT_DETECTED') AS failed,
    COUNT(*) FILTER (WHERE status = 'ERROR')          AS errors,
    AVG(drift_score)                 AS avg_drift_score,
    AVG(execution_time_seconds)      AS avg_execution_time
FROM monitoring.pipeline_runs
GROUP BY date_trunc('day', run_timestamp)
ORDER BY run_date DESC;
