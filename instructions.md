Project Architecture Overview
To visualize how these components interact in a production-grade environment, the data flows from raw storage through an automated orchestration layer, passes through statistical validation, and updates a monitoring state:

Project Outline: The 5 Core Components
Here is the blueprint of the exact parts you need to build. Everything will run locally on your machine using Docker containers.

1. The Ingestion & Simulation Layer (The Data Source)
This layer simulates a production environment where new vehicle data arrives in batches (e.g., nightly drops).

Baseline Registry: A folder containing a clean, untouched slice of the nuScenes-mini dataset (e.g., Day 1 data). This is your reference standard.

The Drift Simulator Script (corrupt_data.py): A Python script that takes another slice of the dataset (e.g., Day 2 data) and artificially corrupts it to simulate real-world errors:

Camera Fault: Reduces brightness/contrast or adds noise to the image arrays using OpenCV.

Sensor Missingness: Randomly drops coordinate values or bounding box dimensions from the metadata files.

2. The Storage Layer (The Target Data Lake)
Where the data lives and where the health logs are kept.

Local Object Storage (MinIO): A free, open-source container that acts exactly like Amazon S3. You will create two buckets here: /incoming-data and /quarantine.

Metadata DB (PostgreSQL): A lightweight relational database that acts as your central ledger, storing the history of every pipeline run, execution times, and whether data drift was detected.

3. The Validation Engine (The Inspector)
This is the mathematical brain of the project that replaces manual testing.

Evidently AI Profiler: A Python module that loads the baseline data from MinIO, loads the new incoming batch, and computes statistical profiles.

Statistical Checks: It runs algorithms (like the Kolmogorov-Smirnov test) to check if the numerical distribution of bounding boxes, sensor coordinates, or image pixel intensities has significantly shifted from the baseline.

Data Quality Suite: Assures schemas match (e.g., checking that a column that should contain float coordinates doesn't suddenly contain text strings).

4. The Orchestration Layer (The Manager)
The skeleton that ties all the scripts together and automates them.

Dagster (or Prefect) Workspace: An orchestration engine that defines your pipeline as a directed graph of tasks (Assets/Ops).

Automated Workflow Steps:

Check /incoming-data bucket for new files.

Trigger the Evidently AI profiling step.

Read the validation output conditional logic:

If Passed: Move data to /clean-production-lake and log SUCCESS to Postgres.

If Failed: Route data to /quarantine and log DRIFT_DETECTED with severity metrics to Postgres.

5. The Infrastructure Layer (The Environment)
Docker Compose Configuration: A single docker-compose.yml file that coordinates the local infrastructure. With one command (docker compose up), it spins up Dagster, MinIO, and PostgreSQL, ensuring network connectivity between them without requiring any local cloud installations.

Interactive Pipeline Simulator & Drift Configurator
To help you conceptualize how data quality rules catch failures and how orchestration logic handles the routing before you write the code, use this interactive sandbox to configure drift profiles and simulate pipeline runs: