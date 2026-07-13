# 🔍 MLops Data Quality Monitoring Pipeline

[![CI — Lint, Test & Build](https://github.com/MaxLMagnusson/mlops-data-quality-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxLMagnusson/mlops-data-quality-monitoring/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Dagster](https://img.shields.io/badge/Dagster-1.9-purple?logo=dagster&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently_AI-0.6-green)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-S3--Compatible-C72E49?logo=minio&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)

A data quality monitoring system for autonomous driving datasets, built to detect data drift, enforce schema validation, and automatically filter corrupt data, all orchestrated with Dagster and running entirely on Docker.

---

## 📐 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Drift         │     │     MinIO         │     │   Dagster          │
│   Simulator     │────▶│  /incoming-data   │────▶│   Sensor           │
│  (corrupt_data) │     │                   │     │   (auto-trigger)   │
└─────────────────┘     └──────────────────┘     └────────┬───────────┘
                                                          │
                        ┌──────────────────┐              ▼
                        │   MinIO          │     ┌────────────────────┐
                        │  /baseline       │────▶│   Evidently AI     │
                        │                  │     │   Validation       │
                        └──────────────────┘     │   Engine           │
                                                 └────────┬───────────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          │               │               │
                                          ▼               ▼               ▼
                                   ┌─────────────┐ ┌───────────┐ ┌──────────────┐
                                   │  ✅ PASSED   │ │ ❌ FAILED  │ │  PostgreSQL  │
                                   │  /clean-     │ │ /quaran-  │ │  Run Logs    │
                                   │  production  │ │  tine     │ │              │
                                   │  -lake       │ │           │ │              │
                                   └─────────────┘ └───────────┘ └──────────────┘
```

### Dagster Asset Graph

```
baseline_data ──┐
                ├──▶ validation_result ──▶ route_data
incoming_batch ─┘
```

---

## 🛠️ Technology Stack

| Component            | Technology         | Purpose                                      |
|----------------------|--------------------|----------------------------------------------|
| Orchestration        | **Dagster**        | Pipeline DAG, sensors, scheduling, UI        |
| Validation           | **Evidently AI**   | Statistical drift detection, quality tests   |
| Object Storage       | **MinIO**          | S3-compatible data lake (local)              |
| Metadata DB          | **PostgreSQL**     | Run history, drift logs, metrics             |
| Data Processing      | **pandas + NumPy** | Data manipulation and analysis               |
| Image Processing     | **OpenCV**         | Camera fault simulation                      |
| Infrastructure       | **Docker Compose** | One-command environment setup                |
| CI/CD                | **GitHub Actions** | Linting (Ruff), testing (pytest), Docker build |

---

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Python 3.11+ (for local development / running scripts)

### 1. Clone and Configure

```bash
git clone https://github.com/YOUR_USERNAME/mlops-data-quality-monitoring.git
cd mlops-data-quality-monitoring
cp .env.example .env
```

### 2. Start Infrastructure

```bash
docker compose up -d
```

This starts 5 services:
- **PostgreSQL** on port `5432`
- **MinIO** API on port `9000`, Console on port `9001`
- **Dagster UI** on port `3000`
- **Dagster Daemon** (background)
- **User Code Server** (gRPC on port `4000`)

### 3. Seed Baseline Data

```bash
pip install -r docker/pipeline/requirements.txt
python scripts/seed_baseline.py
```

### 4. Simulate Data Drift

```bash
# Medium distribution shift — triggers drift detection
python scripts/simulate_drift.py --profile distribution_shift --severity 0.5

# Severe camera fault — triggers quarantine
python scripts/simulate_drift.py --profile camera_fault --severity 0.9

# Clean data — should pass validation
python scripts/simulate_drift.py --profile none
```

### 5. Monitor Results

- **Dagster UI**: [http://localhost:3000](http://localhost:3000) — view pipeline runs, asset graph
- **MinIO Console**: [http://localhost:9001](http://localhost:9001) — browse data lake buckets
  - Login: `minioadmin` / `minioadmin123`

---

## Data Corruption Profiles

The drift simulator supports 4 corruption profiles, each mimicking real-world data quality issues:

| Profile              | What It Simulates                        | Columns Affected                        |
|----------------------|------------------------------------------|-----------------------------------------|
| `camera_fault`       | Dirty lens, exposure errors              | visibility, num_lidar_pts, num_radar_pts |
| `sensor_missing`     | Intermittent sensor failures             | translation_x/y/z, bbox dimensions     |
| `schema_break`       | Upstream ETL bugs, type mismatches       | Float columns → string injection        |
| `distribution_shift` | Calibration drift, environmental change  | Numeric columns (scale + offset)        |

Severity parameter (0.0–1.0) controls corruption intensity:
- `0.0`: No corruption (clean data)
- `0.3`: Subtle drift (might pass validation)
- `0.5`: Moderate drift (typically caught)
- `0.8+`: Severe corruption (always caught, HIGH/CRITICAL severity)

---

## Validation Engine

The validation engine uses **Evidently AI** to perform two types of checks:

### Data Drift Detection
- **Kolmogorov-Smirnov test** for continuous numeric features
- **Chi-squared test** for categorical features
- Per-column drift detection with configurable p-value thresholds
- Overall drift score (fraction of drifted columns)

### Data Quality Tests
- Column type validation (schema consistency)
- Missing value thresholds
- Value range enforcement
- Duplicate detection

### Severity Classification

| Severity   | Drift Score | Test Fail Rate | Action          |
|------------|-------------|----------------|-----------------|
| LOW        | < 10%       | < 5%           | Log only        |
| MEDIUM     | 10–30%      | 5–15%          | Log + alert     |
| HIGH       | 30–60%      | 15–40%         | Quarantine data |
| CRITICAL   | > 60%       | > 40%          | Quarantine data |

---

## Project Structure

```
├── docker-compose.yml           # One-command infrastructure
├── .github/workflows/ci.yml    # CI/CD pipeline
├── docker/
│   ├── dagster/                 # Dagster webserver/daemon image
│   └── pipeline/                # Pipeline code image
├── src/
│   ├── assets/                  # Dagster assets (pipeline logic)
│   │   ├── ingestion.py         # Data loading + MinIO sensor
│   │   ├── validation.py        # Evidently AI engine
│   │   └── routing.py           # Pass/fail routing
│   ├── resources/               # External service connections
│   │   ├── minio_resource.py    # S3-compatible storage client
│   │   └── postgres_resource.py # Pipeline run logger
│   ├── drift_simulator/         # Data corruption engine
│   │   └── corrupt_data.py      # 4 corruption profiles
│   ├── utils/
│   │   └── data_loader.py       # Synthetic data generator
│   └── definitions.py           # Dagster entry point
├── scripts/
│   ├── seed_baseline.py         # Initialize reference data
│   └── simulate_drift.py       # CLI drift injector
├── tests/                       # pytest test suite
├── sql/init.sql                 # PostgreSQL schema
└── reports/                     # Generated Evidently reports
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -r docker/pipeline/requirements.txt
pip install pytest pytest-cov ruff

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Linting
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
```

---

## Using Real nuScenes Data

This project uses synthetic data by default for portability. To use real nuScenes data:

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download)
2. Download the nuScenes-mini dataset
3. Place it in `data/nuscenes-mini/`
4. Update `src/utils/data_loader.py` to load from the real dataset using `nuscenes-devkit`:

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes-mini/')
```

The pipeline code requires **no changes** — the validation engine works on any DataFrame with the same schema.

---

## License

MIT License
