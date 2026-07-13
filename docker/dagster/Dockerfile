FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Dagster components
RUN pip install --no-cache-dir \
    dagster==1.9.* \
    dagster-webserver==1.9.* \
    dagster-postgres==0.25.* \
    dagster-docker==0.25.*

# Set up Dagster home
ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME

# Copy configuration files
COPY dagster.yaml $DAGSTER_HOME/dagster.yaml
COPY workspace.yaml $DAGSTER_HOME/workspace.yaml

WORKDIR $DAGSTER_HOME
