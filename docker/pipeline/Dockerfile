FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline source code
COPY src/ /opt/dagster/app/src/

WORKDIR /opt/dagster/app

# Set PYTHONPATH so Dagster can find our modules
ENV PYTHONPATH=/opt/dagster/app

# Expose gRPC port for Dagster code location
EXPOSE 4000

# Run Dagster gRPC server serving our definitions
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "src.definitions"]
