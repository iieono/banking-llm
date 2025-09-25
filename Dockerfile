# Multi-stage optimized Docker build for BankingLLM system

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Base runtime
FROM python:3.11-slim as base

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Create data directory with correct permissions
RUN mkdir -p data/exports && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser task.md README.md CLAUDE.md ./

# Stage 3: Database generator
FROM base as db-generator

# Switch to root for setup, then back to appuser for generation
USER root

# Ensure proper directory setup and permissions
RUN mkdir -p /app/data/exports && chown -R appuser:appuser /app

# Switch to app user for database generation
USER appuser

# Set environment for database generation
ENV DATABASE_URL=sqlite:///./data/bank.db
ENV PYTHONPATH=/app

# Generate the database with 1M+ records
# This runs during Docker build, so the database is pre-built
RUN echo "Generating database with 1M+ records..." && \
    python -m src.cli setup && \
    echo "Database generation completed!" && \
    ls -la data/

# Verify database was created successfully
RUN python -c "
import sqlite3
import os
if os.path.exists('data/bank.db'):
    conn = sqlite3.connect('data/bank.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM transactions')
    count = cursor.fetchone()[0]
    print(f'Database verified: {count} transactions')
    conn.close()
    if count < 100000:
        raise Exception(f'Database too small: {count} transactions')
else:
    raise Exception('Database file not found')
"

# Stage 4: API service
FROM base as api

# Copy pre-built database from generator stage
COPY --from=db-generator --chown=appuser:appuser /app/data ./data/

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Verify database exists and has data
RUN python -c "
import sqlite3
import os
assert os.path.exists('data/bank.db'), 'Database file missing'
conn = sqlite3.connect('data/bank.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM transactions')
count = cursor.fetchone()[0]
print(f'API container verified: {count} transactions ready')
conn.close()
"

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# API command
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Web service
FROM base as web

# Copy pre-built database from generator stage
COPY --from=db-generator --chown=appuser:appuser /app/data ./data/

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Verify database exists and has data
RUN python -c "
import sqlite3
import os
assert os.path.exists('data/bank.db'), 'Database file missing'
conn = sqlite3.connect('data/bank.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM transactions')
count = cursor.fetchone()[0]
print(f'Web container verified: {count} transactions ready')
conn.close()
"

# Health check for web (check if streamlit is responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Web command
CMD ["streamlit", "run", "src/web.py", "--server.port=8501", "--server.address=0.0.0.0"]