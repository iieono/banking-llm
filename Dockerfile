# Multi-stage optimized Docker build for BankingLLM system
# Python Version: 3.11.9 (locked for consistency across environments)

# Stage 1: Build dependencies
FROM python:3.11.9-slim as builder

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
FROM python:3.11.9-slim as base

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

# Copy application code and version files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser task.md README.md CLAUDE.md ./
COPY --chown=appuser:appuser .python-version runtime.txt requirements.txt ./

# Verify Python version consistency
RUN python -c "import sys; \
expected='3.11.9'; \
actual=f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'; \
print(f'Python version verification: Expected {expected}, Got {actual}'); \
assert actual.startswith('3.11.'), f'Python version mismatch! Expected 3.11.x, got {actual}'"

# Switch to non-root user
USER appuser

# Set environment for database operations
ENV DATABASE_URL=sqlite:///./data/bank.db
ENV PYTHONPATH=/app

# Generate database with mock data during build
RUN python -m src.cli setup

# Web service target
FROM base as web

# Expose Gradio port
EXPOSE 8505

# Health check for web
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8505 || exit 1

# Web command
CMD ["python", "src/gradio_app.py"]