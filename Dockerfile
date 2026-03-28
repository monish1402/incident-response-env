# =============================================================================
# Dockerfile — Incident Response OpenEnv
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) : installs Python dependencies into a venv
#   Stage 2 (runtime) : copies only the venv + source, runs as non-root user
#
# Build:  docker build -t incident-response-env .
# Run:    docker run -p 7860:7860 incident-response-env
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only dependency files first (layer-cache friendly)
COPY pyproject.toml ./

# Install all runtime + baseline dependencies into the venv
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        "openenv-core>=0.0.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "openai>=1.0.0"


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata labels (OCI standard)
LABEL org.opencontainers.image.title="incident-response-env"
LABEL org.opencontainers.image.description="OpenEnv: AI-powered production incident response environment"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install only the curl runtime dependency (for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY server/     ./server/
COPY baseline/   ./baseline/
COPY openenv.yaml ./

# Create a non-root user for security (HF Spaces best practice)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────

# HF Spaces requires port 7860
EXPOSE 7860

# Optional: OpenAI API key for LLM-powered baseline (can be left empty)
ENV OPENAI_API_KEY=""
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Liveness probe — HF Space automated ping checks this
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
