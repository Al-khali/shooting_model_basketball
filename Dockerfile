# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# Install dependencies into an isolated venv using uv.
# uv is copied from the official image to avoid install-script issues under QEMU.
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS builder

# Copy uv binary from official image (avoids curl + install script under QEMU)
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /uvx /bin/

WORKDIR /build

# Layer 1: resolve + install third-party deps only (cached unless lock changes)
COPY pyproject.toml uv.lock ./
RUN mkdir -p src && touch src/__init__.py
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: install the project itself non-editable so site-packages has a real copy
# (editable installs embed the builder path which doesn't exist in runtime)
COPY LICENSE README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# Lean image: copy venv from builder + source code only.
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS runtime

# libgl1 + libglib2.0-0 required by OpenCV (transitive via ultralytics/mediapipe)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /app

# Copy venv from builder — venv shebang scripts reference /build paths, so we use
# "python -m uvicorn" (the Python binary itself, not the wrapper script) to start.
COPY --from=builder --chown=appuser:appuser /build/.venv /app/.venv

# Copy source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Activate venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cloud Run expects the container to listen on $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

# Health check (Cloud Run also probes GET /health separately)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
