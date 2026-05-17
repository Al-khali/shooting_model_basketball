# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# Install dependencies into an isolated venv using uv.
# uv is copied from the official image to avoid install-script issues under QEMU.
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS builder

# Copy uv binary from official image (avoids curl + install script under QEMU)
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /uvx /bin/

# System libs required to import cv2 during the YOLO weights pre-fetch below.
# `ultralytics` transitively pulls `opencv-contrib-python` (full GUI build),
# which links against libxcb / libGL / libglib at module-load time. Without
# these, even `from ultralytics import YOLO` raises ImportError. They are
# already installed in the runtime stage; mirror them here for the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

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

# Layer 3 — pre-fetch the YOLO pose weights so they ship inside the image.
# Without this, ultralytics downloads ~6 MB from GitHub on the FIRST /analyze
# call of every fresh Cloud Run instance (cold start). With scale-to-zero the
# project pays this cost on most requests; baking the weights eliminates it
# entirely and cuts cold-start latency by ~500ms–2s.
#
# `yolo11n-pose.pt` is the default checkpoint used by PoseEstimator
# (src/perception/pose_estimator.py:116). YOLO() looks up the relative path
# against the current working directory at runtime, which is /app — copied
# below in the runtime stage.
RUN .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# Lean image: copy venv from builder + source code only.
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.12-slim AS runtime

# libgl1 + libglib2.0-0 required by OpenCV (transitive via ultralytics/mediapipe).
# `upgrade` picks up published Debian security updates not yet in the base
# layer (e.g. libcap2 CVE-2026-4878, libsystemd0 CVE-2026-29111 surfaced by
# Trivy on the python:3.12-slim image). Trivy gates the build on HIGH/CRITICAL.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
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

# Pre-baked YOLO weights from the builder stage (eliminates cold-start download)
COPY --from=builder --chown=appuser:appuser /build/yolo11n-pose.pt ./yolo11n-pose.pt

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
