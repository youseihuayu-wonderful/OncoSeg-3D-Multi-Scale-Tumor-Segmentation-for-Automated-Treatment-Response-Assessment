# OncoSeg FastAPI inference service.
#
# Build:
#   docker build -t oncoseg-api .
# Run (CPU, with a local checkpoint mounted):
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/experiments/local_results:/ckpt:ro \
#     -e ONCOSEG_CHECKPOINT=/ckpt/oncoseg_best.pth \
#     oncoseg-api
# Smoke-test:
#   curl http://localhost:8000/healthz
#   curl http://localhost:8000/readyz
#
# Notes:
#   - CPU base image; swap to an nvidia/cuda base for GPU inference.
#   - Checkpoint is mounted, not baked in — keeps the image small and avoids
#     shipping model weights in registry layers.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# MONAI / nibabel / scipy runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project first — dependencies change less often than source code,
# so this layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src ./src
COPY train_all.py ./train_all.py

RUN pip install --upgrade pip && \
    pip install ".[serve]"

# Unprivileged runtime user.
RUN groupadd --system oncoseg && useradd --system --gid oncoseg --home /app oncoseg
RUN chown -R oncoseg:oncoseg /app
USER oncoseg

EXPOSE 8000

# ONCOSEG_CHECKPOINT must be provided at runtime; /readyz reports
# model_loaded=false until it is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
