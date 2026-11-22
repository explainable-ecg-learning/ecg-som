# ── build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to build wfdb / scipy / neurokit2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY explorer/requirements.txt explorer-requirements.txt

# Install CPU-only torch first (smaller image; use a CUDA base image if GPU is needed)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt -r explorer-requirements.txt

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code (data/models/logs are mounted at runtime)
COPY src/ src/
COPY explorer/ explorer/

ENV PYTHONUNBUFFERED=1
ENV TORCH_COMPILE_DISABLE=1

EXPOSE 8050

CMD ["python", "explorer/server.py", "--host", "0.0.0.0", "--port", "8050"]
