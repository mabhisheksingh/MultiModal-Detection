FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY anpr-inference-service/requirements.txt ./requirements.txt
COPY paddle_wheels ./paddle_wheels

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir --prefer-binary \
    --find-links ./paddle_wheels \
    --find-links ./paddle_wheels/numpy_1_26_4 \
    -r requirements.txt

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_USE_GPU=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_HOME=/root/.paddle \
    PADDLE_PDX_MODEL_HOME=/root/.paddlex \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY anpr-inference-service/ ./

EXPOSE 9003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9003"]
