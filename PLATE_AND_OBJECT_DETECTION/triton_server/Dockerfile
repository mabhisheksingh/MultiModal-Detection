# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tritonserver:25.10-py3

WORKDIR /opt/tritonserver

ENV MODEL_REPOSITORY=/models \
    TRITON_HTTP_PORT=9000 \
    TRITON_GRPC_PORT=9001 \
    TRITON_METRICS_PORT=9002 \
    TRITON_LOG_VERBOSE=1 \
    PADDLE_HOME=/root/.paddle

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY paddle_wheels /tmp/paddle_wheels
COPY triton_server/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/paddle_wheels /tmp/requirements.txt

EXPOSE 9000 9001 9002

CMD ["tritonserver","--model-repository=/models","--http-port=9000","--grpc-port=9001","--metrics-port=9002","--log-verbose=1","--exit-on-error=false"]