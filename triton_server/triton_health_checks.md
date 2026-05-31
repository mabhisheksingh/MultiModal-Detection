# Triton Server Quick Links

All endpoints assume the server is reachable at `http://localhost` using the default Triton ports from your log (
HTTP `8000`, gRPC `8001`, metrics `8002`). Replace `localhost` with the actual IP if you are calling remotely.

## Basic Health

| Check | Description | Try It |
| --- | --- | --- |
| Ready | Returns `200 OK` when Triton is ready to serve requests. | [Ready](http://localhost:8000/v2/health/ready) |
| Live | Returns `200 OK` when the server process is alive (even if models are still loading). | [Live](http://localhost:8000/v2/health/live) |
| Metadata | Lists server version, extensions, and supported protocols. | [Server Metadata](http://localhost:8000/v2) |

## Model Introspection

| Check | Description | Try It |
| --- | --- | --- |
| Loaded models | Enumerates every model currently loaded. | [Models](http://localhost:8000/v2/models) |
| Specific model config | Shows configuration for a model (replace `<model>`). | [plate_region_detection_rt_detr](http://localhost:8000/v2/models/plate_region_detection_rt_detr) |
| Model statistics | Latency and throughput counters per model+version. Replace `<model>`/`<version>`. | [plate_region_detection stats](http://localhost:8000/v2/models/plate_region_detection_rt_detr/stats) |

## Metrics & Monitoring

| Check | Description | Try It |
| --- | --- | --- |
| Prometheus metrics | Raw metrics feed (requests, queue, compute times, memory, etc.). | [Metrics](http://localhost:8002/metrics) |

## Handy `curl` snippets

```bash
# Ready
curl -s http://localhost:8000/v2/health/ready

# List models
curl -s http://localhost:8000/v2/models | jq

# Model statistics
curl -s http://localhost:8000/v2/models/plate_region_detection_rt_detr/stats | jq

# Metrics (Prometheus text)
curl -s http://localhost:8002/metrics
```

> Tip: If you expose Triton outside localhost, ensure the ports (8000/8001/8002) are allowed through firewalls and consider placing the server behind TLS or a reverse proxy.

## What you should see in verbose startup logs

The sample verbose run (`--log-verbose=1`) prints useful checkpoints:

- Startup banner shows Triton 2.39.0 and port bindings (HTTP 8000, gRPC 8001, metrics 8002).
- If no GPU is available or the NVIDIA toolkit is missing, you’ll see warnings like `WARNING: The NVIDIA Driver was not detected` and `Unable to allocate pinned system memory`. On CPU-only hosts this is expected; for GPU use install NVIDIA Container Toolkit.
- Each model is auto-completed and loaded: look for `loading: <model>:1` followed by `successfully loaded '<model>'`.
- Ready table lists both models as `READY` with version `1`.
- Service start lines confirm endpoints are listening:
  - `Started HTTPService at 0.0.0.0:8000`
  - `Started GRPCInferenceService at 0.0.0.0:8001`
  - `Started Metrics Service at 0.0.0.0:8002`
- Incoming requests are logged, e.g., `HTTP request: 0 /v2/health/ready`.

If the server doesn’t reach the “READY” table or the services don’t start, revisit `config.pbtxt`, model paths, and GPU drivers (if applicable).
