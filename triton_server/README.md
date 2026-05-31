# Triton Inference Server

This is a shared Triton Inference Server for the MultiModal-Detection project, supporting multiple AI services including ANPR, face detection, and drone object detection.

## Project Structure

```
MultiModal-Detection/
├── common_library/              # Shared dependencies and utilities
├── services/                   # Individual service implementations
│   ├── anpr_object/           # ANPR (Automatic Number Plate Recognition) service
│   ├── face/                  # Face detection service
│   └── drone/                 # Drone object detection service
└── triton_server/             # Shared Triton Inference Server
    ├── paddle_wheels/         # PaddlePaddle custom wheels (Triton-specific)
    ├── model_repository/      # Model configurations and weights
    │   ├── plate_region_detection_rt_detr/
    │   │   ├── config.pbtxt   # Triton model configuration
    │   │   └── 1/
    │   │       └── model.pt   # TorchScript weights
    │   ├── vehicle_detection_rt_detr/
    │   │   ├── config.pbtxt
    │   │   └── 1/
    │   │       └── model.pt
    │   ├── retinaface/        # Face detection model (disabled)
    │   ├── object_detection/  # General object detection
    │   └── paddle_ocr_gpu/     # PaddleOCR model (disabled on macOS)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md
```

> **Naming reminders:** Model folders map to Triton model IDs; version folders must be numeric; the PyTorch backend only loads `model.pt` (filename is fixed).

## 1. Create/refresh config files (CPU-only example)

```bash
cat <<'EOF' > model_repository/plate_region_detection_rt_detr/config.pbtxt
name: "plate_region_detection_rt_detr"
platform: "pytorch_libtorch"
max_batch_size: 0
instance_group [ { kind: KIND_CPU } ]
EOF

cat <<'EOF' > model_repository/vehicle_detection_rt_detr/config.pbtxt
name: "vehicle_detection_rt_detr"
platform: "pytorch_libtorch"
max_batch_size: 0
instance_group [ { kind: KIND_CPU } ]
EOF
```

### Switching to GPU (per model)

For GPU execution change the `instance_group` block to:

```
instance_group [
  {
    kind: KIND_GPU
    count: 1          # number of copies to keep on the card
    gpus: [0]         # GPU IDs to target
  }
]
```

You may keep one model on GPU and another on CPU by editing each config separately.

## 2. Ensure your `model.pt` files are TorchScript

Triton expects TorchScript modules, not raw `state_dict`s. Export using your training codebase:

```python
model.eval()
example = torch.randn(1, 3, 640, 640)
scripted = torch.jit.trace(model, example)
scripted.save("model_plate.pt")
```

If you're using the Ultralytics YOLO CLI, export the RT-DETR checkpoint twice—once for CUDA and once for CPU—to produce TorchScript weights:

```bash
# GPU (CUDA device 0)
CUDA_VISIBLE_DEVICES=0 yolo export \
  model=RTDETR_pretrained_model-l.pt \
  format=torchscript device=0 half=False

yolo export model=RTDETR_pretrained_model-l.pt format=onnx dynamic=False

# CPU (no GPU visibility)
CUDA_VISIBLE_DEVICES="" yolo export \
  model=vehicle_detection_RT-DETR_best.pt \
  format=torchscript device=cpu half=False
```

Run whichever command matches the hardware you have available when converting. The resulting `*.torchscript` file should be renamed/moved to `model_repository/<model>/1/model.pt`.

> **Input resolution guidance:** RT-DETR blocks require both spatial dimensions to be divisible by 32. When you need a larger field of view, set `imgsz=<width>,<height>` during export (e.g. `imgsz=640,640` or `imgsz=1280,704`). After exporting, update the matching `config.pbtxt` `dims` and every client-side preprocessing target. We observed that bumping from 640×640 to ~1280-wide inputs can reduce recall on plates unless the checkpoint was trained/fine-tuned at that resolution, so validate accuracy before promoting bigger tensors.

Example CPU export with explicit shape:

```bash
CUDA_VISIBLE_DEVICES="" yolo export \
  model=plate_region_detection-RT-DETR_weights_best.pt \
  format=torchscript \
  device=cpu \
  imgsz=640,640 \
  half=False
```

Switch `imgsz` to the exact height/width you intend to serve and keep both numbers multiples of 32 to avoid TorchScript runtime errors.

To verify an existing file locally:

```python
import torch
from pathlib import Path

path = Path("model/vehicle_detection_RT-DETR_best.pt")
try:
    torch.jit.load(path)
except Exception as exc:
    raise SystemExit(f"NOT TorchScript: {exc}")
print("TorchScript OK")
```

*(PyTorch must be installed in the environment where you run this test.)*

## 3. Launch Triton (choose CPU or GPU)

All commands run from the project root directory.

### CPU-only container

```bash
docker run --rm -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

To enable per-request and batch-level logging for debugging, add `--log-verbose=1` (higher numbers produce even more detail):

```bash
docker run --rm -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### GPU container

```bash
docker run --rm --gpus all -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

#### Run on custom port

```bash
docker run --rm --gpus all \
  -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models \
  --http-port=9000 \
  --grpc-port=9001 \
  --metrics-port=9002 \
  --log-verbose=1
```

#### Run on custom port with daemon mode

```bash
docker run -d --rm --gpus all \
  -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v "$(pwd)/triton_server/model_repository:/models" \
  --name triton-server \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models \
               --http-port=9000 \
               --grpc-port=9001 \
               --metrics-port=9002 \
               --log-verbose=1
```

> **Prerequisite:** Install the NVIDIA Container Toolkit (`nvidia-smi` must work inside containers) so the `--gpus all` flag is honored.

## 4. Control which model versions are served

Triton defaults to the **latest** numeric subfolder (highest version). Override that behavior in each `config.pbtxt` using `version_policy`:

1. **Latest only** (keep the highest numbered folder active):
   ```
   version_policy: { latest { num_versions: 1 } }
   ```
2. **Specific versions** (pin to `1` even if `2/` exists):
   ```
   version_policy: { specific { versions: [1] } }
   ```
3. **All versions** (serve every numbered folder simultaneously):
   ```
   version_policy: { all { } }
   ```

### Hot swapping with polling

To publish a new version without restarting the container, run Triton with polling enabled:

```bash
docker run --rm -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v $(pwd)/triton_server/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --model-control-mode=poll
```

Now you can drop `triton_server/model_repository/<model>/2/model.pt`, update `version_policy`, and Triton will pick up changes on the next poll interval (default 60s).

## 5. Health checks & testing

1. Verify server is up:
   ```bash
   curl -s http://localhost:9000/v2/health/ready
   ```
2. List loaded models:
   ```bash
   curl -s http://localhost:9000/v2/models
   ```
3. Send an inference request (example uses `tritonclient[http]`):
   ```python
   from tritonclient.http import InferenceServerClient, InferInput
   import numpy as np

   client = InferenceServerClient(url="localhost:9000")
   inp = InferInput("images", [1, 3, 640, 640], "FP32")
   data = np.random.rand(1, 3, 640, 640).astype(np.float32)
   inp.set_data_from_numpy(data)
   result = client.infer("vehicle_detection_rt_detr", inputs=[inp])
   print(result.as_numpy("output0").shape)
   ```

## 6. Client integration contract (important)

The ANPR service (located in `services/anpr_object/`) expects **both** models to be available and configured with these names:

- `vehicle_detection_rt_detr`
- `plate_region_detection_rt_detr`

Client-side `.env` example (for services/anpr_object/):

```env
TRITON_SERVER_URL="host.docker.internal:9001"
TRITON_PROTOCOL="grpc"
VEHICLE_MODEL_NAME="vehicle_detection_rt_detr"
PLATE_MODEL_NAME="plate_region_detection_rt_detr"
VEHICLE_CLASS_ID_MAP='{"0":"vehicle"}'
PLATE_CLASS_ID_MAP='{"0":"number_plate"}'
OCR_CLASS_NAMES="number_plate"
```

For other services (face, drone), refer to their respective configuration files in `services/face/` and `services/drone/`.

### Output shape notes

- Plate model currently serves `output0` with `dims: [300, 6]`.
- Vehicle model currently serves `output0` with `dims: [300, 19]`.

If either output tensor name or dimensions change, update both Triton `config.pbtxt` and client post-processing logic.

## 7. Operational notes for new teammates

1. **Versioning:** Keep previous exports by adding `triton_server/model_repository/<model>/<version>/model.pt`. Triton can serve multiple versions simultaneously.
2. **CPU vs GPU:** Update `instance_group` accordingly. On CPU, latency is higher; tweak `OMP_NUM_THREADS` via Docker env vars if needed.
3. **Logging:** `--log-verbose=1` helps debug load errors. Triton prints detailed reasons if `model.pt` is malformed.
4. **Pre/Post-processing:** Pure PyTorch models expect normalized tensors. For additional logic, add a Python backend or create an ensemble model.
5. **Security:** Expose ports only within trusted networks; add TLS/proxies for production use.
6. **Backend trade-offs:** TorchScript (libtorch backend) is the most general option—it runs anywhere Triton does and keeps control-flow identical to PyTorch, but you pay a latency/throughput tax versus graph-optimized formats. If you need maximum performance and have stable ops, prefer ONNX + TensorRT (or pure TensorRT) exports; they compile the graph, fuse kernels, and deliver lower latency, especially on GPU. Maintain TorchScript for quick iteration or CPU-only deployments, and graduate to ONNX/TensorRT once the model architecture is frozen.
7. **Multi-service architecture:** This Triton server is shared across multiple services (ANPR, face, drone). Each service has its own directory in `services/` with its own configuration and dependencies. The Triton server provides a unified inference backend for all services.
8. **Dependencies:** PaddlePaddle wheels are stored in `triton_server/paddle_wheels/` since they are specific to the Triton server deployment. Individual services reference these wheels in their `pyproject.toml` files using the relative path `../triton_server/paddle_wheels`.
