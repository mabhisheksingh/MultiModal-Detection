# MultiModal-Detection

MultiModal-Detection is a comprehensive computer vision platform for face detection, license plate recognition (ANPR), and object detection. It splits responsibilities into deployable projects for specialized detection and inference tasks. The workflow has been modernized to output standardized JSON detection payloads in place of legacy CSV files.

## Repository Layout

```text
MultiModal-Detection/
├── FACE_DETECTION/
│   └── retinaface/                # Face detection models and services
├── PLATE_AND_OBJECT_DETECTION/
│   ├── triton_server/             # Triton configs, model repository for ANPR/Vehicles
│   └── anpr-inference-service/    # FastAPI service, tracking, OCR pipelines
└── README.md                      # Overview (this file)
```

Each subfolder contains its own README / guides for deeper setup steps.

## New Code Flow & JSON Output

Instead of outputting `consolidated.csv`, the new processing flow exports a standardized JSON array. This format unified face detection, object detection, and license plate tracking across all services.

### Standardized JSON Format

During inference, each detection generates an object adhering to the following structure. The `sources` array contains the specific model name that generated the detection (e.g., `"retinaface"`, `"rt-detr"`, `"yolov8"`):

```json
[
  {
    "frame_id": 148,
    "ts_ms": null,
    "track_id": "8E31DDC3",
    "cls": null,
    "name": "face",
    "conf": 0.32082128524780273,
    "bbox_xyxy": [
      200.56854248046875,
      -28.52631187438965,
      625.0314331054688,
      263.47369384765625
    ],
    "polygon": null,
    "area_px": 123943,
    "center": [
      412.79998779296875,
      117.47369384765625
    ],
    "velocity": [
      0.0,
      0.0
    ],
    "direction": "stationary",
    "sources": [
      "retinaface"
    ]
  }
]
```

### Implementing the JSON Flow

If you are updating your inference services (like `anpr_service.py`) to produce this JSON:
1. **Accumulate Detections:** Rather than flattening properties for a CSV, map each bounding box, tracker ID, and class name into a Python dictionary that mirrors the JSON structure above.
2. **Calculate Missing Fields:** Use the `bbox_xyxy` to compute `area_px` and `center` coordinates. Default `velocity` to `[0.0, 0.0]` and `direction` to `"stationary"` unless explicitly tracked across frames.
3. **Include the Model Source:** Grab the active model name from your settings/configuration (e.g., `settings.triton_model_name` or hardcoded per service pipeline) and append it to the `"sources"` list.
4. **Serialize and Save:** Use Python's built-in `json` module (`json.dump`) to write the list of dictionaries to a `.json` file in the output directory instead of using the `csv.DictWriter`.

## Goals

- **Unified Outputs:** Standardize detection events across different models via the JSON format.
- **Triton for Inference:** Keep Triton focused on serving models (TorchScript/ONNX) with dynamic batching.
- **Client Orchestration:** Let the client layer manage ByteTrack assignments, PaddelOCR extractions, and API endpoints.

## Quick Start

### 1. Launch Triton (Plate/Object Detection)

```bash
cd PLATE_AND_OBJECT_DETECTION/triton_server
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### 2. Run the Inference Service

```bash
cd ../anpr-inference-service
cp .env.example .env   # Set your API keys, model names, and DB configurations
uv run local_setup.py  # Install dependencies
uv run uvicorn app.main:app --reload
```

Use this root README as the map; jump into the subprojects you’re working on for step-by-step details. Happy tagging!
