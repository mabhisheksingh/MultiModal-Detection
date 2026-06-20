# ANPR Inference Service (FastAPI)

ANPR service that runs **vehicle + plate detection on the same frame** via Triton Inference Server and applies OCR only to configured plate classes.

## Features

- **Dual-model inference** on each frame:
  - `vehicle_detection_rt_detr` - Vehicle detection
  - `plate_region_detection_rt_detr` - License plate detection
- **Combined annotation output** (vehicle + plate boxes on final frame)
- **Class-id dictionary config** for model outputs (`{"id":"class_name"}`)
- **OCR integration** with PaddleOCR (CPU-based to avoid GPU memory issues)
- **Plate candidate filtering** (`PLATE_CANDIDATE_VEHICLE_CLASSES`) - OCR runs only for allowed vehicle classes
- **FastAPI endpoints** for URL-based image/video processing
- **YouTube video support** via `yt-dlp` integration
- **Frame sampling** with configurable `FRAMES_PER_SECOND` for video processing
- **Tracking + CSV outputs** for video flows with ByteTrack
- **Global cross-camera/cross-video tracking** with a persistent `global_id`
- **Plate-text assisted global matching** with OCR-confusion tolerance (`I/1`, `S/5`, `O/0` etc.)
- **Identity refresh over time** so later frames can improve stored plate text, score, and match reason
- **Structured JSON response** with standardized detection format

## Project Structure

```text
anpr-inference-service/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   └── routes/
│   │   │       ├── health.py           # Health check endpoints
│   │   │       └── anpr.py             # Legacy ANPR processing endpoints
│   │   └── v2/
│   │       └── routes/
│   │           └── anpr.py             # Primary structured ANPR input flow
│   ├── core/
│   │   ├── config.py                   # Configuration management
│   │   └── logging.py                  # Logging setup
│   ├── schemas/
│   │   └── anpr.py                     # Pydantic models
│   ├── repository/
│   │   ├── database.py                 # SQLAlchemy DB setup for global tracking
│   │   ├── models.py                   # ORM models for global identities
│   │   └── global_track_repository.py  # Repository layer for global tracking
│   ├── services/
│   │   ├── triton_client.py            # Triton gRPC client
│   │   ├── paddle_ocr_engine.py        # PaddleOCR wrapper
│   │   ├── global_tracking_service.py  # Global identity assignment logic
│   │   ├── video_source_processor.py   # Video stream orchestration
│   │   └── anpr_service.py             # Main ANPR processing logic
│   ├── utils/
│   │   ├── dependencies.py             # Centralized FastAPI dependency aliases
│   │   └── output_serializers.py       # CSV/JSON output formatting
│   └── main.py                         # FastAPI application
├── data/
│   ├── input/                          # Optional local data mount
│   └── output/                         # Output annotated results
├── .env                                # Environment configuration
├── docker-compose.yml                  # Docker Compose setup
├── Dockerfile                          # Multi-stage Docker build
└── pyproject.toml                      # Dependencies (uv)
```

## Installation

1. Install dependencies:

```bash
uv sync
```

1. Copy the example environment file and configure:

```bash
cp .env.example .env
# Edit .env with your Triton server details
```

## Docker Deployment

### Building the Docker Image

The project uses a **multi-stage Dockerfile** to minimize image size:

```bash
# Build the slim runtime image (recommended)
docker build -f anpr-inference-service/Dockerfile -t anpr-inference-service:slim --target runtime .

# Build includes build dependencies, runtime only includes what's needed to run
```

**Image size comparison:**
- Full build stage: ~8-10 GB (includes build tools, compilers)
- Slim runtime stage: ~3-4 GB (runtime dependencies only)

### Running with Docker Compose

**Recommended approach** - handles environment variables correctly and persists model caches:

```bash
cd anpr-inference-service
docker compose up
```

**What it does:**
- Loads environment variables from `.env` (no shell escaping issues)
- Exposes API on port `9003`
- Connects to Triton server via `host.docker.internal:9001`
- Persists PaddleOCR and HuggingFace model caches in named volumes
- Enables GPU access

**Docker image versioning:**
- Edit the `image` tag in `docker-compose.yml` to use different versions (e.g., `v1.0.0`, `latest`, `slim`)
- If the image exists locally, Docker uses it; otherwise it will try to pull from registry
- To build a new image with a custom tag:
  ```bash
  docker build -f anpr-inference-service/Dockerfile -t anpr-inference-service:your-tag --target runtime ..
  ```

**Persistent cache volumes:**
- `anpr_inference_service_paddle_cache` - PaddleOCR models
- `anpr_inference_service_paddlex_cache` - PaddleX models  
- `anpr_inference_service_hf_cache` - HuggingFace models

**First run:** Downloads models (~30-60s)  
**Subsequent runs:** Uses cached models (instant startup)

To rebuild and restart:

```bash
docker compose down
docker compose build
docker compose up
```

### Running with Docker Run (Alternative)

If you prefer `docker run` over compose:

```bash
docker run --rm --gpus all \
  --name anpr-api \
  -p 9003:9003 \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -e 'VEHICLE_CLASS_ID_MAP={"0":"vehicle"}' \
  -e 'PLATE_CLASS_ID_MAP={"0":"number_plate"}' \
  -v "$(pwd)/data:/app/data" \
  -v anpr_inference_service_paddle_cache:/root/.paddle \
  -v anpr_inference_service_paddlex_cache:/root/.paddlex \
  -v anpr_inference_service_hf_cache:/root/.cache/huggingface \
  anpr-inference-service:slim
```

**Note:** JSON environment variables must be explicitly passed with `-e` and proper quoting to avoid shell escaping issues.

## Usage

### 1. Start API server

Preferred:

```bash
uv run anpr-api
```

Alternative:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Docs: `http://localhost:8000/docs`

#### Main endpoints

- `GET /health`
- `POST /v2/anpr/process` (primary structured multi-input flow)

### New input flow

Use `POST /v2/anpr/process` for all new integrations.

The v2 request accepts a list of structured inputs. Each input describes:

- `id`
- `input_type`
- `options`
- `metadata`

`options` can carry the actual media reference plus analytics metadata such as:

- `uri`
- `content_base64`
- `camera_id`
- `lat`
- `lon`
- `pixels_per_meter`
- `zones`
- `behavior_config`
- `frames`

This flow replaces the older flat `urls`-based request model and is the recommended path going forward. The legacy v1 endpoints will be deprecated in a few days.

#### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Preferred: process a structured video input with analytics metadata
curl -X POST "http://localhost:8000/v2/anpr/process" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "id": "gate-video-01",
        "input_type": "video_url",
        "options": {
          "uri": "https://example.com/sample.mp4",
          "camera_id": "cam_gate_01",
          "lat": 23.2501,
          "lon": 77.4102,
          "pixels_per_meter": 22.5,
          "zones": [
            {
              "zone_id": "entry_lane",
              "zone_type": "entry",
              "coordinates": [[0.1, 0.2], [0.4, 0.2], [0.4, 0.6], [0.1, 0.6]]
            }
          ],
          "behavior_config": {
            "repeat_visit_threshold": 3,
            "linger_threshold_ms": 30000,
            "sensitive_zone_types": ["sensitive", "restricted"],
            "min_behavior_score": 0.6
          }
        },
        "metadata": {
          "site": "north-gate",
          "priority": "high"
        }
      }
    ]
  }'

# Preferred: process a structured image input
curl -X POST "http://localhost:8000/v2/anpr/process" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "id": "snapshot-01",
        "input_type": "image_url",
        "options": {
          "uri": "https://example.com/sample.jpg",
          "camera_id": "cam_entrance_02"
        },
        "metadata": {
          "source": "manual-review"
        }
      }
    ]
  }'
```

## Configuration (.env)

Edit `.env` file to configure:

```env
# Application
APP_NAME=ANPR API
LOG_LEVEL=INFO

# Triton Server (gRPC)
TRITON_SERVER_URL=host.docker.internal:9001
TRITON_PROTOCOL=grpc

# Dual-model configuration
VEHICLE_MODEL_NAME=vehicle_detection_rt_detr
PLATE_MODEL_NAME=plate_region_detection_rt_detr

# Class-id mapping (JSON format with single quotes)
VEHICLE_CLASS_ID_MAP='{"0":"sedan","1":"suv","2":"hatchback","3":"pickup_truck","4":"van"}'
PLATE_CLASS_ID_MAP='{"0":"number_plate"}'

# Plate candidate filtering
PLATE_CANDIDATE_VEHICLE_CLASSES=sedan,suv,hatchback,pickup_truck,van

# OCR configuration
ENABLE_OCR=True

# Processing settings
BATCH_SIZE=4
CONFIDENCE_THRESHOLD=0.5
FRAMES_PER_SECOND=10.0

# Platform-based model selection (API request parameter)
# platform: "anpr" - uses vehicle detection model with optional OCR
# platform: "object_detection" - uses object detection model, OCR always disabled
# is_ocr_enabled: true/false - controls OCR execution for ANPR platform (ignored for object_detection)

# Paths
INPUT_FOLDER=data/input
OUTPUT_FOLDER=data/output
```

### Key Configuration Options

**FRAMES_PER_SECOND**: Controls video frame sampling rate
- `10.0` = Process 10 frames per second from the video
- Higher values = more frames processed (slower but more detections)
- Lower values = fewer frames processed (faster but may miss detections)

**CONFIDENCE_THRESHOLD**: Minimum detection confidence (0.0-1.0)
- `0.5` = Only keep detections with ≥50% confidence
- Increase to reduce false positives

**Class-id mapping**: JSON dictionary mapping model output IDs to class names
- Keys: Class IDs from model output (as strings)
- Values: Human-readable labels for annotation and OCR filtering

**PLATE_CANDIDATE_VEHICLE_CLASSES**: Comma-separated vehicle class names allowed to receive plate association
- Plate OCR/attachment is only kept for these vehicle classes
- Useful when your detector also finds non-vehicle objects like `animal`, `person`, `traffic light`, or `traffic sign`
- Matching is case-insensitive and based on the names from `VEHICLE_CLASS_ID_MAP`

Example multi-class vehicle detector:
```env
VEHICLE_CLASS_ID_MAP='{"0":"car","1":"truck","2":"bus","3":"motorcycle"}'
```

Example with mixed object classes where only road vehicles should get plates:
```env
VEHICLE_CLASS_ID_MAP='{"0":"animal","1":"autorickshaw","2":"bicycle","3":"bus","4":"car","5":"caravan","6":"motorcycle","7":"person","8":"rider","9":"traffic light","10":"traffic sign","11":"trailer","12":"train","13":"truck","14":"vehicle fallback"}'
PLATE_CANDIDATE_VEHICLE_CLASSES=autorickshaw,bicycle,bus,car,caravan,motorcycle,truck,vehicle fallback
```

## Global Tracking

The pipeline now supports **global identity assignment** across different videos or cameras.

Processing order is:

```text
detect -> local track -> OCR -> global match -> JSON/CSV output
```

### How global matching works

- **Local tracking first**
  - `BYTETracker` assigns `track_id` inside a single video stream.
- **Global matching second**
  - `GlobalTrackingService` tries to map that local track to a reusable `global_id`.
- **Repository-backed state**
  - Global identity state is stored in the configured SQLAlchemy database through SQLAlchemy ORM.

### Matching signals

The global matcher uses multiple signals with different priorities:

- **Class match**
  - Strong signal. Different classes are treated as a mismatch.
- **Color match**
  - Medium signal. Helpful but can vary due to lighting.
- **Dimension / aspect-ratio match**
  - Medium signal. Useful for consistent object shape.
- **License plate text match**
  - Supporting signal used together with class, color, and aspect ratio.
  - Handles common OCR confusion such as `I` vs `1`, `S` vs `5`/`8`, `O` vs `0`.

### Plate text storage and refresh policy

The identity store keeps a single `license_plate_text` and `license_plate_confidence` for each `global_id`.

When newer OCR arrives, the stored plate is updated only if the new observation is better:

- stored text is empty
- new text is longer
- same length but new confidence is higher
- new text matches the expected plate pattern better

This prevents short partial OCR strings from replacing stronger full-plate reads while still allowing later frames to upgrade a weak initial plate observation.

### Global tracking storage

The following state is kept for global identity resolution:

- `global_id`
- `vehicle_class`
- `vehicle_color`
- `license_plate_text`
- `license_plate_confidence`
- `avg_width`
- `avg_height`
- `aspect_ratio`
- `last_camera_id`
- `last_seen_epoch`
- `sighting_count`

## Response Format

Detections follow a standardized JSON schema:

```json
{
  "frame_id": 2150,
  "ts_ms": 86000,
  "track_id": "73C64743",
  "cls": 0,
  "name": "number_plate",
  "conf": 0.95,
  "bbox_xyxy": [445.6, 5.87, 477.6, 29.89],
  "polygon": null,
  "area_px": 768,
  "center": [461.6, 17.88],
  "velocity": ["0.0 km/h"],
  "direction": "stationary",
  "orientation": "stationary",
  "sources": ["triton"],
  "blur_score": 85.3,
  "ocr_text": "ABC1234",
  "ocr_confidence": 0.92,
  "color": "white",
  "camera_id": "cam_01",
  "global_id": "gid_abc123def456",
  "match_score": 0.81,
  "match_reason": "class+color+dimension+plate",
  "plate_bbox_xyxy": null,
  "plate_color": "",
  "plate_conf": 0.0,
  "plate_area_px": 0,
  "direction_vector": [0.0, 0.0],
  "spatial_state": {},
  "behavior_state": {}
}
```

### Global tracking response fields

- `global_id`
  - Stable identity assigned across multiple sources when a match is found.
- `match_score`
  - Final weighted score used for the best global match.
- `match_reason`
  - Human-readable list of signals that contributed to the match.
  - Example values: `class+dimension`, `class+color+dimension+plate`, `new_identity`

### Detection Response Schema (Detailed Field Documentation)

Each detection object in the response contains the following fields:

```json
{
  "frame_id": 882,
  "ts_ms": 29400,
  "track_id": "14",
  "cls": 4,
  "name": "car",
  "conf": 0.9608777761459351,
  "bbox_xyxy": [785.41, 140.01, 1040.43, 394.09],
  "polygon": null,
  "area_px": 64770,
  "center": [912.5, 267.0],
  "velocity": ["17.0 km/h"],
  "direction": "left_to_right",
  "orientation": "left_to_right",
  "sources": ["vehicle_detection_rt_detr", "plate_region_detection_rt_detr"],
  "blur_score": 752.8,
  "ocr_confidence": 0.9457,
  "ocr_text": "DL1T7262",
  "color": "grey",
  "camera_id": "",
  "global_id": "gid_ec7d487b7edd",
  "match_score": 0.9995,
  "match_reason": "class+dimension",
  "plate_bbox_xyxy": [911.59, 270.01, 985.89, 299.95],
  "plate_color": "",
  "plate_conf": 0.8305,
  "plate_area_px": 2146,
  "direction_vector": [0.8, 0.1],
  "spatial_state": {},
  "behavior_state": {}
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame number in the video sequence |
| `ts_ms` | int | Timestamp in milliseconds from video start |
| `track_id` | string | Unique identifier for this tracked object across frames (local tracking) |
| `cls` | int | Class ID from detection model (e.g., 4 = car) |
| `name` | string | Human-readable class name (e.g., "car", "truck") |
| `conf` | float | Detection confidence score (0-1, higher is better) |
| `bbox_xyxy` | array[4] | Bounding box in [x1, y1, x2, y2] format (pixels) |
| `polygon` | null/array | Polygon coordinates if available (e.g., for rotated boxes) |
| `area_px` | int | Bounding box area in pixels |
| `center` | array[2] | Center point of bounding box [x, y] |
| `velocity` | array[string] | Estimated speed display, typically in km/h text form such as `"17.0 km/h"` |
| `direction` | string | Static-CCTV screen-space movement label such as `stationary`, `left_to_right`, `right_to_left`, `towards_top`, or `towards_bottom` |
| `orientation` | string | Static-CCTV motion tendency such as `stationary`, `left_to_right`, `right_to_left`, `approaching`, or `receding` |
| `sources` | array | Models that contributed to this detection |
| `blur_score` | float | Laplacian variance (higher = sharper image) |
| `ocr_confidence` | float | Confidence of OCR text recognition (0-1) |
| `ocr_text` | string | Recognized license plate text |
| `color` | string | **Detected vehicle object color** (not the plate) |
| `camera_id` | string | Camera identifier (empty if not assigned) |
| `global_id` | string | Global unique identifier across all cameras/frames |
| `match_score` | float | Score indicating confidence of multi-model association (0-1) |
| `match_reason` | string | Reason why objects were matched (e.g., "class+dimension") |
| `plate_bbox_xyxy` | array[4] | **License plate** bounding box [x1, y1, x2, y2] (pixels) |
| `plate_color` | string | **Detected license plate color** (not vehicle color) |
| `plate_conf` | float | Confidence of license plate detection (0-1) |
| `plate_area_px` | int | License plate area in pixels |
| `direction_vector` | array[2] | Normalized motion vector derived from tracked center movement between frames |
| `spatial_state` | object | Spatial analytics state for the detection |
| `behavior_state` | object | Behavioral analytics state for the detection |

**Important Notes:**
- All `plate_*` fields refer to the **license plate**, not the vehicle
- `color` field refers to the **vehicle object** color
- `plate_color` field refers to the **license plate** color
- `ocr_text` contains the recognized license plate text extracted via PaddleOCR
- `direction` and `orientation` are heuristics designed for **static CCTV cameras** and are derived from tracked screen-space motion, not true 3D vehicle pose

### Video Processing Response

Video endpoints return comprehensive metadata:

```json
{
  "video_path": "https://example.com/video.mp4",
  "output_path": "/app/data/output/video/annotated.mp4",
  "input_video_fps": 30.0,
  "input_video_duration_sec": 120.5,
  "target_fps": 10.0,
  "sample_interval": 3,
  "total_frames": 3615,
  "processed_frames": 1205,
  "processed_fps": 8.5,
  "total_detections": 45,
  "csv_path": "/app/data/output/video.csv",
  "detections": []
}
```

## Features Explained

### Triton Batching

Triton server configuration supports dynamic batching:
- `max_queue_delay_microseconds: 100000` (100ms)
- Batches requests arriving within the delay window
- Efficient GPU utilization for multiple concurrent requests

### ByteTrack Object Tracking

Tracks vehicles and plates across video frames:
- Assigns unique IDs to detected objects
- Maintains tracking state across frames
- Enables per-track OCR aggregation

### Global Cross-Camera Tracking

Tracks the same object across different videos or camera feeds:
- Uses `GlobalTrackingService` after local tracking is finished
- Stores identity state in the configured SQLAlchemy database
- Returns `global_id`, `match_score`, and `match_reason`
- Refreshes stored identity fields when later frames provide better OCR or visual signals

### FastAPI Dependency Pattern

The API follows a centralized dependency pattern:
- Cached provider functions in `app/utils/dependencies.py` create shared service instances
- `app/utils/dependencies.py` exposes `Annotated` aliases such as `ServiceDep`
- Route handlers import dependency aliases instead of scattering `Depends(...)` across route files

### PaddleOCR Integration

CPU-based OCR to avoid GPU memory conflicts:
- Runs only on plate detections associated with allowed vehicle classes (see `PLATE_CANDIDATE_VEHICLE_CLASSES`)
- Blur score filtering to skip low-quality crops
- Confidence-based triggering

## License Plate Regex Reference

Use these regex patterns as a **starting point** for country-specific plate validation or post-processing. Plate formats can change by state, province, or vehicle type, so adjust them if needed.

### India

```regex
^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
```

Examples:
- `MH12AB1234`
- `DL1CAB1234`

### United States (generic)

```regex
^[A-Z0-9]{5,8}$
```

Examples:
- `ABC1234`
- `7XYZ891`

### United Kingdom

```regex
^[A-Z]{2}[0-9]{2}[A-Z]{3}$
```

Examples:
- `AB12CDE`

### Germany

```regex
^[A-ZÄÖÜ]{1,3}-[A-Z]{1,2} [0-9]{1,4}$
```

Examples:
- `B-MX 1234`
- `M-A 123`

### France

```regex
^[A-Z]{2}-[0-9]{3}-[A-Z]{2}$
```

Examples:
- `AB-123-CD`

### Italy

```regex
^[A-Z]{2}[0-9]{3}[A-Z]{2}$
```

Examples:
- `AB123CD`

### Spain

```regex
^[0-9]{4}[A-Z]{3}$
```

Examples:
- `1234BCD`

### Netherlands

```regex
^(?:[A-Z]{2}-[0-9]{2}-[0-9]{2}|[0-9]{2}-[A-Z]{2}-[0-9]{2}|[0-9]{2}-[0-9]{2}-[A-Z]{2}|[A-Z]{2}-[0-9]{2}-[A-Z]{2}|[A-Z]{2}-[A-Z]{2}-[0-9]{2}|[0-9]{2}-[A-Z]{2}-[A-Z]{2})$
```

Examples:
- `AB-12-34`
- `12-AB-34`

### United Arab Emirates (generic Dubai-style base format)

```regex
^[0-9]{1,5}$
```

Examples:
- `12345`

### Saudi Arabia (simplified alphanumeric OCR-friendly form)

```regex
^[A-Z]{3}[0-9]{1,4}$
```

Examples:
- `ABC1234`

### Singapore

```regex
^[A-Z]{1,3}[0-9]{1,4}[A-Z]$
```

Examples:
- `SBA1234A`
- `EV1234Z`

### Australia (generic)

```regex
^[A-Z0-9]{5,8}$
```

Examples:
- `ABC123`
- `1AB2CD`

### South Africa (generic)

```regex
^[A-Z]{2,3}[0-9]{3,6}$
```

Examples:
- `CA123456`
- `ND12345`

### Notes for regex usage

- OCR output should usually be normalized before regex matching:
  - Convert to uppercase
  - Remove spaces and punctuation if your country format allows it
- Some countries have multiple active formats
- For production validation, combine regex with confidence thresholds and human review for edge cases

### YouTube Video Support

Direct processing of YouTube URLs:
- Uses `yt-dlp` to extract stream URLs
- Handles playlists (processes single video only)
- Works with public YouTube videos

## Development

### Code Structure

- **Services Layer**: Core business logic
  - `triton_client.py` - Triton gRPC client with preprocessing
  - `paddle_ocr_engine.py` - PaddleOCR wrapper (CPU-based)
  - `anpr_service.py` - Main processing pipeline
- **API Layer**: FastAPI routes and endpoints
  - `routes/anpr.py` - ANPR processing endpoints
  - `routes/health.py` - Health checks
- **Utils**: Output formatting and serialization
  - `output_serializers.py` - CSV/JSON writers, track aggregation
- **Schemas**: Pydantic models for validation
- **Core**: Configuration and logging setup

### Logging

Structured logging with `structlog`:
- `video_source_opened` - Video metadata and sampling info
- `detections_filtered` - Detection counts per frame
- `ocr_triggered` - OCR execution details
- `ocr_success` - Recognized text and confidence
- `video_processing_complete` - Final processing stats

## Triton Server Setup

Ensure Triton serves both required models with batching enabled:

```bash
# Example Triton server startup
docker run --gpus all --rm -p 9000:9000 -p 9001:9001 -p 9002:9002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models \
    --http-port=9000 --grpc-port=9001 --metrics-port=9002
```

### Model Configuration

Both models should have dynamic batching enabled in `config.pbtxt`:

```protobuf
max_batch_size: 32

dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100000
}
```

See `../triton_server/model_repository/` for complete model configs.

## Troubleshooting

### GPU Memory Issues

**Problem**: `CUBLAS_STATUS_ALLOC_FAILED` during OCR

**Solution**: PaddleOCR is configured to use CPU to avoid conflicts with Triton models on GPU.

### Triton Batching Not Working

**Problem**: Only processing 1 request at a time

**Solution**: Increase `max_queue_delay_microseconds` in Triton model configs (set to 100000 = 100ms).

### YouTube Videos Failing

**Problem**: `Video unavailable` or playlist errors

**Solution**: Ensure URL points to a single public video. Playlist processing is disabled by default.

### Low Detection Counts

**Problem**: Missing expected detections

**Solution**: Lower `CONFIDENCE_THRESHOLD` (e.g., from 0.5 to 0.3) or increase `FRAMES_PER_SECOND` for videos.

## License

MIT License
