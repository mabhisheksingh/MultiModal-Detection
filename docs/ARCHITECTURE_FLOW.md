# ANPR Architecture Flow

## 🎯 Overview

This document explains the complete ANPR (Automatic Number Plate Recognition) system architecture in simple terms for new team members. It covers:

- **API Layer**: FastAPI routes and request handling
- **Service Layer**: Core ANPR processing logic
- **Detection Pipeline**: Vehicle → Plate → OCR flow
- **Data Processing**: Tracking, association, and enrichment
- **Output Generation**: Results formatting and media export

---

## 🏗️ High-Level Architecture

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│  ANPRService    │
│                 │    │   (anpr_v2.py)   │    │ (anpr_service.py)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  Dependency      │    │  Core Services  │
                    │  Injection       │    │                 │
                    │ (dependencies.py)│    │ • TritonClient  │
                    │                  │    │ • PaddleOCREng  │
                    │ • get_anpr_service│    │ • GlobalTrack   │
                    │ • get_ocr_engine │    │ • SpatialCorr   │
                    │ • get_triton_cli │    │ • BehaviorPat   │
                    │ • get_global_svc │    │ • VideoProc     │
                    └──────────────────┘    └─────────────────┘
```

### 🔄 Key Design Principles

1. **Cached Dependencies**: All major services are cached with `@lru_cache` for performance
2. **Sequential Detection**: Reduces server calls by filtering vehicles before plate detection
3. **Modular Services**: Each component has a single responsibility
4. **Structured Input**: V2 API supports rich metadata (camera_id, zones, behavior config)

---

## 🚀 Request Flow: From API to Processing

### 1. API Entry Point

**Endpoint**: `POST /v2/anpr/process`

**Request Format** (V2 Structured Input):
```json
{
  "inputs": [
    {
      "id": "video-01",
      "input_type": "video_url",
      "options": {
        "uri": "https://example.com/video.mp4",
        "camera_id": "cam_gate_01",
        "lat": 23.25,
        "lon": 77.41,
        "pixels_per_meter": 22.5,
        "zones": [...],
        "behavior_config": {...}
      },
      "metadata": {...}
    }
  ],
  "processing_config": {...}
}
```

### 2. Request Processing Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. FastAPI Route Handler (anpr_v2.py)                      │
│    • Validate request payload                               │
│    • Extract request_id from headers                       │
│    • Resolve ANPRService dependency                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Input Processing (_process_common_anpr_input)           │
│    • Validate each input_type (video_url, image_url, etc.) │
│    • Normalize zones and behavior config                   │
│    • Route to appropriate processor                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Service Layer Entry                                     │
│    • process_video_source() for videos                     │
│    • process_image_url() for image URLs                   │
│    • process_image_source() for image files               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎥 Video Processing Flow

### Video Processing Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. LiveVideoSourceProcessor                                 │
│    (video_source_processor.py)                              │
│    • Open video/stream                                       │
│    • Sample frames at target_fps                             │
│    • Process frames in batches                              │
│    • Write annotated video output                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Per-Frame Processing Loop                                │
│    For each frame batch:                                    │
│    • Call infer_frame_payloads()                            │
│    • Call process_frame_after_inference()                   │
│    • Accumulate results                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Final Output Generation                                  │
│    • Aggregate all detections                               │
│    • Generate CSV summary                                   │
│    • Return processing summary                              │
└─────────────────────────────────────────────────────────────┘
```

### Image Processing Flow

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Single Image Processing                                  │
│    • Load image from URL or file                            │
│    • Call infer_frame_payloads()                            │
│    • Call process_frame_after_inference()                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Immediate Result Return                                  │
│    • Annotated image                                        │
│    • Detection list                                         │
│    • Output path                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detection Pipeline: Core Processing Logic

### Sequential Detection Flow (Optimized)

The system uses a **sequential detection approach** to reduce Triton server calls:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Vehicle Detection (Triton Call #1)                       │
│    • Run vehicle detection on full image                    │
│    • Filter by PLATE_CANDIDATE_VEHICLE_CLASSES              │
│    • Skip plate detection if no valid vehicles             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Only if valid vehicles found)
┌─────────────────────────────────────────────────────────────┐
│ 2. Plate Detection (Multiple Triton Calls)                 │
│    • Crop each valid vehicle region                         │
│    • Run plate detection on crops only                      │
│    • Transform coordinates to full image space              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Detection Collection & Merging                           │
│    • _collect_model_detections()                            │
│    • _merge_detections() with NMS                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Local Tracking                                           │
│    • BYTETracker for local track IDs                        │
│    • Track association across frames                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

**Vehicle Filtering**:
- Only vehicles in `PLATE_CANDIDATE_VEHICLE_CLASSES` trigger plate detection
- Config: `car, bus, truck, motorcycle, autorickshaw, bicycle, vehicle fallback`
- Reduces unnecessary plate detection calls

**Plate Detection Optimization**:
- Crops vehicle regions before plate detection
- Parallel processing of multiple vehicle crops
- Simple coordinate transformation (offset addition)

**Non-Maximum Suppression (NMS)**:
- Applied after merging all detections
- Removes overlapping boxes
- Configurable IOU threshold

### Core Frame Processing Method

The main processing happens in `process_frame_after_inference()`:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Detection Collection                                      │
│    _collect_model_detections()                             │
│    • Handle raw detections and pre-decoded boxes           │
│    • Apply confidence filtering                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Detection Merging                                        │
│    _merge_detections()                                     │
│    • Concatenate all detections                            │
│    • Apply NMS to remove overlaps                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Local Tracking                                           │
│    _track_detections()                                     │
│    • BYTETracker for consistent track IDs                 │
│    • Handle tracker initialization/reset                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Record Building                                         │
│    _build_detection_records()                              │
│    • Crop detection regions                                │
│    • Compute blur scores                                   │
│    • Extract vehicle/plate colors                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. OCR Processing                                           │
│    _run_ocr_on_records()                                  │
│    • Run OCR only on plate classes                         │
│    • Apply confidence thresholds                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Plate-Vehicle Association                               │
│    _associate_plates_to_vehicles()                        │
│    • Match plates to enclosing vehicles                    │
│    • Merge plate metadata into vehicle results             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Global Tracking                                          │
│    GlobalTrackingService.resolve_detections()              │
│    • Cross-camera identity matching                        │
│    • Feature-based similarity scoring                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Analytics Enrichment                                     │
│    • SpatialCorrelationService (zone analysis)             │
│    • BehavioralPatternService (behavior analysis)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Output Generation                                        │
│    _annotate_detection_records()                          │
│    • Draw bounding boxes and labels                        │
│    • Format response items                                 │
└─────────────────────────────────────────────────────────────┘
```

## Detection Record Contents

Each detection record is built from tracked boxes and carries both runtime metadata and response-ready fields.

Typical fields include:

- `track_id`
- `cls`, `name`
- `conf`
- `bbox_xyxy`
- `area_px`
- `blur_score`
- `color`
- `plate_color`
- `ocr_text`
- `ocr_confidence`
- `sources`

---

## 🔤 OCR Processing Pipeline

### OCR Flow

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Plate Crop Preparation                                    │
│    • Extract plate bounding box from frame                  │
│    • Validate crop size and quality                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PaddleOCR Engine                                         │
│    PaddleOCREngine.recognize()                             │
│    • PP-ChatOCRv4Doc backend                                │
│    • Deskewing and upscaling                               │
│    • Multi-angle fallback                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Text Normalization                                       │
│    OCRUtils.normalize_text()                               │
│    • Remove non-alphanumeric characters                    │
│    • Handle OCR artifacts (confusion groups)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Result Formatting                                        │
│    • Combine text segments                                  │
│    • Compute average confidence                            │
│    • Apply confidence threshold filtering                   │
└─────────────────────────────────────────────────────────────┘
```

### OCR Configuration

- **Engine**: PaddleOCR with PP-ChatOCRv4Doc backend
- **Threshold**: `ocr_confidence_threshold` (default: 0.5)
- **Enhancements**: Deskewing, upscaling, multi-angle fallback
- **Normalization**: Removes artifacts, handles character confusions

---

## 🔗 Plate-Vehicle Association

### Association Algorithm

Plate detections are matched to parent vehicles using spatial containment:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Split Detection Records                                   │
│    • Separate vehicle detections                           │
│    • Separate plate detections                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. For Each Plate: Find Best Vehicle                        │
│    _bbox_containment_score()                               │
│    • Calculate fraction of plate inside vehicle bbox       │
│    • Threshold: ≥ 0.5 (50% containment)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Merge Plate Metadata into Vehicle                        │
│    If containment_score ≥ threshold:                       │
│    • Copy ocr_text, ocr_confidence                         │
│    • Copy plate_bbox_xyxy, plate_conf, plate_area_px       │
│    • Copy plate_color, merge source labels                 │
│    • Remove plate record from results                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Final Results                                            │
│    • Vehicles with merged plate information                 │
│    • Orphan plates (no enclosing vehicle)                   │
└─────────────────────────────────────────────────────────────┘
```

### Vehicle Result After Merge

A merged vehicle detection contains:
- Vehicle `color` and class information
- Merged plate `ocr_text` and `ocr_confidence`
- `plate_bbox_xyxy`, `plate_color`, `plate_conf`, `plate_area_px`
- Combined `sources` list (vehicle + plate models)

---

## 🌍 Global Identity Tracking

### Cross-Camera Identity Matching

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Input: Resolved Detections                               │
│    • After plate-vehicle association                        │
│    • With local track_id and OCR text                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Extraction                                       │
│    Build TrackFeatures for each detection:                 │
│    • local_track_id, request_id, camera_id                 │
│    • vehicle_class, vehicle_color                           │
│    • license_plate_text, license_plate_confidence           │
│    • bbox dimensions (width, height, aspect ratio)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Similarity Scoring                                       │
│    Multiple signals with adaptive weights:                  │
│    • Class match (30%)                                      │
│    • Color similarity (20%)                                 │
│    • Dimension similarity (20%)                             │
│    • Plate text similarity (30%)                            │
│    • Adaptive: Redistribute weights when signals missing    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Identity Resolution                                       │
│    • Check existing track associations                      │
│    • Refresh identity if better observation arrives         │
│    • Score recent identities (within time window)           │
│    • Assign existing global_id OR create new one           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Database Persistence                                     │
│    • GlobalTrackRepository CRUD operations                  │
│    • GlobalIdentity table (persistent identity data)        │
│    • TrackAssociation table (linking local to global)      │
└─────────────────────────────────────────────────────────────┘
```

### Plate Text Matching

- **Fuzzy Levenshtein distance** with OCR confusion awareness
- **Character substitution costs**: `M↔9`, `O↔0`, `C↔0`, `P↔9` (cost: 0.3)
- **Artifact normalization**: `(`→`C`, `$`→`S`, `|`→`I`, etc.
- **Threshold**: 0.70 similarity score

---

## 🗄️ Database & Persistence Layer

### Database Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Database Initialization                                   │
│    FastAPI lifespan startup → init_db()                    │
│    • SQLAlchemy ORM setup                                   │
│    • In-memory SQLite database                              │
│    • Schema repair for missing columns                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Repository Layer                                         │
│    GlobalTrackRepository                                   │
│    • get_recent_identities()                               │
│    • get_identity_by_global_id()                           │
│    • upsert_identity()                                      │
│    • find_association(), create_association()               │
│    • update_association()                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Database Tables                                           │
│    • GlobalIdentity: best observations per global_id        │
│    • TrackAssociation: local_track_id ↔ global_id links    │
│    • CameraConfig: per-camera settings                      │
│    • VisitHistory, BehaviorEvents (analytics)              │
└─────────────────────────────────────────────────────────────┘
```

### Identity Storage Strategy

- **Best Observation Storage**: Keep highest quality plate text per identity
- **Upgrade Policy**: Replace stored text only when new observation is better
- **Association Tracking**: Store match scores and reasons for each association
- **Time Window**: Recent identities for matching (configurable duration)

---

## 📊 Analytics & Enrichment Services

### Spatial Correlation Service

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Zone Analysis                                            │
│    • Point-in-polygon testing                              │
│    • Box-to-zone matching                                   │
│    • Zone type classification (entry, restricted, etc.)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Spatial State Enrichment                                  │
│    For each detection:                                      │
│    • active_zone_id, active_zone_type                       │
│    • is_inside_zone, spatial_label                         │
│    • spatial_score (confidence of zone assignment)          │
└─────────────────────────────────────────────────────────────┘
```

### Behavioral Pattern Service

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Historical Analysis                                      │
│    • Visit history from database                            │
│    • Repeated visit detection                               │
│    • Dwell time calculation                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Behavior State Enrichment                                │
│    For each detection:                                      │
│    • is_repeat_visit, is_lingering                         │
│    • is_sensitive_zone_presence                             │
│    • visit_count, dwell_time_ms                            │
│    • behavior_label, behavior_score                        │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

- **Zones**: Defined per input, with zone_id and coordinates
- **Behavior Config**: Thresholds for repeat visits, lingering, sensitive zones
- **Camera Metadata**: lat/lon, pixels_per_meter for spatial calculations

---

## 📤 Output Generation & Response Format

### Output Processing Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Result Serialization                                     │
│    output_serializers.build_detection_response_item()      │
│    • Convert internal records to API format                 │
│    • Include all enriched fields                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Frame Annotation                                         │
│    _annotate_detection_records()                           │
│    • Draw bounding boxes                                    │
│    • Add labels (class, track_id, global_id, OCR)         │
│    • Use ultralytics Annotator for consistent styling      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Multi-Format Output                                      │
│    • API JSON response                                     │
│    • Annotated video/image files                           │
│    • CSV detection summary                                 │
│    • Track summary CSV                                      │
└─────────────────────────────────────────────────────────────┘
```

### Detection Response Item Structure

```json
{
  "frame_id": 12,
  "ts_ms": 2400,
  "track_id": "17",
  "cls": 4,
  "name": "car",
  "conf": 0.96,
  "bbox_xyxy": [100, 200, 300, 400],
  "area_px": 40000,
  "center": [200, 300],
  "velocity": ["17.0 km/h"],
  "direction": "left_to_right",
  "orientation": "left_to_right",
  "direction_vector": [0.8, 0.1],
  "color": "blue",
  "camera_id": "cam_001",
  "global_id": "gid_101",
  "match_score": 0.85,
  "match_reason": "class_color_plate",
  "ocr_text": "MP09AB1234",
  "ocr_confidence": 0.92,
  "plate_bbox_xyxy": [150, 250, 250, 280],
  "plate_color": "white",
  "plate_conf": 0.88,
  "plate_area_px": 2400,
  "spatial_state": {
    "active_zone_id": "entry_1",
    "active_zone_type": "entry",
    "is_inside_zone": true,
    "spatial_label": "inside_zone",
    "spatial_score": 0.9
  },
  "behavior_state": {
    "is_repeat_visit": false,
    "is_lingering": false,
    "is_sensitive_zone_presence": true,
    "visit_count": 3,
    "dwell_time_ms": 18000,
    "behavior_label": "normal",
    "behavior_score": 0.3
  },
  "sources": ["yolov8n_vehicle", "yolov8n_plate"]
}
```

---

## 🏗️ Component Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   API Routes    │  │   Dependencies  │  │   Config     │ │
│  │  (anpr_v2.py)   │  │ (dependencies)  │  │   Settings   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   ANPRService   │  │VideoSourceProc  │  │GlobalTrack   │ │
│  │ (Core Logic)    │  │ (Video Handler) │  │ (Identity)   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ SpatialCorr     │  │BehaviorPattern  │  │TritonClient  │ │
│  │ (Zones)         │  │ (Analytics)     │  │ (Inference)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐                                            │
│  │ PaddleOCREngine │                                            │
│  │ (OCR Processing)│                                            │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Repository    │  │   Database      │  │   Utils      │ │
│  │ (ORM Operations)│  │ (SQLite/SQLA)   │  │ (Helpers)    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Service Responsibilities

| Service | Primary Responsibility | Key Methods |
|---------|----------------------|-------------|
| **ANPRService** | Core detection pipeline | `process_frame_after_inference()`, `infer_frame_payloads()` |
| **VideoSourceProcessor** | Video handling & batching | `process()`, frame batching logic |
| **GlobalTrackingService** | Cross-camera identity | `resolve_detections()`, feature matching |
| **SpatialCorrelationService** | Zone analysis | `enrich_detections_with_spatial_state()` |
| **BehavioralPatternService** | Behavior analytics | `enrich_detections_with_behavior_state()` |
| **TritonClient** | Model inference | Vehicle/plate detection calls |
| **PaddleOCREngine** | OCR processing | `recognize()`, text normalization |

### Data Flow Summary

1. **Request → API → Service**: Structured input flows through cached dependencies
2. **Inference**: Sequential vehicle → plate detection reduces server calls
3. **Processing**: Tracking → OCR → Association → Global identity → Analytics
4. **Output**: Multi-format results (JSON, annotated media, CSV)

---

## 🎯 Quick Start for New Developers

### Understanding the Flow

1. **Start with the API**: Look at `anpr_v2.py` to understand request format
2. **Follow the service**: `ANPRService.process_video_source()` for videos
3. **Core processing**: `process_frame_after_inference()` is the main pipeline
4. **Detection details**: `infer_frame_payloads()` shows the sequential optimization

### Key Configuration Files

- **`.env`**: Model paths, OCR settings, class mappings
- **`settings.py`**: Application configuration and defaults
- **`constants.py`**: Shared constants and error codes

### Important Design Patterns

- **Cached Dependencies**: Services are singletons via `@lru_cache`
- **Sequential Detection**: Filter vehicles before plate detection
- **Modular Services**: Each service has a single responsibility
- **Structured Input**: V2 API supports rich metadata

### Common Modifications

- **Add new detection classes**: Update class mappings and filters
- **Modify OCR behavior**: Change PaddleOCREngine settings
- **Adjust tracking**: Modify BYTETracker parameters
- **Add analytics**: Extend spatial/behavioral services

---

## 📈 Performance Considerations

### Optimizations Built-In

- **Sequential Detection**: Reduces Triton calls by filtering vehicles first
- **Cached Services**: Avoids repeated initialization overhead
- **Batch Processing**: Video frames processed in efficient batches
- **NMS**: Removes overlapping detections early

### Scaling Tips

- **Triton Server**: Scale inference horizontally
- **Database**: Consider PostgreSQL for production vs SQLite in-memory
- **Video Processing**: Adjust batch sizes based on hardware
- **OCR**: PaddleOCR is CPU-intensive; consider GPU acceleration

---

## 🚧 Current Limitations & Known Issues

### 📋 Supported Inputs & Current Capabilities

**Currently Supported Input Types**:
- ✅ **Video**: MP4, AVI, MOV, WebM (URLs, local files, streams)
- ✅ **Images**: JPEG, PNG, BMP, TIFF (URLs, local files, base64)
- ✅ **Structured Input**: V2 API with rich metadata (camera_id, zones, behavior config)

**Detection Capabilities**:
- ✅ **Vehicle Detection**: 8 classes (car, bus, truck, motorcycle, autorickshaw, bicycle, caravan, vehicle fallback)
- ✅ **Plate Detection**: Number plates on vehicle crops
- ✅ **OCR Processing**: Text extraction with confidence scoring
- ✅ **Local Tracking**: BYTETracker for consistent track IDs
- ✅ **Global Tracking**: Cross-camera identity matching
- ✅ **Analytics**: Zone analysis and behavioral patterns

**Output Formats**:
- ✅ **API JSON**: Structured response with enriched metadata
- ✅ **Annotated Media**: Videos/images with bounding boxes and labels
- ✅ **CSV Reports**: Detection summaries and track analytics

### 🔤 OCR Limitations & Challenges

**Current OCR Performance**:

⚠️ **Important Note**: OCR does not work 100% of the time. When number plates are clearly visible and well-positioned, it works approximately 70% of the time.

**Common Character Confusions**:
The OCR system frequently confuses certain characters, especially with similar visual patterns:

```json
{
    "(": "C",
    "{": "C",
    "$": "S",
    "|": "I",
    "!": "1",
    "@": "A",
    "#": "H",
    "\"": "",
    "H": "M",
    "M": "H"
}
```

**Specific Performance Issues**:

1. **Character Confusion**:
   - **Issue**: Similar-looking characters are frequently misidentified
   - **Examples**: `H` ↔ `M`, `(` → `C`, `$` → `S`, `|` → `I`, `!` → `1`
   - **Impact**: Plate text like "MH12AB1234" might become "H12AB1234" or "M12AB1234"

2. **Symbol Interference**:
   - **Issue**: Non-alphanumeric characters are incorrectly interpreted as letters
   - **Examples**: `@` → `A`, `#` → `H`, quotes (`"`) are removed
   - **Impact:"
     - Plate with reflections or artifacts gets false characters
     - Clean plates may have valid characters incorrectly replaced

3. **Partial Recognition**:
   - **Issue**: Only portions of the plate text are correctly identified
   - **Impact**: "MP09CP1234" might become "M9CP123" or "MP091234"
   - **Current Mitigation**: Post-processing attempts to normalize, but accuracy varies

4. **Quality Dependency**:
   - **Issue**: Performance heavily depends on image quality and conditions
   - **Good Conditions** (clear, frontal, well-lit): ~70% success rate
   - **Poor Conditions** (blurry, angled, low-light): ~20-30% success rate

**Real-World Performance**:
- **Ideal Conditions**: 70% accuracy when plates are clearly visible
- **Average Conditions**: 40-50% accuracy with typical surveillance footage
- **Challenging Conditions**: 20-30% accuracy with poor quality or angled plates

**Current Mitigation Strategies**:
- Character confusion normalization rules
- Artifact removal and symbol filtering
- Multi-angle processing for tilted plates
- Confidence threshold filtering

### 📚 Additional Resources

- **API Documentation**: See `app/api/anpr_v2.py` for endpoint details
- **Configuration Guide**: See `app/core/config.py` for settings
- **Troubleshooting**: Common issues and solutions in logs
- **Performance Tuning**: Adjust thresholds and batch sizes based on hardware
