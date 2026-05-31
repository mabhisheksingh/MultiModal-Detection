# RetinaFace Video Face Detection & ArcFace Embedding System

## Overview
This project implements a comprehensive face detection and recognition pipeline combining RetinaFace for face detection and ArcFace for face embedding generation. The system processes video files, detects faces, tracks them across frames, and generates high-quality face embeddings for recognition purposes.

## Features
- **Face Detection**: Real-time face detection using RetinaFace model via NVIDIA Triton Inference Server
- **Face Tracking**: Robust face tracking with Kalman filters and Hungarian algorithm matching
- **ArcFace Embeddings**: High-quality 512-dimensional face embeddings using ArcFace via DeepFace
- **Multiple Input Support**: Video files, URLs, JSON with frame data, and Ceph storage
- **Multiple Output Formats**: Annotated videos, JSON exports, and face sample cropping
- **Advanced Features**:
  - Bounding box visualization with confidence scores and tracking IDs
  - Face count display and tracking statistics
  - Batch processing of multiple inputs via configuration files
  - Automatic output directory creation with proper naming
  - Progress tracking and performance monitoring
  - Embedding metadata and face quality assessment

## Directory Structure
```
retinaface/
├── README.md                     # This documentation file
├── video_client.py               # Main video processing script with ArcFace integration
├── face_tracker.py               # Advanced face tracking with Kalman filters
├── face_embedding_generator.py   # ArcFace embedding generation module
├── input_config.json            # Configuration file for processing parameters
├── client.py                     # Alternative client implementation for images
├── check_onnx_ip_op.py          # ONNX model ip/op generation script
├── requirements.txt              # Python dependencies
├── input/                        # Input video files directory
│   ├── 1338595-hd_1920_1080_30fps.mp4
│   ├── 5538300-uhd_3840_2160_25fps.mp4
│   └── frames_data.json          # Frame data for JSON processing
├── output/                       # Processed outputs directory
│   ├── VideoName/                # Output folder named after video file
│   │   ├── processed_VideoName.mp4       # Annotated video
│   │   ├── face_data_export.json         # Face tracking + ArcFace embeddings
│   │   └── face_samples/                 # Cropped face images (optional)
│   └── ...                       # Additional video outputs
├── triton_face_detection/        # Triton server configuration
│   └── model_repository/         # Model repository directory
│       └── retinaface/           # RetinaFace model directory
│           ├── 1/                # Model version directory
│           │   └── model.onnx    # ONNX model file
│           └── config.pbtxt      # Model configuration file
└── CephTest/                     # Ceph storage integration (optional)
    ├── ceph_client.py
    └── ceph_operations.log
```

## Requirements

### System Requirements
- Python 3.7+
- NVIDIA Triton Inference Server running on localhost:8001

### Python Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- opencv-python
- numpy
- tritonclient[grpc]
- requests
- deepface (for ArcFace embeddings)
- tensorflow (for DeepFace backend)
- scikit-learn (for face tracking algorithms)
- scipy (for tracking computations)

### Model Requirements

#### RetinaFace Detection Model
- **Required Location**: `triton_face_detection/model_repository/retinaface/1/model.onnx`
- **Format**: ONNX
- **Input Resolution**: 640x608
- **Purpose**: Face detection

#### ArcFace Embedding Model
- **Model**: ArcFace (via DeepFace library)
- **Auto-Download**: DeepFace automatically downloads models on first use
- **Expected Location**: `~/.deepface/weights/` (user home directory)
- **Current Location**: [USER TO ADD ACTUAL PATH]
- **Format**: H5/SavedModel
- **Purpose**: Face embedding generation (512 dimensions)

> **Note**: The ArcFace model should be located in the DeepFace weights directory, but may currently be stored elsewhere. Please move the model files to the expected location or update the path configuration.

## Configuration

### Processing Configuration (`input_config.json`)
```json
{
  "processing_config": {
    "confidence_threshold": 0.3,        // Face detection confidence threshold
    "nms_threshold": 0.3,               // Non-maximum suppression threshold
    "similarity_threshold": 0.65,        // Face tracking similarity threshold
    "spatial_threshold": 300.0,         // Spatial distance threshold for tracking
    "max_disappeared": 60,              // Max frames before track deletion
    "confirmation_frames": 5,           // Frames needed to confirm new track
    "save_cropped_faces": false,        // Save individual face crops
    "generate_embeddings": true,        // Enable ArcFace embedding generation
    "embedding_model": "ArcFace",       // Embedding model type
    "embedding_detector_backend": "opencv", // Face detector for embedding
    "min_face_size": 10,                // Minimum face size in pixels
    "enable_face_alignment": true,      // Enable face alignment for embeddings
    "custom_output_fps": 10             // Output video FPS (null for original)
  }
}
```

### Model Configuration
- **RetinaFace**: Face detection via Triton Server (localhost:8001)
- **ArcFace**: 512-dimensional embeddings via DeepFace
- **Input Resolution**: 640x608 for RetinaFace
- **Embedding Size**: 512 dimensions

## Usage

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Triton Server is running
# Start Triton with RetinaFace model loaded
```

### 2. Configure Processing
Edit `input_config.json` to specify:
- Input videos/data sources
- Processing parameters
- Embedding generation settings

### 3. Run triton server using docker image (from \retinaface\triton_face_detection folder)
docker run --rm -it -p8000:8000 -p8001:8001 -p8002:8002 -v %cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-repository=/models

### 4. Run Processing
```bash
python video_client.py
```

### 5. View Results
Results are saved in `output/<VideoName>/`:
- **Annotated Video**: `processed_<VideoName>.mp4`
- **Face Data**: `face_data_export.json` (with ArcFace embeddings)
- **Face Samples**: `face_samples/` directory (optional)

## Input/Output Specifications

### Supported Input Types
1. **Video Files** - Local MP4, AVI, MOV, MKV, WMV, FLV, WebM files
2. **Video URLs** - HTTP/HTTPS URLs for remote video download
3. **JSON with Frame Data** - Frame-by-frame data with timestamps
4. **Video Bytes** - Direct video data as byte arrays
5. **Ceph Storage** - Integration with Ceph object storage

### Output Structure
```
output/
└── VideoName/                           # Named after input video file
    ├── processed_VideoName.mp4          # Annotated video output
    ├── face_data_export.json           # Complete face tracking data
    └── face_samples/                    # Individual face crops (optional)
        ├── face_<track_id>_<frame>.jpg
        └── ...
```

### Output Details

#### Annotated Video (`processed_VideoName.mp4`)
- Green bounding boxes around detected faces
- Confidence scores and tracking IDs displayed
- Face count and tracking statistics
- Preserved original resolution and frame rate

#### Face Data Export (`face_data_export.json`)
```json
{
  "face_detections": [
    {
      "track_id": "ABC123XYZ",
      "frame_number": 45,
      "timestamp": "00:00:01.500",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "embedding": [512-dimensional vector],
      "embedding_size": 512,
      "embedding_model": "ArcFace",
      "detector_backend": "opencv",
      "face_confidence": 0.98,
      "embedding_timestamp": "2024-03-10T15:30:25",
      "face_size": {"width": 64, "height": 80}
    }
  ],
  "summary": {
    "total_frames": 1500,
    "total_detections": 2340,
    "unique_tracks": 15,
    "processing_time": "00:02:34"
  }
}
```

## Processing Details

### Complete Processing Pipeline

#### 1. Face Detection (RetinaFace via Triton)
1. Resize frame to 640x608
2. Convert to float32 and add batch dimension
3. Send to Triton server for inference
4. Decode predictions using anchor boxes
5. Apply confidence threshold (0.3)
6. Perform Non-Maximum Suppression (0.3)

#### 2. Face Embedding Generation (ArcFace)
1. Validate and clamp face bounding boxes
2. Crop faces with minimum 10px size
3. Convert BGR to RGB for DeepFace
4. Generate 512-dimensional embeddings via ArcFace
5. Store embedding metadata and quality metrics

#### 3. Face Tracking & Association
1. Extract face features for tracking
2. Use Hungarian algorithm for optimal matching
3. Update Kalman filters for motion prediction
4. Assign persistent tracking IDs
5. Map embeddings to tracking IDs

#### 4. Output Generation
1. Draw bounding boxes with tracking IDs
2. Export face data with embeddings to JSON
3. Save annotated video with tracking visualization
4. Generate face samples (optional)

### Performance Features
- Progress tracking with frame-by-frame statistics
- Robust tracking across occlusions and pose changes
- Efficient embedding generation with validation
- Automatic output folder organization by video name
- Comprehensive error handling and logging

## Advanced Features

### ArcFace Embeddings
- **High Quality**: 512-dimensional embeddings for superior face recognition
- **Metadata Rich**: Includes confidence scores, face quality metrics, and timestamps
- **Compatible**: Works with DeepFace framework and multiple ArcFace variants
- **Efficient**: Optimized for batch processing and real-time generation

### Face Tracking
- **Persistent IDs**: Maintains consistent tracking IDs across frames
- **Motion Prediction**: Kalman filters for robust motion estimation
- **Occlusion Handling**: Maintains tracks through temporary occlusions
- **Multi-face Scenes**: Handles complex scenes with multiple faces

### Configuration Flexibility
- **JSON-driven**: Easy configuration through `input_config.json`
- **Multiple Inputs**: Support for various input formats and sources
- **Output Control**: Configurable output formats and naming schemes
- **Performance Tuning**: Adjustable thresholds and processing parameters

## Troubleshooting

### Common Issues

#### 1. Model Not Found
```bash
Error: RetinaFace model not found in Triton repository
```
**Solution**: Ensure `model.onnx` is in `triton_face_detection/model_repository/retinaface/1/`

#### 2. ArcFace Model Loading Issues
```bash
Error: DeepFace ArcFace model download failed
```
**Solution**: Check internet connection and DeepFace model cache directory

#### 3. Embedding Generation Failures
```bash
Warning: Failed to generate embedding for detection
```
**Solution**: Check face crop quality, minimum face size settings, and ArcFace model availability

#### 4. Output Folder Naming Issues
```bash
Error: Permission denied creating output directory
```
**Solution**: Check write permissions for output directory and ensure video filename validity

## Model Installation Guide

### Step 1: RetinaFace Model Setup
1. Download the RetinaFace ONNX model
2. Create directory: `triton_face_detection/model_repository/retinaface/1/`
3. Place `model.onnx` in the version directory
4. Configure `config.pbtxt` for Triton Server

### Step 2: ArcFace Model Setup
1. **Expected Location**: `~/.deepface/weights/`
2. **Current Location**: [TO BE UPDATED BY USER]
3. **Auto-Download**: First run will download models automatically
4. **Manual Setup**: Move existing models to expected location if needed

> **Important**: If ArcFace models are stored in a different location, please move them to the expected DeepFace weights directory or update the configuration accordingly.

## Dependencies Installation
```bash
# Install all requirements
pip install -r requirements.txt

# Or install manually:
pip install opencv-python numpy tritonclient[grpc] requests deepface tensorflow scikit-learn scipy
```

## Model Information

### RetinaFace Detection
- **Architecture**: ResNet-based anchor-free detection
- **Multi-scale**: Detects faces from 16x16 to 1024x1024 pixels
- **Real-time**: Optimized for live video processing
- **Accuracy**: State-of-the-art performance on WIDER FACE dataset

### ArcFace Embeddings
- **Deep Learning**: CNN-based feature extraction
- **High Accuracy**: Superior performance on face verification tasks
- **Robust**: Handles pose, lighting, and expression variations
- **Standardized**: 512-dimensional L2-normalized feature vectors

---

## REST API Usage

The project includes a comprehensive REST API that allows you to process videos with face detection and ArcFace embeddings via HTTP endpoints. The API provides the same functionality as the command-line interface but through a web service.

### API Architecture

```
api/
├── app.py              # Flask REST API server
├── job_manager.py      # Background job management
├── models.py           # Data models and response helpers
├── config.py           # API configuration settings
├── run_server.py       # Server launcher with dependency checks
├── test_api_client.py  # Python test client
├── requirements.txt    # API-specific dependencies
└── README_API.md       # Detailed API documentation
```

### Quick Start

#### 1. Install API Dependencies
```bash
# Install API-specific requirements
cd api
pip install -r requirements.txt
```

#### 2. Start the API Server
```bash
# From the api/ directory
python run_server.py
```

**Expected output:**
```
🚀 Face Detection API Server
============================
✅ Checking dependencies...
✅ Checking Triton server...
✅ Testing video_client import...
✅ Ensuring directories...
🌐 Starting API server...
   - Server URL: http://localhost:5000
```

#### 3. Test the API
```bash
# In a new terminal, from the api/ directory
python test_api_client.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Check API and service health |
| `/api/v1/process-config` | POST | Submit `input_config.json` for processing |
| `/api/v1/process-video-file` | POST | Upload and process video files directly |
| `/api/v1/job/<job_id>/status` | GET | Check job processing status |
| `/api/v1/job/<job_id>/results` | GET | Get job results and download links |
| `/api/v1/job/<job_id>/download/<output_id>/json` | GET | Download `face_data_export.json` |
| `/api/v1/job/<job_id>/download/<output_id>/video` | GET | Download processed video |

### Testing with Python Client

The included test client (`test_api_client.py`) provides a complete workflow example:

```bash
cd api
python test_api_client.py
```

**What the test client does:**
1. **Health Check** - Verifies API server, Triton, and ArcFace models
2. **Config Submission** - Automatically finds and submits your `input_config.json`
3. **Progress Monitoring** - Shows real-time progress updates every 15 seconds
4. **Result Download** - Downloads all generated files to `api_downloads/` folder

**Example output:**
```
🚀 Face Detection API Test Client
==================================================
🔍 Checking API health...
✅ API is healthy!
📤 Submitting configuration: ../input_config.json
✅ Job submitted successfully!
   - Job ID: job_abc123def456

⏳ Monitoring job: job_abc123def456
🔄 [015s] PROCESSING:  25% - face_detection
🔄 [045s] PROCESSING:  75% - embedding_generation
✅ Job completed successfully!

📥 Downloading all results...
✅ Downloaded: api_downloads/VideoName/face_data_export.json
✅ Downloaded: api_downloads/VideoName/processed_VideoName.mp4
```

### Testing with cURL

For manual testing or integration with other tools:

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Submit configuration
curl -X POST http://localhost:5000/api/v1/process-config \
  -H "Content-Type: application/json" \
  -d @../input_config.json

# Check job status (replace job_abc123def456 with actual job ID)
curl http://localhost:5000/api/v1/job/job_abc123def456/status

# Get download links
curl http://localhost:5000/api/v1/job/job_abc123def456/results

# Download face data JSON
curl http://localhost:5000/api/v1/job/job_abc123def456/download/VideoName/json \
  -o face_data_export_api.json
```

### API Input/Output

**Input**: Same `input_config.json` format as command-line version
**Output**: Identical results to direct `python video_client.py` execution
- `face_data_export.json` - Face tracking data with ArcFace embeddings
- `processed_video.mp4` - Annotated video with face tracking overlays
- `face_samples/` - Individual face crop images (if enabled)

### Configuration

API settings can be modified in `api/config.py`:

```python
# Server settings
SERVER_HOST = "0.0.0.0"      # Bind to all interfaces
SERVER_PORT = 5000           # API port
DEBUG_MODE = False           # Production mode

# File limits
MAX_FILE_SIZE_MB = 500       # Maximum upload size
JOB_TIMEOUT_MINUTES = 60     # Job processing timeout

# Directories
UPLOAD_FOLDER = "uploads"    # Uploaded files
TEMP_FOLDER = "temp"         # Temporary processing files
OUTPUT_FOLDER = "../output"  # Results directory
```

### Integration Examples

The REST API enables easy integration with:
- **Web Applications** - Frontend interfaces for video upload/processing
- **Mobile Apps** - Submit videos from mobile devices
- **Batch Processing** - Automated video processing workflows
- **Microservices** - Face detection as a containerized service
- **CI/CD Pipelines** - Automated testing with video analysis

### Performance Notes

- **Asynchronous Processing** - Jobs run in background threads
- **Progress Tracking** - Real-time status updates during processing
- **Concurrent Jobs** - Multiple videos can be processed simultaneously
- **Resource Management** - Automatic cleanup of temporary files
- **Error Handling** - Comprehensive error reporting and recovery

For detailed API documentation including request/response schemas, authentication, and advanced usage, see `api/README_API.md`.
