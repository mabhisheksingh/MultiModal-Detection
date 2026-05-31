# Face Detection & ArcFace Embedding REST API

## Overview
This REST API provides HTTP endpoints for the face detection and ArcFace embedding pipeline. You can submit your existing `input_config.json` and get the same `face_data_export.json` output via HTTP requests.

## Quick Start

### 1. Install API Dependencies
```bash
cd api/
pip install -r requirements.txt
```

### 2. Start Triton Server
```bash
# In project root directory
docker run --gpus all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v$(pwd)/triton_face_detection/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
```

### 3. Start API Server
```bash
cd api/
python run_server.py
```

### 4. Test API
```bash
curl http://localhost:5000/api/v1/health
```

## API Endpoints

### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "triton_server": "connected",
    "arcface_model": "loaded"
  }
}
```

### Process Configuration
```http
POST /api/v1/process-config
Content-Type: application/json
```

**Request Body:** Your existing `input_config.json`
```json
{
  "processing_config": {
    "confidence_threshold": 0.3,
    "generate_embeddings": true,
    "embedding_model": "ArcFace"
  },
  "inputs": [
    {
      "id": "video_1",
      "input_type": "video_file",
      "data": {
        "path": "input/your_video.mp4"
      }
    }
  ]
}
```

**Response:**
```json
{
  "status": "accepted",
  "job_id": "job_abc123def456",
  "message": "Processing started with 1 input(s)"
}
```

### Check Job Status
```http
GET /api/v1/job/{job_id}/status
```

**Response:**
```json
{
  "job_id": "job_abc123def456",
  "status": "processing",
  "progress_percentage": 65,
  "current_stage": "face_detection"
}
```

### Get Results
```http
GET /api/v1/job/{job_id}/results
```

**Response:**
```json
{
  "job_id": "job_abc123def456",
  "status": "completed",
  "downloads": {
    "VideoName": {
      "processed_video": {
        "url": "/api/v1/job/job_abc123def456/download/VideoName/video"
      },
      "face_data_json": {
        "url": "/api/v1/job/job_abc123def456/download/VideoName/json"
      }
    }
  }
}
```

### Download Face Data JSON
```http
GET /api/v1/job/{job_id}/download/{output_id}/json
```

**Response:** Your exact `face_data_export.json` file with ArcFace embeddings

## Testing with cURL

### 1. Submit Config
```bash
curl -X POST http://localhost:5000/api/v1/process-config \
  -H "Content-Type: application/json" \
  -d @../input_config.json
```

### 2. Check Status (replace job_id)
```bash
curl http://localhost:5000/api/v1/job/job_abc123def456/status
```

### 3. Download Results
```bash
# Download face_data_export.json
curl http://localhost:5000/api/v1/job/job_abc123def456/download/VideoName/json \
  -o face_data_export.json

# Download processed video
curl http://localhost:5000/api/v1/job/job_abc123def456/download/VideoName/video \
  -o processed_video.mp4
```

## Testing with Python

### Simple Client Example
```python
import requests
import json
import time

# Load your config
with open('../input_config.json') as f:
    config = json.load(f)

# Submit job
response = requests.post('http://localhost:5000/api/v1/process-config', 
                        json=config)
job_data = response.json()
job_id = job_data['job_id']

print(f"Job started: {job_id}")

# Wait for completion
while True:
    status = requests.get(f'http://localhost:5000/api/v1/job/{job_id}/status').json()
    print(f"Status: {status['status']} ({status.get('progress_percentage', 0)}%)")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Job failed: {status.get('error')}")
        exit(1)
    
    time.sleep(10)

# Get results
results = requests.get(f'http://localhost:5000/api/v1/job/{job_id}/results').json()

# Download face data JSON
for output_id, downloads in results['downloads'].items():
    if 'face_data_json' in downloads:
        json_url = downloads['face_data_json']['url']
        face_data = requests.get(f'http://localhost:5000{json_url}')
        
        with open('face_data_export.json', 'wb') as f:
            f.write(face_data.content)
        
        print(f"Downloaded face_data_export.json for {output_id}")

print("✅ Same output as running video_client.py directly!")
```

## File Upload Example

### Upload Video File
```bash
curl -X POST http://localhost:5000/api/v1/process-video-file \
  -F "video_file=@../input/your_video.mp4" \
  -F "config={\"generate_embeddings\": true, \"embedding_model\": \"ArcFace\"}"
```

## Directory Structure
```
api/
├── app.py              # Main Flask application
├── job_manager.py      # Background job management
├── models.py           # Data models and responses
├── config.py           # Configuration settings
├── run_server.py       # Server launcher
├── requirements.txt    # API dependencies
├── README_API.md       # This documentation
├── uploads/            # Uploaded files (created automatically)
├── temp/               # Temporary processing files
└── examples/           # Example requests and clients
```

## Configuration

Edit `config.py` to customize:
- Server port (default: 5000)
- Upload limits (default: 500MB)
- Triton server URL
- Processing timeouts

## Troubleshooting

### Common Issues

1. **"Triton server not ready"**
   - Ensure Triton server is running on localhost:8001
   - Check if RetinaFace model is loaded

2. **"ArcFace model error"**
   - Ensure DeepFace is installed: `pip install deepface`
   - Check internet connection for model download

3. **"Job failed during processing"**
   - Check server logs in console output
   - Verify video file format is supported
   - Check available disk space in output/ directory

4. **"File upload too large"**
   - Check MAX_FILE_SIZE_MB in config.py
   - Compress video or increase limit

### Server Logs
The API server prints detailed logs to console. Look for:
- Job creation messages
- Processing progress
- Error details and stack traces

## Production Notes

For production deployment:
1. Change DEBUG_MODE to False in config.py
2. Use a proper WSGI server (gunicorn, uwsgi)
3. Set up proper logging to files
4. Configure reverse proxy (nginx)
5. Add authentication/API keys if needed
6. Set up SSL/TLS certificates

## Support

The API provides the exact same functionality as running `python video_client.py` directly, but accessible via HTTP requests. Your existing `input_config.json` works unchanged, and you get the same `face_data_export.json` output with ArcFace embeddings.
