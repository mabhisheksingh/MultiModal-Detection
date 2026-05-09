# Drone Object Detection

This module provides a gRPC client for sending drone footage (images/videos) to Triton Inference Server for object detection. It processes the results and saves:
- Processed images with bounding boxes
- JSON detection results
- Original input images

## Folder Structure

```
DRONE_OBJECT_DETECTION/
├── input/              # Original input images (automatically populated)
├── output/             # Processed images and JSON results
├── dataset/            # Sample drone images for testing
├── test_data/          # Additional test data
├── grpc_client.py      # Main gRPC client script
└── README.md           # This file
```

## Prerequisites

1. **Triton Server**: Must be running with an object detection model
   ```bash
   cd ../PLATE_AND_OBJECT_DETECTION/triton_server
   docker-compose up -d
   ```

2. **Python Dependencies**:
   ```bash
   pip install tritonclient[grpc] opencv-python numpy
   ```

## Configuration

Edit the configuration at the top of `grpc_client.py`:

```python
TRITON_SERVER_URL = "localhost:8001"  # gRPC port
MODEL_NAME = "object_detection"         # Your drone detection model
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
CONFIDENCE_THRESHOLD = 0.5
```

## Usage

### Process a Single Image

```bash
python grpc_client.py <image_path>
```

Example:
```bash
python grpc_client.py dataset/drone_image.jpg
```

### Process a Directory of Images

```bash
python grpc_client.py --dir <directory_path>
```

Example:
```bash
python grpc_client.py --dir dataset/
```

### Process a Video

```bash
python grpc_client.py --video <video_path>
```

Example:
```bash
python grpc_client.py --video dataset/drone_footage.mp4
```

## Output

### Image Processing
- **Input**: Original image saved to `input/`
- **Output**: Processed image with bounding boxes saved to `output/processed_<filename>`
- **JSON**: Detection results saved to `output/<filename>_detections.json`

### Video Processing
- **Output**: Processed video with bounding boxes saved to `output/processed_<filename>`
- **JSON**: All detections saved to `output/<filename>_detections.json`

### JSON Output Format

```json
{
  "image_name": "drone_image.jpg",
  "image_shape": [720, 1280, 3],
  "inference_time": 0.123,
  "detections": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "label": "object"
    }
  ],
  "num_detections": 1
}
```

## Model Integration

The client currently uses a placeholder detection parser. Adapt the `_parse_detections` method in `grpc_client.py` to match your specific model output format:

```python
def _parse_detections(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> List[Dict]:
    """Parse model output to detection format."""
    # Adapt this to your model's output format
    detections = []
    # Your parsing logic here
    return detections
```

## Adding Drone Images

Place your drone images in the `dataset/` folder:

```bash
cp /path/to/drone_images/* dataset/
```

Then process them:

```bash
python grpc_client.py --dir dataset/
```

## Troubleshooting

### Connection Failed
- Ensure Triton server is running: `docker-compose ps`
- Check the server URL in `grpc_client.py`

### Model Not Ready
- Verify the model name in `grpc_client.py` matches the Triton model
- Check model status: `curl localhost:8002/v2/models`

### No Detections
- Lower the `CONFIDENCE_THRESHOLD`
- Verify the model is trained for your use case
- Check input image preprocessing matches model requirements

## Development

To add new features or modify the detection pipeline:

1. Edit `grpc_client.py`
2. Add new preprocessing methods to `ImageProcessor` class
3. Extend `DroneDetectionClient` with new processing methods
