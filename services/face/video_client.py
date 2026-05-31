import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import requests
import os
import json
from urllib.parse import urlparse
import urllib3
from face_tracker import RobustFaceTracker
from CephTest.ceph_client import CephClient
from CephTest.utils import ConfigLoader, LoggerSetup
import tempfile
import uuid
try:
    from face_embedding_generator import create_embedding_generator
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️ Face embedding generator not available. Embedding generation will be disabled.")

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TRITON_URL = "localhost:8001"  # Local Triton on same machine
MODEL_NAME = "retinaface"

INPUT_H = 608
INPUT_W = 640

# Configuration file path
CONFIG_FILE = "input_config.json"

# Global embedding generator (will be initialized when needed)
embedding_generator = None

def resolve_frame_timestamp_ms(frame_metadata, frame_index, fps):
    """Resolve frame timestamp in milliseconds from metadata or fps fallback."""
    if isinstance(frame_metadata, dict):
        for key in ("timestamp_ms", "ts_ms", "frame_timestamp_ms"):
            value = frame_metadata.get(key)
            if value is not None:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    pass
        for key in ("timestamp", "ts", "time_sec"):
            value = frame_metadata.get(key)
            if value is not None:
                try:
                    return int(float(value) * 1000)
                except (TypeError, ValueError):
                    pass
    if fps and fps > 0:
        return int((frame_index / fps) * 1000)
    return None

def initialize_embedding_generator(processing_config):
    """Initialize the ArcFace embedding generator."""
    global embedding_generator
    
    if not EMBEDDING_AVAILABLE or not processing_config.get('generate_embeddings', False):
        embedding_generator = None
        return False
    
    try:
        embedding_generator = create_embedding_generator(processing_config)
        print(f"✅ ArcFace embedding generator initialized")
        print(f"   Model: {embedding_generator.model_name}")
        print(f"   Embedding size: {embedding_generator.model_specs.get(embedding_generator.model_name, {}).get('embedding_size', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize embedding generator: {e}")
        embedding_generator = None
        return False

def generate_face_embedding(face_image, processing_config):
    """Generate face embedding using ArcFace model."""
    global embedding_generator
    
    # Initialize embedding generator if needed
    if embedding_generator is None and processing_config.get('generate_embeddings', False):
        print(f"🔧 Debug: Attempting to initialize embedding generator (was None)")
        if not initialize_embedding_generator(processing_config):
            print(f"❌ Debug: Failed to initialize embedding generator")
            return None
        print(f"✅ Debug: Embedding generator initialized successfully")
    
    # Skip if embedding generation is disabled or generator unavailable
    if embedding_generator is None:
        print(f"⚠️ Debug: Skipping embedding - generator is None")
        return None
    
    if not processing_config.get('generate_embeddings', False):
        print(f"⚠️ Debug: Skipping embedding - generate_embeddings is False")
        return None
    
    # Validate face image
    if face_image is None or face_image.size == 0:
        return None
    
    try:
        # Generate embedding using our ArcFace module
        result = embedding_generator.generate_embedding(face_image)
        
        if result and 'embedding' in result:
            return result  # Return full result with metadata
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ Error generating ArcFace embedding: {e}")
        return None

# Function to load input configuration from JSON file
def load_input_configuration(config_file=CONFIG_FILE):
    """Load input configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract enabled inputs only
        enabled_inputs = []
        for input_item in config.get('inputs', []):
            if input_item.get('metadata', {}).get('enabled', True):
                enabled_inputs.append(input_item)
        
        # Processing configuration with defaults
        processing_config = config.get('processing_config', {})
        
        # Set defaults for all processing parameters
        defaults = {
            'confidence_threshold': 0.3,
            'nms_threshold': 0.3,
            'similarity_threshold': 0.65,
            'spatial_threshold': 300.0,
            'max_disappeared': 60,
            'confirmation_frames': 5,
            'save_cropped_faces': False,
            'generate_embeddings': True,
            'embedding_model': 'ArcFace',
            'embedding_detector_backend': 'opencv',
            'min_face_size': 10,
            'enable_face_alignment': True,
            'custom_output_fps': None
        }
        
        # Apply defaults for missing values
        for key, default_value in defaults.items():
            if key not in processing_config:
                processing_config[key] = default_value
        
        print(f"✅ Loaded configuration from {config_file}")
        print(f"📊 Found {len(enabled_inputs)} enabled inputs")
        print(f"⚙️ Processing config loaded:")
        for key, value in processing_config.items():
            print(f"   {key}: {value}")
        
        return enabled_inputs, processing_config
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_file}")
        print("Using fallback configuration...")
        # Fallback configuration with defaults
        fallback_processing_config = {
            'confidence_threshold': 0.3,
            'nms_threshold': 0.3,
            'similarity_threshold': 0.65,
            'spatial_threshold': 300.0,
            'max_disappeared': 60,
            'confirmation_frames': 5,
            'save_cropped_faces': False
        }
        return [
            {
                "id": "fallback_video",
                "input_type": "video_file",
                "data": {
                    "path": r"input\Dashcam_Helps_Police_Track_Down_Offenders_Heavy_Penalty.mp4"
                }
            }
        ], fallback_processing_config
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        # Return empty inputs but default processing config
        fallback_processing_config = {
            'confidence_threshold': 0.3,
            'nms_threshold': 0.3,
            'similarity_threshold': 0.65,
            'spatial_threshold': 300.0,
            'max_disappeared': 60,
            'confirmation_frames': 5,
            'save_cropped_faces': False
        }
        return [], fallback_processing_config

client = grpcclient.InferenceServerClient(TRITON_URL)

# Face tracker will be initialized per video
face_tracker = None

# Initialize Ceph client for S3A URL handling
ceph_client = None
def initialize_ceph_client():
    """Initialize Ceph client for frame fetching."""
    global ceph_client
    try:
        config_loader = ConfigLoader('CephTest/config.yaml')
        config = config_loader.load()
        logger = LoggerSetup.setup_logger(config)
        ceph_client = CephClient(config, logger)
        print("✅ Ceph client initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Could not initialize Ceph client: {e}")
        print("Will fallback to HTTP requests for frame fetching")
        return False

def parse_s3a_url(s3a_url):
    """Parse S3A URL to extract bucket and object key."""
    if not s3a_url.startswith('s3a://'):
        raise ValueError(f"Invalid S3A URL format: {s3a_url}")
    
    # Remove 's3a://' prefix
    path = s3a_url[6:]
    
    # Split into bucket and object key
    parts = path.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3A URL format: {s3a_url}")
    
    bucket, object_key = parts
    return bucket, object_key

# Function to detect input type (video, JSON, or bytes data)
def get_input_type(input_data):
    # Check if input is bytes data (dict with type and data)
    if isinstance(input_data, dict) and 'type' in input_data and 'data' in input_data:
        if input_data['type'] == 'video_bytes':
            return 'video_bytes'
        elif input_data['type'] == 'frame_bytes_json':
            return 'frame_bytes_json'
        else:
            raise ValueError(f"Unsupported bytes input type: {input_data['type']}")
    
    # Treat as path/URL for backward compatibility
    path_or_url = input_data if isinstance(input_data, str) else str(input_data)
    
    if path_or_url.startswith(('http://', 'https://')):
        # For URLs, check extension if available
        ext = os.path.splitext(path_or_url.split('?')[0])[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        if ext in video_extensions:
            return 'video'
        else:
            raise ValueError(f"Unsupported format for URL: {path_or_url}")
    else:
        # For local files, check if it exists and determine type
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"File not found: {path_or_url}")
        
        ext = os.path.splitext(path_or_url)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        if ext in video_extensions:
            return 'video'
        elif ext == '.json':
            # Validate JSON - check if it's Ceph URLs or bytes data
            try:
                with open(path_or_url, 'r') as f:
                    data = json.load(f)
                
                # First check if it's bytes frame data
                if validate_json_bytes_frame_data(data):
                    return 'json_frame_bytes'
                # Then check if it's traditional Ceph URL frame data
                elif validate_json_frame_data(data):
                    return 'json'
                else:
                    raise ValueError(f"Invalid JSON format for frame processing: {path_or_url}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON file: {path_or_url}")
        else:
            # Try to determine by attempting to open as video
            cap = cv2.VideoCapture(path_or_url)
            if cap.isOpened():
                cap.release()
                return 'video'
            else:
                raise ValueError(f"Unsupported format: {path_or_url}")

# Function to validate JSON frame data structure
def validate_json_frame_data(data):
    """Validate that JSON contains valid frame data with Ceph URLs."""
    if not isinstance(data, list):
        return False
    
    if len(data) == 0:
        return False
        
    # Check if each frame has required fields
    for frame in data:
        if not isinstance(frame, dict):
            return False
        if 'frame_id' not in frame or 'frame_path' not in frame:
            return False
        if not isinstance(frame['frame_path'], str) or not frame['frame_path'].startswith(('http://', 'https://', 's3a://')):
            return False
    
    return True

# Function to validate JSON with bytes frame data structure
def validate_json_bytes_frame_data(data):
    """Validate that JSON contains valid frame data with base64 bytes."""
    # Check if it's the new structured format with input_type
    if isinstance(data, dict):
        if 'input_type' in data and data['input_type'] == 'frame_bytes':
            if 'frames' not in data or not isinstance(data['frames'], list):
                return False
            frames = data['frames']
        else:
            return False
    elif isinstance(data, list):
        # Direct list format (backward compatibility)
        frames = data
    else:
        return False
    
    if len(frames) == 0:
        return False
        
    # Check if each frame has required fields for bytes data
    for frame in frames:
        if not isinstance(frame, dict):
            return False
        if 'frame_id' not in frame or 'frame_data' not in frame:
            return False
        if not isinstance(frame['frame_data'], str):
            return False
    
    return True

# Function to fetch frame from URL using CephClient
def fetch_frame_from_url(frame_url):
    """Fetch frame image from URL (supports HTTP/HTTPS/S3A)."""
    
    try:
        if frame_url.startswith('s3a://') and ceph_client is not None:
            # Use CephClient for S3A URLs
            print(f"📡 Using CephClient for S3A URL")
            bucket, object_key = parse_s3a_url(frame_url)
            print(f"📦 Bucket: {bucket}, Object: {object_key}")
            
            # Create temporary file for download
            temp_dir = tempfile.gettempdir()
            temp_filename = f"frame_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            print(f"💾 Temp file path: {temp_path}")
            
            try:
                # Download frame using CephClient
                print(f"⬇️ Downloading from Ceph...")
                downloaded_path = ceph_client.download_file(bucket, object_key, temp_path)
                print(f"✅ Downloaded to: {downloaded_path}")
                
                # Load image using OpenCV
                frame = cv2.imread(downloaded_path)
                print(f"🖼️ Loaded image, shape: {frame.shape if frame is not None else 'None'}")
                
                # Clean up temporary file
                if os.path.exists(downloaded_path):
                    os.remove(downloaded_path)
                    print(f"🗑️ Cleaned up temp file")
                
                if frame is None:
                    raise ValueError(f"Could not decode image from Ceph: {frame_url}")
                    
                return frame
                
            except Exception as ceph_error:
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise ceph_error
                
        else:
            # Fallback to HTTP requests for regular URLs or when CephClient is unavailable
            if frame_url.startswith('s3a://'):
                # Convert s3a:// URLs to https:// for direct access (fallback)
                url_parts = frame_url[6:].split('/', 1)  # Remove 's3a://' prefix
                if len(url_parts) == 2:
                    bucket, path = url_parts
                    frame_url = f"https://{bucket}.s3.amazonaws.com/{path}"
                else:
                    raise ValueError(f"Invalid S3A URL format: {frame_url}")
            
            response = requests.get(frame_url, timeout=30, verify=False)
            response.raise_for_status()
            
            # Convert response content to numpy array
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError(f"Could not decode image from URL: {frame_url}")
                
            return frame
            
    except Exception as e:
        print(f"Error fetching frame from {frame_url}: {e}")
        return None

# Function to process frames from JSON input
def process_json_frames(json_path, output_name, json_base_name, processing_config):
    """Process frames from JSON file containing Ceph URLs."""
    # Create output directory
    json_output_dir = os.path.join("output", json_base_name)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Initialize face tracker for this JSON processing
    global face_tracker
    face_tracker = RobustFaceTracker(
        db_path=None,  # Database disabled
        similarity_threshold=processing_config.get('similarity_threshold', 0.65),
        spatial_threshold=processing_config.get('spatial_threshold', 300.0),
        max_disappeared=processing_config.get('max_disappeared', 60),
        confirmation_frames=processing_config.get('confirmation_frames', 5),
        output_base_dir=json_output_dir,
        save_cropped_faces=processing_config.get('save_cropped_faces', False)
    )
    
    # Load JSON data
    with open(json_path, 'r') as f:
        frames_data = json.load(f)
    
    total_frames = len(frames_data)
    processed_frames = 0
    total_faces_detected = 0
    
    # Sort frames by frame_id to ensure sequential processing
    frames_data.sort(key=lambda x: x.get('frame_id', 0))
    
    # Initialize video writer variables (will be set when first frame is processed)
    out = None
    fps = processing_config.get('custom_output_fps') or 30  # Use custom FPS or default to 30
    
    print(f"Processing {total_frames} frames from JSON")
    print(f"Output FPS: {fps}")
    print(f"Output resolution: {json_output_dir}")
    
    for frame_data in frames_data:
        frame_id = frame_data.get('frame_id', processed_frames + 1)
        frame_url = frame_data['frame_path']
        
        if processed_frames % 30 == 0:
            print(f"Processing frame {processed_frames + 1}/{total_frames}")
        
        # Fetch frame from URL
        frame = fetch_frame_from_url(frame_url)
        if frame is None:
            continue
            
        # Initialize video writer with first successfully fetched frame
        if out is None:
            height, width = frame.shape[:2]
            video_output_path = os.path.join(json_output_dir, os.path.basename(output_name))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        frame_timestamp_ms = resolve_frame_timestamp_ms(frame_data, processed_frames, fps)
        
        # Process frame for face detection (same logic as video processing)
        orig = frame.copy()
        img = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        inputs = grpcclient.InferInput(
            "RetinaFace::input_0",
            img.shape,
            "FP32"
        )
        inputs.set_data_from_numpy(img)
        
        outputs = [
            grpcclient.InferRequestedOutput("1156"),
            grpcclient.InferRequestedOutput("1235"),
            grpcclient.InferRequestedOutput("1314")
        ]
        
        result = client.infer(
            MODEL_NAME,
            inputs=[inputs],
            outputs=outputs
        )
        
        loc = result.as_numpy("1156")[0]
        conf_raw = result.as_numpy("1235")[0]
        exp = np.exp(conf_raw - np.max(conf_raw, axis=1, keepdims=True))
        conf_softmax = exp / np.sum(exp, axis=1, keepdims=True)
        conf = conf_softmax[:, 1]
        
        boxes = decode(loc, priors)
        
        h, w = orig.shape[:2]
        boxes[:,0] *= w
        boxes[:,1] *= h
        boxes[:,2] *= w
        boxes[:,3] *= h
        
        inds = np.where(conf > CONF_THRESHOLD)[0]
        boxes = boxes[inds]
        scores = conf[inds]
        keep = nms(boxes, scores, NMS_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Generate embeddings for all detected faces BEFORE tracking
        embeddings_data = {}
        face_crops = {}
        
        if processed_frames == 1:  # Debug log on first frame
            print(f"🔍 Debug: generate_embeddings={processing_config.get('generate_embeddings', False)}")
            print(f"🔍 Debug: embedding_generator={'initialized' if embedding_generator is not None else 'None'}")
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Validate and clamp bounding box coordinates
            h, w = orig.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Ensure minimum face size for cropping
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            # Crop face for embedding generation
            face_crop = orig[y1:y2, x1:x2]
            face_crops[int(i)] = face_crop
            
            # Generate embedding for this face
            embedding = generate_face_embedding(face_crop, processing_config)
            if embedding is not None:
                embeddings_data[int(i)] = embedding
                if processed_frames == 1 and i == 0:  # Debug log first embedding
                    print(f"✅ Debug: Generated embedding for face {i}, size={len(embedding.get('embedding', []))}")
        
        # Process faces for tracking with embeddings
        tracked_faces = face_tracker.process_faces(
            orig, boxes, scores, frame_id, embeddings_data, frame_timestamp_ms
        )
        total_faces_detected += len(boxes)
        
        # Draw rectangles and tracking info
        for tracking_id, face_info in tracked_faces.items():
            box = face_info['bbox']
            confidence = face_info['confidence']
            is_new = face_info['is_new']
            
            x1, y1, x2, y2 = box.astype(int)
            face_tracker.crop_and_save_face(orig, box, tracking_id, frame_id)
            
            color = (0, 255, 0) if not is_new else (255, 0, 0)
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            
            text = f"ID:{tracking_id} {confidence:.3f}"
            if is_new:
                text += " NEW"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(orig, (x1, y1 - text_size[1] - 7), (x1 + text_size[0], y1 + 2), (0, 0, 0), -1)
            cv2.putText(orig, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write frame to output video
        if out is not None:
            out.write(orig)
        
        processed_frames += 1
        
        if processed_frames % 30 == 0:
            print(f"Frame {processed_frames}: Faces detected: {len(boxes)} | Total so far: {total_faces_detected}")
    
    # Clean up
    if out is not None:
        out.release()
    
    # Get final statistics
    final_stats = face_tracker.get_statistics()
    face_summary = face_tracker.get_face_summary()
    
    avg_faces_per_frame = total_faces_detected / processed_frames if processed_frames > 0 else 0
    print(f"\nJSON processing complete!")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Average faces per frame: {avg_faces_per_frame:.2f}")
    print(f"\n{'='*40}")
    print("FACE TRACKING STATISTICS:")
    print(f"{'='*40}")
    print(f"Total unique faces: {final_stats['total_unique_faces']}")
    print(f"Total face appearances: {final_stats['total_appearances']}")
    print(f"Active tracks: {final_stats['active_tracks']}")
    
    if face_summary:
        print(f"\nFace Summary (ID: Appearance Count):")
        for tracking_id, count in face_summary.items():
            print(f"  Person {tracking_id}: {count} detections")
    
    # Export face data
    face_tracker.export_face_data()
    print(f"\n✅ Face images saved to: {os.path.join(json_output_dir, 'faces')}/")
    print(f"✅ Face data exported to: {os.path.join(json_output_dir, 'face_data_export.json')}")
    print(f"✅ Face samples saved to: {os.path.join(json_output_dir, 'face_samples')}/")
    
    return video_output_path if 'video_output_path' in locals() else None

# Function to process video frames
def process_video(video_path, output_name, video_base_name, processing_config):
    # Create video-specific output directory
    video_output_dir = os.path.join("output", video_base_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Initialize face tracker for this specific video (no database)
    global face_tracker
    face_tracker = RobustFaceTracker(
        db_path=None,  # Database disabled
        similarity_threshold=processing_config.get('similarity_threshold', 0.65),
        spatial_threshold=processing_config.get('spatial_threshold', 300.0),
        max_disappeared=processing_config.get('max_disappeared', 60),
        confirmation_frames=processing_config.get('confirmation_frames', 5),
        output_base_dir=video_output_dir,  # Video-specific output directory
        save_cropped_faces=processing_config.get('save_cropped_faces', False)
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    custom_fps = processing_config.get('custom_output_fps')
    fps = custom_fps if custom_fps else (original_fps if original_fps > 0 else 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling interval if custom FPS is set
    frame_interval = 1
    if custom_fps and custom_fps < original_fps:
        frame_interval = int(round(original_fps / custom_fps))
        print(f"Frame sampling enabled: processing every {frame_interval} frame(s)")
    
    # Create output video writer in video-specific directory
    video_output_path = os.path.join(video_output_dir, os.path.basename(output_name))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    total_faces_detected = 0
    
    print(f"Processing video: {total_frames} frames")
    print(f"Original FPS: {original_fps}, Output FPS: {fps} {'(custom setting)' if fps != original_fps else '(original)'}")
    print(f"Output resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames based on sampling interval
        if (frame_count - 1) % frame_interval != 0:
            continue
        
        processed_count += 1
        if processed_count % 30 == 0:  # Progress update every 30 processed frames
            print(f"Processing frame {frame_count}/{total_frames} (processed: {processed_count})")
        
        # Process frame for face detection
        orig = frame.copy()
        img = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        inputs = grpcclient.InferInput(
            "RetinaFace::input_0",
            img.shape,
            "FP32"
        )
        
        inputs.set_data_from_numpy(img)
        
        outputs = [
            grpcclient.InferRequestedOutput("1156"),
            grpcclient.InferRequestedOutput("1235"),
            grpcclient.InferRequestedOutput("1314")
        ]
        
        result = client.infer(
            MODEL_NAME,
            inputs=[inputs],
            outputs=outputs
        )
        
        loc = result.as_numpy("1156")[0]
        # conf = result.as_numpy("1235")[0][:,1]
        conf_raw = result.as_numpy("1235")[0]      # (15960, 2)
        # stable softmax
        exp = np.exp(conf_raw - np.max(conf_raw, axis=1, keepdims=True))
        conf_softmax = exp / np.sum(exp, axis=1, keepdims=True)
        conf = conf_softmax[:, 1]   # real probability
        
        boxes = decode(loc, priors)
        
        h, w = orig.shape[:2]
        
        boxes[:,0] *= w
        boxes[:,1] *= h
        boxes[:,2] *= w
        boxes[:,3] *= h
        
        inds = np.where(conf > CONF_THRESHOLD)[0]
        
        boxes = boxes[inds]
        scores = conf[inds]
        
        keep = nms(boxes, scores, NMS_THRESHOLD)
        
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Generate embeddings for all detected faces BEFORE tracking
        embeddings_data = {}
        face_crops = {}
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Validate and clamp bounding box coordinates
            h, w = orig.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Ensure minimum face size for cropping
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            # Crop face for embedding generation
            face_crop = orig[y1:y2, x1:x2]
            face_crops[int(i)] = face_crop  # Ensure int key
            
            # Generate embedding for this face
            embedding = generate_face_embedding(face_crop, processing_config)
            if embedding is not None:
                embeddings_data[int(i)] = embedding  # Ensure int key
        
        frame_timestamp_ms = int(((frame_count - 1) / original_fps) * 1000) if original_fps > 0 else None
        
        # Process faces for tracking and duplicate filtering
        tracked_faces = face_tracker.process_faces(
            orig, boxes, scores, frame_count, embeddings_data, frame_timestamp_ms
        )
        
        total_faces_detected += len(boxes)
        
        # Draw rectangles, confidence scores, and tracking IDs on frame
        for tracking_id, face_info in tracked_faces.items():
            box = face_info['bbox']
            confidence = face_info['confidence']
            is_new = face_info['is_new']
            
            x1, y1, x2, y2 = box.astype(int)
            
            # Crop and save face for this person
            face_tracker.crop_and_save_face(orig, box, tracking_id, frame_count)
            
            # Choose color based on whether this is a new face
            color = (0, 255, 0) if not is_new else (255, 0, 0)  # Green for existing, blue for new
            
            # Draw bounding box
            cv2.rectangle(
                orig,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            
            # Add tracking ID and confidence score text
            text = f"ID:{tracking_id} {confidence:.3f}"
            if is_new:
                text += " NEW"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - 5  # Position above the box
            
            # Draw background rectangle for better text visibility
            cv2.rectangle(
                orig,
                (text_x, text_y - text_size[1] - 2),
                (text_x + text_size[0], text_y + 2),
                (0, 0, 0),
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                orig,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # # Add face count text on frame
        # stats = face_tracker.get_statistics()
        # cv2.putText(
        #     orig,
        #     f"Faces: {len(boxes)} | Unique: {stats['total_unique_faces']}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2
        # )
        
        # Write frame to output video
        out.write(orig)
        
        if frame_count % 60 == 0:  # Show detailed stats every 60 frames
            print(f"Frame {frame_count}: Faces detected: {len(boxes)} | Total so far: {total_faces_detected}")
    
    cap.release()
    out.release()
    
    # Get final tracking statistics
    final_stats = face_tracker.get_statistics()
    face_summary = face_tracker.get_face_summary()
    
    avg_faces_per_frame = total_faces_detected / frame_count if frame_count > 0 else 0
    print(f"\nVideo processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Average faces per frame: {avg_faces_per_frame:.2f}")
    print(f"\n{'='*40}")
    print("FACE TRACKING STATISTICS:")
    print(f"{'='*40}")
    print(f"Total unique faces: {final_stats['total_unique_faces']}")
    print(f"Total face appearances: {final_stats['total_appearances']}")
    print(f"Active tracks: {final_stats['active_tracks']}")
    
    if face_summary:
        print(f"\nFace Summary (ID: Appearance Count):")
        for tracking_id, count in face_summary.items():
            print(f"  Person {tracking_id}: {count} detections")
    
    # Export face data
    face_tracker.export_face_data()  # Will use default path in video output directory
    print(f"\n✅ Face images saved to: {os.path.join(video_output_dir, 'faces')}/")
    print(f"✅ Face data exported to: {os.path.join(video_output_dir, 'face_data_export.json')}")
    print(f"✅ Face samples saved to: {os.path.join(video_output_dir, 'face_samples')}/")
    
    return video_output_path  # Return the actual output video path

# Function to process video from bytes array
def process_video_bytes(video_bytes, output_name, video_base_name, metadata, processing_config):
    """Process video from bytes array input."""
    # Create temporary file from bytes
    temp_dir = tempfile.gettempdir()
    temp_filename = f"temp_video_{uuid.uuid4().hex}.mp4"
    temp_video_path = os.path.join(temp_dir, temp_filename)
    
    try:
        # Write bytes to temporary file
        with open(temp_video_path, 'wb') as temp_file:
            if isinstance(video_bytes, str):
                # If bytes are base64 encoded
                import base64
                video_bytes = base64.b64decode(video_bytes)
            temp_file.write(video_bytes)
        
        print(f"💾 Created temporary video file: {temp_video_path}")
        
        # Verify the temporary file can be opened by OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ValueError("Could not decode video from bytes data")
        cap.release()
        
        # Use existing process_video function
        result = process_video(temp_video_path, output_name, video_base_name, processing_config)
        
        print(f"✅ Successfully processed video from bytes")
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"🗑️ Cleaned up temporary video file")

# Function to process frame bytes from JSON
def process_frame_bytes_json(frame_bytes_data, output_name, json_base_name, metadata, processing_config):
    """Process frames from JSON containing base64 encoded frame bytes."""
    # Create output directory
    json_output_dir = os.path.join("output", json_base_name)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Initialize face tracker
    global face_tracker
    face_tracker = RobustFaceTracker(
        db_path=None,
        similarity_threshold=processing_config.get('similarity_threshold', 0.65),
        spatial_threshold=processing_config.get('spatial_threshold', 300.0),
        max_disappeared=processing_config.get('max_disappeared', 60),
        confirmation_frames=processing_config.get('confirmation_frames', 5),
        output_base_dir=json_output_dir,
        save_cropped_faces=processing_config.get('save_cropped_faces', False)
    )
    
    frames_data = frame_bytes_data
    if isinstance(frame_bytes_data, str):
        # If it's a JSON string, parse it
        frames_data = json.loads(frame_bytes_data)
    
    total_frames = len(frames_data)
    processed_frames = 0
    total_faces_detected = 0
    
    # Sort frames by frame_id
    frames_data.sort(key=lambda x: x.get('frame_id', 0))
    
    print(f"Processing {total_frames} frames from bytes JSON")
    print(f"Output directory: {json_output_dir}")
    
    # Initialize video writer variables
    out = None
    fps = metadata.get('fps') if metadata else None
    fps = fps or processing_config.get('custom_output_fps') or 30
    
    for frame_data in frames_data:
        frame_id = frame_data.get('frame_id', processed_frames + 1)
        frame_bytes = frame_data.get('frame_data') or frame_data.get('frame_bytes')
        
        if not frame_bytes:
            print(f"❌ No frame data found for frame {frame_id}")
            continue
            
        print(f"Processing frame {processed_frames + 1}/{total_frames} (ID: {frame_id})")
        
        try:
            # Decode frame from bytes
            if isinstance(frame_bytes, str):
                # Base64 encoded
                import base64
                frame_bytes = base64.b64decode(frame_bytes)
            
            # Convert bytes to numpy array and decode
            img_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print(f"❌ Failed to decode frame {frame_id} from bytes")
                continue
            
            print(f"✅ Successfully decoded frame {frame_id}, shape: {frame.shape}")
            
        except Exception as e:
            print(f"❌ Error decoding frame {frame_id}: {e}")
            continue
            
        # Initialize video writer with first successfully processed frame
        if out is None:
            height, width = frame.shape[:2]
            video_output_path = os.path.join(json_output_dir, os.path.basename(output_name))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
            print(f"Output resolution: {width}x{height}")
        
        frame_timestamp_ms = resolve_frame_timestamp_ms(frame_data, processed_frames, fps)
        
        # Process frame for face detection (same logic as other processing functions)
        orig = frame.copy()
        img = cv2.resize(frame, (INPUT_W, INPUT_H))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        inputs = grpcclient.InferInput(
            "RetinaFace::input_0",
            img.shape,
            "FP32"
        )
        inputs.set_data_from_numpy(img)
        
        outputs = [
            grpcclient.InferRequestedOutput("1156"),
            grpcclient.InferRequestedOutput("1235"),
            grpcclient.InferRequestedOutput("1314")
        ]
        
        result = client.infer(
            MODEL_NAME,
            inputs=[inputs],
            outputs=outputs
        )
        
        loc = result.as_numpy("1156")[0]
        conf_raw = result.as_numpy("1235")[0]
        exp = np.exp(conf_raw - np.max(conf_raw, axis=1, keepdims=True))
        conf_softmax = exp / np.sum(exp, axis=1, keepdims=True)
        conf = conf_softmax[:, 1]
        
        boxes = decode(loc, priors)
        
        h, w = orig.shape[:2]
        boxes[:,0] *= w
        boxes[:,1] *= h
        boxes[:,2] *= w
        boxes[:,3] *= h
        
        inds = np.where(conf > CONF_THRESHOLD)[0]
        boxes = boxes[inds]
        scores = conf[inds]
        keep = nms(boxes, scores, NMS_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Process faces for tracking
        print(f"🔍 Detected {len(boxes)} faces in frame {frame_id}")
        tracked_faces = face_tracker.process_faces(
            orig, boxes, scores, frame_id, frame_timestamp_ms=frame_timestamp_ms
        )
        total_faces_detected += len(boxes)
        print(f"📊 Tracked faces: {len(tracked_faces)}")
        
        # Draw rectangles and tracking info
        for tracking_id, face_info in tracked_faces.items():
            box = face_info['bbox']
            confidence = face_info['confidence']
            is_new = face_info['is_new']
            
            x1, y1, x2, y2 = box.astype(int)
            face_tracker.crop_and_save_face(orig, box, tracking_id, frame_id)
            
            color = (0, 255, 0) if not is_new else (255, 0, 0)
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            
            text = f"ID:{tracking_id} {confidence:.3f}"
            if is_new:
                text += " NEW"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(orig, (x1, y1 - text_size[1] - 7), (x1 + text_size[0], y1 + 2), (0, 0, 0), -1)
            cv2.putText(orig, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write frame to output video
        if out is not None:
            out.write(orig)
        
        processed_frames += 1
        
        if processed_frames % 30 == 0:
            print(f"Frame {processed_frames}: Faces detected: {len(boxes)} | Total so far: {total_faces_detected}")
    
    # Clean up
    if out is not None:
        out.release()
    
    # Get final statistics
    final_stats = face_tracker.get_statistics()
    face_summary = face_tracker.get_face_summary()
    
    avg_faces_per_frame = total_faces_detected / processed_frames if processed_frames > 0 else 0
    print(f"\nFrame bytes JSON processing complete!")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Average faces per frame: {avg_faces_per_frame:.2f}")
    print(f"\n{'='*40}")
    print("FACE TRACKING STATISTICS:")
    print(f"{'='*40}")
    print(f"Total unique faces: {final_stats['total_unique_faces']}")
    print(f"Total face appearances: {final_stats['total_appearances']}")
    print(f"Active tracks: {final_stats['active_tracks']}")
    
    if face_summary:
        print(f"\nFace Summary (ID: Appearance Count):")
        for tracking_id, count in face_summary.items():
            print(f"  Person {tracking_id}: {count} detections")
    
    # Export face data
    face_tracker.export_face_data()
    print(f"\n✅ Face images saved to: {os.path.join(json_output_dir, 'faces')}/")
    print(f"✅ Face data exported to: {os.path.join(json_output_dir, 'face_data_export.json')}")
    print(f"✅ Face samples saved to: {os.path.join(json_output_dir, 'face_samples')}/")
    
    return video_output_path if 'video_output_path' in locals() else None

# Generate priors
def generate_priors():

    min_sizes = [[16,32],[64,128],[256,512]]
    steps = [8,16,32]

    priors = []

    for k, step in enumerate(steps):

        feature_h = INPUT_H // step
        feature_w = INPUT_W // step

        for i in range(feature_h):
            for j in range(feature_w):

                for min_size in min_sizes[k]:

                    cx = (j+0.5)*step/INPUT_W
                    cy = (i+0.5)*step/INPUT_H

                    s_kx = min_size/INPUT_W
                    s_ky = min_size/INPUT_H

                    priors.append([cx,cy,s_kx,s_ky])

    return np.array(priors,dtype=np.float32)

# Decode boxes
def decode(loc, priors):

    variances=[0.1,0.2]

    boxes=np.concatenate((
        priors[:,:2]+loc[:,:2]*variances[0]*priors[:,2:],
        priors[:,2:]*np.exp(loc[:,2:]*variances[1])
    ),axis=1)

    boxes[:,:2]-=boxes[:,2:]/2
    boxes[:,2:]+=boxes[:,:2]

    return boxes

# NMS
def nms(boxes,scores,threshold):

    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]

    areas=(x2-x1)*(y2-y1)
    order=scores.argsort()[::-1]

    keep=[]

    while order.size>0:

        i=order[0]
        keep.append(i)

        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])

        w=np.maximum(0,xx2-xx1)
        h=np.maximum(0,yy2-yy1)

        inter=w*h
        iou=inter/(areas[i]+areas[order[1:]]-inter)

        inds=np.where(iou<=threshold)[0]
        order=order[inds+1]

    return keep

priors=generate_priors()

# Create main output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

def main(config_file=None):
    """
    Main function for API integration.
    Process videos using the specified config file.
    """
    if config_file:
        # Use provided config file path
        config_path = config_file
    else:
        config_path = CONFIG_FILE
    
    # Execute the existing processing logic but return results for API
    try:
        # This calls the existing processing logic that was already in the script
        exec_result = process_all_inputs(config_path)
        return {"success": True, "message": "Processing completed successfully"}
        
    except Exception as e:
        print(f"❌ Error in main processing: {e}")
        return {"success": False, "error": str(e)}

def process_all_inputs(config_path=None):
    """
    Process all inputs from configuration - extracted from the main script logic
    """
    global ceph_client, CONF_THRESHOLD, NMS_THRESHOLD
    
    if config_path is None:
        config_path = CONFIG_FILE
    
    # Initialize ceph client (keeping original logic)
    print("============================================================")
    print("INITIALIZING CEPH CLIENT")
    print("============================================================")
    try:
        config_loader = ConfigLoader('CephTest/config.yaml')
        ceph_config = config_loader.load()
        logger = LoggerSetup.setup_logger(ceph_config)
        ceph_client = CephClient(ceph_config, logger)
        print("✅ Ceph client initialized successfully!")
    except Exception as e:
        print(f"⚠️ Could not initialize Ceph client: {e}")
        print("Will fallback to HTTP requests for frame fetching")
        ceph_client = None

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Using fallback configuration...")
        config = {
            'processing_config': {},
            'inputs': [{
                'id': 'fallback_video',
                'input_type': 'video_file',
                'data': {'path': 'input\\Dashcam_Helps_Police_Track_Down_Offenders_Heavy_Penalty.mp4'},
                'metadata': {'description': 'No description', 'enabled': True}
            }]
        }

    # Get processing configuration and apply defaults
    processing_config = config.get('processing_config', {})
    
    # Apply default values - set as global variables
    CONF_THRESHOLD = processing_config.get('confidence_threshold', 0.3)
    NMS_THRESHOLD = processing_config.get('nms_threshold', 0.3)
    
    print(f"⚙️ Using CONF_THRESHOLD: {CONF_THRESHOLD}")
    print(f"⚙️ Using NMS_THRESHOLD: {NMS_THRESHOLD}")

    # Initialize embedding generator if needed
    if processing_config.get('generate_embeddings', False):
        if not initialize_embedding_generator(processing_config):
            print("⚠️ Failed to initialize embedding generator, continuing without embeddings")

    # Process all inputs from config
    inputs = config.get('inputs', [])
    
    for input_config in inputs:
        # Skip disabled inputs
        if not input_config.get('metadata', {}).get('enabled', True):
            continue
            
        input_id = input_config.get('id', 'unknown')
        input_type = input_config.get('input_type', 'unknown')
        input_data = input_config.get('data', {})
        
        print(f"\n{'='*60}")
        print(f"Processing input: {input_id} ({input_type})")
        print(f"Description: {input_config.get('metadata', {}).get('description', 'No description')}")
        print(f"{'='*60}")
        
        try:
            if input_type == 'video_file':
                # Process local video file
                video_path = input_data.get('path')
                if not video_path or not os.path.exists(video_path):
                    print(f"❌ Video file not found: {video_path}")
                    continue
                
                # Extract video filename without extension for folder name
                video_filename = os.path.splitext(os.path.basename(video_path))[0]
                base_name = video_filename
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing video file: {video_path}")
                actual_output_path = process_video(video_path, output_name, video_base_name, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
                
            elif input_type == 'video_url':
                # Process video from URL
                video_url = input_data.get('url')
                if not video_url:
                    print(f"❌ No URL provided")
                    continue
                
                print(f"Downloading video from URL: {video_url}")
                response = requests.get(video_url, stream=True)
                temp_video_path = f"temp_video_{input_id}.mp4"
                with open(temp_video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract filename from URL for folder name
                from urllib.parse import urlparse
                parsed_url = urlparse(video_url)
                url_filename = os.path.splitext(os.path.basename(parsed_url.path))[0]
                base_name = url_filename if url_filename else input_id
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                actual_output_path = process_video(temp_video_path, output_name, video_base_name, processing_config)
                output_path = actual_output_path
                
                # Clean up temporary file
                os.remove(temp_video_path)
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
                
            elif input_type == 'json_ceph_urls':
                json_path = input_data.get('path')
                if not json_path or not os.path.exists(json_path):
                    print(f"❌ JSON file not found: {json_path}")
                    continue
                
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing JSON file with Ceph URLs: {json_path}")
                actual_output_path = process_json_frames(json_path, output_name, json_base_name, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
            
            elif input_type == 'json_frame_bytes':
                json_path = input_data.get('path')
                if not json_path or not os.path.exists(json_path):
                    print(f"❌ JSON file not found: {json_path}")
                    continue
                
                print(f"Processing JSON file with frame bytes: {json_path}")
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, dict) and 'frames' in json_data:
                    frame_bytes_data = json_data['frames']
                    metadata = json_data.get('metadata', {})
                else:
                    frame_bytes_data = json_data
                    metadata = {}
                
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                actual_output_path = process_frame_bytes_json(frame_bytes_data, output_name, json_base_name, metadata, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
            
            elif input_type == 'video_bytes':
                video_bytes = input_data.get('video_data')
                if not video_bytes:
                    print(f"❌ No video bytes data provided")
                    continue
                
                base_name = f"{input_id}_video_bytes"
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing video from bytes array...")
                actual_output_path = process_video_bytes(video_bytes, output_name, video_base_name, input_data, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
            
            elif input_type == 'frame_bytes_direct':
                frame_bytes_data = input_data.get('frames')
                if not frame_bytes_data:
                    print(f"❌ No frame bytes data provided")
                    continue
                
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing frames from direct bytes data...")
                actual_output_path = process_frame_bytes_json(frame_bytes_data, output_name, json_base_name, input_data, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
            
            # Add other input types as needed...
            else:
                print(f"❌ Unsupported input type: {input_type}")
                continue
                
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            continue

    print(f"\n{'='*60}")
    print("🎉 All input processing completed!")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    # When run directly, execute the processing logic
    # Initialize Ceph client for S3A URL handling
    print("\n" + "="*60)
    print("INITIALIZING CEPH CLIENT")
    print("="*60)
    initialize_ceph_client()

    # Load input configuration
    enabled_inputs, processing_config = load_input_configuration()

    # Extract processing parameters from config
    CONF_THRESHOLD = processing_config.get('confidence_threshold', 0.3)
    NMS_THRESHOLD = processing_config.get('nms_threshold', 0.3)
    print(f"⚙️ Using CONF_THRESHOLD: {CONF_THRESHOLD}")
    print(f"⚙️ Using NMS_THRESHOLD: {NMS_THRESHOLD}")

    # Process each configured input
    for idx, input_config in enumerate(enabled_inputs):
        
        input_id = input_config.get('id', f'input_{idx+1}')
        input_type = input_config.get('input_type')
        input_data = input_config.get('data', {})
        input_metadata = input_config.get('metadata', {})
        
        print(f"\n{'='*60}")
        print(f"Processing input {idx+1}: {input_id} ({input_type})")
        print(f"Description: {input_metadata.get('description', 'No description')}")
        print(f"{'='*60}")
        
        try:
            
            if input_type == 'video_file':
                # Process local video file
                video_path = input_data.get('path')
                if not video_path or not os.path.exists(video_path):
                    print(f"❌ Video file not found: {video_path}")
                    continue
                
                # Extract video filename without extension for folder name
                video_filename = os.path.splitext(os.path.basename(video_path))[0]
                base_name = video_filename
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing local video file: {video_path}")
                actual_output_path = process_video(video_path, output_name, video_base_name, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
                
            elif input_type == 'video_url':
                # Process video from URL
                video_url = input_data.get('url')
                if not video_url:
                    print(f"❌ No video URL provided")
                    continue
                
                print(f"Downloading video from URL: {video_url}")
                response = requests.get(video_url, stream=True)
                temp_video_path = f"temp_video_{input_id}.mp4"
                with open(temp_video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract filename from URL for folder name
                from urllib.parse import urlparse
                parsed_url = urlparse(video_url)
                url_filename = os.path.splitext(os.path.basename(parsed_url.path))[0]
                base_name = url_filename if url_filename else input_id
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                actual_output_path = process_video(temp_video_path, output_name, video_base_name, processing_config)
                output_path = actual_output_path
                
                # Clean up temporary file
                os.remove(temp_video_path)
                print(f"Cleaned up temporary file: {temp_video_path}")
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
                
            elif input_type == 'json_ceph_urls':
                # Process JSON file containing Ceph URLs
                json_path = input_data.get('path')
                if not json_path or not os.path.exists(json_path):
                    print(f"❌ JSON file not found: {json_path}")
                    continue
                
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing JSON file with Ceph URLs: {json_path}")
                actual_output_path = process_json_frames(json_path, output_name, json_base_name, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
                
            elif input_type == 'json_frame_bytes':
                # Process JSON file containing frame bytes
                json_path = input_data.get('path')
                if not json_path or not os.path.exists(json_path):
                    print(f"❌ JSON file not found: {json_path}")
                    continue
                
                print(f"Processing JSON file with frame bytes: {json_path}")
                
                # Load JSON data from file
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                # Extract frames data and metadata
                if isinstance(json_data, dict) and 'frames' in json_data:
                    frame_bytes_data = json_data['frames']
                    metadata = json_data.get('metadata', {})
                else:
                    # Direct list format (backward compatibility)
                    frame_bytes_data = json_data
                    metadata = {}
                
                # Generate base name from input_id or metadata
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                # Process frame bytes
                actual_output_path = process_frame_bytes_json(frame_bytes_data, output_name, json_base_name, metadata, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
                
            elif input_type == 'video_bytes':
                # Process video from bytes array
                video_bytes = input_data.get('video_data')
                if not video_bytes:
                    print(f"❌ No video bytes data provided")
                    continue
                
                # Use descriptive name for video bytes processing
                base_name = f"{input_id}_video_bytes"
                video_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing video from bytes array...")
                actual_output_path = process_video_bytes(video_bytes, output_name, video_base_name, input_data, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 Video output directory: {os.path.abspath(os.path.dirname(output_path))}")
                print(f"📁 All video outputs saved to: {os.path.join('output', video_base_name)}/")
                
            elif input_type == 'frame_bytes_direct':
                # Process frame bytes directly from config
                frame_bytes_data = input_data.get('frames')
                if not frame_bytes_data:
                    print(f"❌ No frame bytes data provided")
                    continue
                    
                base_name = input_id
                json_base_name = base_name
                output_name = f"processed_{base_name}.mp4"
                
                print(f"Processing frames from direct bytes data...")
                actual_output_path = process_frame_bytes_json(frame_bytes_data, output_name, json_base_name, input_data, processing_config)
                output_path = actual_output_path
                
                print(f"✅ Saved processed video: {output_path}")
                print(f"📁 JSON output directory: {os.path.abspath(os.path.dirname(output_path)) if output_path else 'N/A'}")
                print(f"📁 All JSON outputs saved to: {os.path.join('output', json_base_name)}/")
                
            else:
                print(f"❌ Unsupported input type: {input_type}")
                continue
                
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            continue

    print(f"\n{'='*60}")
    print("🎉 All input processing completed!")
    print(f"{'='*60}")
