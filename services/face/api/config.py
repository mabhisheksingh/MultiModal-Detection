"""
API Configuration
================
Configuration settings for the Face Detection REST API server.
"""

import os

# Server Configuration
SERVER_HOST = "0.0.0.0"  # Bind to all interfaces (allows remote access)
SERVER_PORT = 80
DEBUG_MODE = False  # Production mode for remote deployment
MAX_FILE_SIZE_MB = 500  # Maximum upload file size in MB

# Directory Configuration
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
UPLOAD_FOLDER = os.path.join(API_DIR, 'uploads')
TEMP_FOLDER = os.path.join(API_DIR, 'temp')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output')

# Job Configuration
MAX_CONCURRENT_JOBS = 3
JOB_CLEANUP_HOURS = 24
JOB_TIMEOUT_MINUTES = 60

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

# API Configuration
API_VERSION = "v1"
API_BASE_URL = f"/api/{API_VERSION}"

# Triton Server Configuration
TRITON_SERVER_URL = "localhost:8001"
TRITON_MODEL_NAME = "retinaface"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(API_DIR, 'api.log')

# Default Processing Configuration
DEFAULT_PROCESSING_CONFIG = {
    "confidence_threshold": 0.3,
    "nms_threshold": 0.3,
    "similarity_threshold": 0.65,
    "spatial_threshold": 300.0,
    "max_disappeared": 60,
    "confirmation_frames": 5,
    "save_cropped_faces": False,
    "generate_embeddings": True,
    "embedding_model": "ArcFace",
    "embedding_detector_backend": "opencv",
    "min_face_size": 10,
    "enable_face_alignment": True,
    "custom_output_fps": 10
}

# CORS Configuration (for web interfaces)
CORS_ORIGINS = ["*"]  # In production, specify actual domains

# Security Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change this in production
REQUIRE_API_KEY = False  # Set to True for API key authentication

def ensure_directories():
    """Ensure required directories exist."""
    directories = [UPLOAD_FOLDER, TEMP_FOLDER, OUTPUT_FOLDER]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_config_dict():
    """Get configuration as dictionary."""
    return {
        "server_port": SERVER_PORT,
        "debug_mode": DEBUG_MODE,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "upload_folder": UPLOAD_FOLDER,
        "temp_folder": TEMP_FOLDER,
        "output_folder": OUTPUT_FOLDER,
        "triton_server_url": TRITON_SERVER_URL,
        "api_version": API_VERSION
    }
