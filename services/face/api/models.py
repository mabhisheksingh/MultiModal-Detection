"""
API Models and Constants
========================
Data models, status constants, and response helpers for the Face Detection API.
"""

from flask import jsonify

class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class APIResponse:
    """Standard API response helpers."""
    
    @staticmethod
    def success(data, status_code=200):
        """Create successful response."""
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }), status_code
    
    @staticmethod
    def error(message, status_code=400, error_code=None):
        """Create error response."""
        response = {
            "success": False,
            "error": {
                "message": message,
                "status_code": status_code
            },
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        if error_code:
            response["error"]["code"] = error_code
            
        return jsonify(response), status_code

class ProcessingConfig:
    """Default processing configuration."""
    
    DEFAULT_CONFIG = {
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
    
    @staticmethod
    def merge_with_defaults(config):
        """Merge provided config with defaults."""
        merged = ProcessingConfig.DEFAULT_CONFIG.copy()
        if config:
            merged.update(config)
        return merged

class JobRequest:
    """Job request data model."""
    
    def __init__(self, config_data):
        self.processing_config = config_data.get('processing_config', {})
        self.inputs = config_data.get('inputs', [])
        
        # Merge processing config with defaults
        self.processing_config = ProcessingConfig.merge_with_defaults(self.processing_config)
    
    def validate(self):
        """Validate request data."""
        errors = []
        
        if not self.inputs:
            errors.append("No inputs specified")
        
        for i, inp in enumerate(self.inputs):
            if 'input_type' not in inp:
                errors.append(f"Input {i}: missing input_type")
            
            if 'data' not in inp:
                errors.append(f"Input {i}: missing data")
            
            # Validate specific input types
            input_type = inp.get('input_type')
            data = inp.get('data', {})
            
            if input_type == 'video_file':
                if 'path' not in data:
                    errors.append(f"Input {i}: video_file requires 'path' in data")
            
            elif input_type == 'video_url':
                if 'url' not in data:
                    errors.append(f"Input {i}: video_url requires 'url' in data")
        
        return errors

class JobResponse:
    """Job response data model."""
    
    def __init__(self, job):
        self.job = job
    
    def to_dict(self):
        """Convert to dictionary for JSON response."""
        response = {
            "job_id": self.job["job_id"],
            "status": self.job["status"],
            "created_at": self.job["created_at"],
            "progress_percentage": self.job.get("progress_percentage", 0)
        }
        
        # Add stage info if processing
        if self.job["status"] == JobStatus.PROCESSING:
            response.update({
                "current_stage": self.job.get("current_stage"),
                "started_at": self.job.get("started_at")
            })
        
        # Add completion info if done
        if self.job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            response["completed_at"] = self.job.get("completed_at")
            response["processing_time"] = self.job.get("processing_time")
        
        # Add results if completed
        if self.job["status"] == JobStatus.COMPLETED:
            response["results"] = self.job.get("results", {})
        
        # Add error if failed
        if self.job["status"] == JobStatus.FAILED:
            response["error"] = self.job.get("error")
        
        return response
