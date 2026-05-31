"""
Flask REST API Server for Face Detection & ArcFace Embedding Pipeline
====================================================================
This API wraps the existing video_client.py pipeline to provide HTTP endpoints
for processing videos with face detection and ArcFace embedding generation.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from threading import Thread
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import traceback

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_manager import JobManager
from models import JobStatus, APIResponse
import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE_MB * 1024 * 1024

# Initialize job manager
job_manager = JobManager()

# Health check endpoint
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Check API and service health."""
    try:
        # Import video_client to test pipeline availability
        import video_client
        
        # Check if Triton server is accessible (basic check)
        triton_status = "unknown"
        try:
            import tritonclient.grpc as grpcclient
            triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
            if triton_client.is_server_ready():
                triton_status = "connected"
            else:
                triton_status = "disconnected"
        except:
            triton_status = "error"
        
        # Check ArcFace model availability
        arcface_status = "unknown"
        try:
            from face_embedding_generator import ArcFaceEmbeddingGenerator
            generator = ArcFaceEmbeddingGenerator()
            arcface_status = "loaded"
        except Exception as e:
            arcface_status = f"error: {str(e)}"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "triton_server": triton_status,
                "arcface_model": arcface_status,
                "job_manager": "active",
                "active_jobs": len(job_manager.get_active_jobs())
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

# Main config processing endpoint
@app.route('/api/v1/process-config', methods=['POST'])
def process_config():
    """Process video using input_config.json format."""
    try:
        # Get JSON config from request
        config_data = request.get_json()
        
        if not config_data:
            return APIResponse.error("No configuration data provided", 400)
        
        # Validate required fields
        if 'processing_config' not in config_data:
            return APIResponse.error("Missing processing_config", 400)
        
        if 'inputs' not in config_data or not config_data['inputs']:
            return APIResponse.error("No inputs specified", 400)
        
        # Create job
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Initialize job
        job_manager.create_job(job_id, config_data)
        
        # Start processing in background thread
        processing_thread = Thread(
            target=_process_config_async,
            args=(job_id, config_data),
            daemon=True
        )
        processing_thread.start()
        
        # Return immediate response
        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "message": f"Processing started with {len(config_data['inputs'])} input(s)",
            "inputs_summary": [
                {
                    "id": inp.get('id', 'unknown'),
                    "type": inp.get('input_type', 'unknown'),
                    "enabled": inp.get('metadata', {}).get('enabled', True)
                } for inp in config_data['inputs']
            ],
            "estimated_time": "3-10 minutes per video",
            "status_url": f"/api/v1/job/{job_id}/status",
            "results_url": f"/api/v1/job/{job_id}/results"
        }), 202
        
    except Exception as e:
        app.logger.error(f"Error in process_config: {e}")
        app.logger.error(traceback.format_exc())
        return APIResponse.error(f"Internal server error: {str(e)}", 500)

# Video file upload endpoint
@app.route('/api/v1/process-video-file', methods=['POST'])
def process_video_file():
    """Process uploaded video file."""
    try:
        # Check if file is present
        if 'video_file' not in request.files:
            return APIResponse.error("No video file provided", 400)
        
        file = request.files['video_file']
        if file.filename == '':
            return APIResponse.error("No file selected", 400)
        
        # Get processing config
        config_str = request.form.get('config', '{}')
        try:
            processing_config = json.loads(config_str)
        except json.JSONDecodeError:
            return APIResponse.error("Invalid config JSON", 400)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(config.UPLOAD_FOLDER, filename)
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        file.save(upload_path)
        
        # Create config in standard format
        config_data = {
            "processing_config": processing_config,
            "inputs": [{
                "id": f"upload_{int(time.time())}",
                "input_type": "video_file",
                "data": {"path": upload_path},
                "metadata": {
                    "description": f"Uploaded file: {filename}",
                    "enabled": True,
                    "original_filename": filename
                }
            }]
        }
        
        # Create and start job
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_manager.create_job(job_id, config_data)
        
        processing_thread = Thread(
            target=_process_config_async,
            args=(job_id, config_data),
            daemon=True
        )
        processing_thread.start()
        
        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "message": f"Processing uploaded file: {filename}",
            "filename": filename,
            "estimated_time": "3-10 minutes",
            "status_url": f"/api/v1/job/{job_id}/status"
        }), 202
        
    except Exception as e:
        app.logger.error(f"Error in process_video_file: {e}")
        return APIResponse.error(f"Internal server error: {str(e)}", 500)

# Job status endpoint
@app.route('/api/v1/job/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job processing status."""
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            return APIResponse.error("Job not found", 404)
        
        return jsonify(job), 200
        
    except Exception as e:
        app.logger.error(f"Error getting job status: {e}")
        return APIResponse.error(f"Internal server error: {str(e)}", 500)

# Job results endpoint
@app.route('/api/v1/job/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get job results and download links."""
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            return APIResponse.error("Job not found", 404)
        
        if job['status'] != JobStatus.COMPLETED:
            return APIResponse.error("Job not completed yet", 400)
        
        # Build download URLs
        results = job.get('results', {})
        downloads = {}
        
        if 'output_paths' in results:
            for output_id, paths in results['output_paths'].items():
                downloads[output_id] = {}
                
                if 'video' in paths:
                    downloads[output_id]['processed_video'] = {
                        "url": f"/api/v1/job/{job_id}/download/{output_id}/video",
                        "filename": os.path.basename(paths['video']),
                        "description": "Annotated video with face tracking"
                    }
                
                if 'json' in paths:
                    downloads[output_id]['face_data_json'] = {
                        "url": f"/api/v1/job/{job_id}/download/{output_id}/json",
                        "filename": "face_data_export.json",
                        "description": "Face tracking data with ArcFace embeddings"
                    }
                
                if 'samples' in paths:
                    downloads[output_id]['face_samples'] = {
                        "url": f"/api/v1/job/{job_id}/download/{output_id}/samples",
                        "filename": "face_samples.zip",
                        "description": "Individual face crop images"
                    }
        
        return jsonify({
            "job_id": job_id,
            "status": job['status'],
            "completed_at": job.get('completed_at'),
            "processing_time": job.get('processing_time'),
            "statistics": job.get('statistics', {}),
            "downloads": downloads,
            "results_summary": results.get('summary', {})
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error getting job results: {e}")
        return APIResponse.error(f"Internal server error: {str(e)}", 500)

# File download endpoints
@app.route('/api/v1/job/<job_id>/download/<output_id>/video', methods=['GET'])
def download_video(job_id, output_id):
    """Download processed video file."""
    try:
        job = job_manager.get_job(job_id)
        if not job or job['status'] != JobStatus.COMPLETED:
            return APIResponse.error("Job not found or not completed", 404)
        
        video_path = job.get('results', {}).get('output_paths', {}).get(output_id, {}).get('video')
        if not video_path or not os.path.exists(video_path):
            return APIResponse.error("Video file not found", 404)
        
        return send_file(video_path, as_attachment=True)
        
    except Exception as e:
        return APIResponse.error(f"Download error: {str(e)}", 500)

@app.route('/api/v1/job/<job_id>/download/<output_id>/json', methods=['GET'])
def download_json(job_id, output_id):
    """Download face data JSON file."""
    try:
        job = job_manager.get_job(job_id)
        if not job or job['status'] != JobStatus.COMPLETED:
            return APIResponse.error("Job not found or not completed", 404)
        
        json_path = job.get('results', {}).get('output_paths', {}).get(output_id, {}).get('json')
        if not json_path or not os.path.exists(json_path):
            return APIResponse.error("JSON file not found", 404)
        
        return send_file(json_path, as_attachment=True, download_name="face_data_export.json")
        
    except Exception as e:
        return APIResponse.error(f"Download error: {str(e)}", 500)

# Background processing function
def _process_config_async(job_id, config_data):
    """Process configuration in background thread."""
    try:
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        # Import video processing module
        import video_client
        
        # Save config to temp file
        temp_config_path = os.path.join(config.TEMP_FOLDER, f"config_{job_id}.json")
        os.makedirs(config.TEMP_FOLDER, exist_ok=True)
        
        with open(temp_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Update job with processing info
        job_manager.update_job_progress(job_id, 10, "Starting video processing...")
        
        # IMPORTANT: Change working directory to project root
        # API runs from api/ folder, but video_client expects to run from project root
        original_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.abspath(__file__))  # Go up from api/ to project root
        project_root = os.path.dirname(project_root)  # Go up one more level to get to retinaface/
        
        try:
            os.chdir(project_root)
            print(f"🔧 Changed working directory to: {project_root}")
            
            # Convert temp config path to be relative to project root
            relative_temp_config = os.path.relpath(temp_config_path, project_root)
            print(f"🔧 Using config path: {relative_temp_config}")
            
            # Process using existing video_client
            start_time = time.time()
            
            # Call main processing function
            results = video_client.main(config_file=relative_temp_config)
            
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"🔧 Restored working directory to: {original_cwd}")
        
        processing_time = time.time() - start_time
        
        # Collect output information
        output_paths = {}
        statistics = {}
        
        # The existing pipeline saves outputs to output/ directory
        # We need to find the generated files in the project root output directory
        output_dir = os.path.join(project_root, "output")
        print(f"🔍 Looking for output files in: {output_dir}")
        
        if os.path.exists(output_dir):
            print(f"📁 Output directory exists, scanning...")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                print(f"🔍 Found item: {item} at {item_path}")
                
                if os.path.isdir(item_path):
                    # This is a video output folder
                    paths = {}
                    
                    # Look for processed video
                    files_in_dir = os.listdir(item_path)
                    print(f"📂 Files in {item}: {files_in_dir}")
                    
                    for file in files_in_dir:
                        full_file_path = os.path.join(item_path, file)
                        if file.startswith("processed_") and file.endswith(".mp4"):
                            paths['video'] = full_file_path
                            print(f"✅ Found video: {file}")
                        elif file == "face_data_export.json":
                            paths['json'] = full_file_path
                            print(f"✅ Found JSON: {file}")
                        elif file == "face_samples" and os.path.isdir(full_file_path):
                            paths['samples'] = full_file_path
                            print(f"✅ Found samples: {file}")
                    
                    if paths:
                        output_paths[item] = paths
                        print(f"✅ Added output paths for {item}: {paths}")
                    else:
                        print(f"❌ No valid files found in {item}")
        else:
            print(f"❌ Output directory does not exist: {output_dir}")
        
        # Mark job as completed
        job_manager.complete_job(job_id, {
            "output_paths": output_paths,
            "processing_time": f"{int(processing_time // 60):02d}:{int(processing_time % 60):02d}",
            "summary": {
                "outputs_generated": len(output_paths),
                "processing_duration_seconds": processing_time
            }
        })
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
    except Exception as e:
        app.logger.error(f"Error processing job {job_id}: {e}")
        app.logger.error(traceback.format_exc())
        job_manager.fail_job(job_id, str(e))

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return APIResponse.error("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(error):
    return APIResponse.error("Internal server error", 500)

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.TEMP_FOLDER, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Start the Flask server
    print("🚀 Starting Face Detection & ArcFace Embedding API Server...")
    print(f"📁 Upload folder: {config.UPLOAD_FOLDER}")
    print(f"📁 Temp folder: {config.TEMP_FOLDER}")
    print("🌐 Server running on: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000/api/v1/health")
    
    app.run(
        host='0.0.0.0',
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        threaded=True
    )
