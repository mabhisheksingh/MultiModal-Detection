"""
Face Detection API Test Client
=============================
Python client to test the REST API functionality.
Replaces curl commands with an easy-to-use Python script.
"""

import requests
import json
import time
import os
import sys
from datetime import datetime

class FaceDetectionAPIClient:
    """Client for Face Detection & ArcFace Embedding API."""
    
    def __init__(self, base_url="http://localhost:80"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def health_check(self):
        """Check API health status."""
        try:
            print("🔍 Checking API health...")
            response = requests.get(f"{self.api_base}/health", timeout=60)
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ API is healthy!")
                print(f"   - Triton server: {health_data['services']['triton_server']}")
                print(f"   - ArcFace model: {health_data['services']['arcface_model']}")
                print(f"   - Active jobs: {health_data['services']['active_jobs']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API server: {e}")
            print("Make sure the server is running: python api/run_server.py")
            return False
    
    def submit_config(self, config_path):
        """Submit input_config.json for processing."""
        try:
            print(f"📤 Submitting configuration: {config_path}")
            
            # Check if config file exists
            if not os.path.exists(config_path):
                print(f"❌ Config file not found: {config_path}")
                return None
            
            # Load and submit config
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            response = requests.post(
                f"{self.api_base}/process-config",
                json=config_data,
                timeout=30
            )
            
            if response.status_code == 202:
                result = response.json()
                job_id = result['job_id']
                print(f"✅ Job submitted successfully!")
                print(f"   - Job ID: {job_id}")
                print(f"   - Inputs: {len(config_data.get('inputs', []))}")
                print(f"   - Message: {result['message']}")
                return job_id
            else:
                print(f"❌ Failed to submit job: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            return None
    
    def check_job_status(self, job_id):
        """Check job processing status."""
        try:
            response = requests.get(f"{self.api_base}/job/{job_id}/status", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def monitor_job(self, job_id, check_interval=15):
        """Monitor job until completion."""
        print(f"⏳ Monitoring job: {job_id}")
        print("📊 Progress updates every 15 seconds...")
        
        start_time = time.time()
        last_percentage = -1
        
        while True:
            status = self.check_job_status(job_id)
            
            if not status:
                print("❌ Failed to get job status")
                break
            
            current_status = status['status']
            percentage = status.get('progress_percentage', 0)
            
            # Print progress update if changed
            if percentage != last_percentage:
                elapsed = int(time.time() - start_time)
                print(f"🔄 [{elapsed:03d}s] {current_status.upper()}: {percentage:3d}% - {status.get('current_stage', 'processing')}")
                last_percentage = percentage
            
            if current_status == 'completed':
                print(f"✅ Job completed successfully!")
                print(f"   - Processing time: {status.get('processing_time', 'unknown')}")
                return status
                
            elif current_status == 'failed':
                print(f"❌ Job failed!")
                error = status.get('error', {})
                print(f"   - Error: {error.get('message', 'Unknown error')}")
                return status
                
            time.sleep(check_interval)
    
    def get_results(self, job_id):
        """Get job results and download links."""
        try:
            print(f"📋 Getting results for job: {job_id}")
            
            response = requests.get(f"{self.api_base}/job/{job_id}/results", timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Results ready!")
                print(f"   - Status: {results['status']}")
                print(f"   - Completed: {results.get('completed_at', 'unknown')}")
                
                # Show available downloads
                downloads = results.get('downloads', {})
                if downloads:
                    print(f"📥 Available downloads:")
                    for output_id, files in downloads.items():
                        print(f"   📁 {output_id}:")
                        for file_type, file_info in files.items():
                            print(f"      - {file_type}: {file_info['filename']}")
                
                return results
            else:
                print(f"❌ Failed to get results: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def download_file(self, job_id, output_id, file_type, save_path):
        """Download a specific file."""
        try:
            
            url = f"{self.api_base}/job/{job_id}/download/{output_id}/{file_type}"
            response = requests.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(save_path)
                print(f"✅ Downloaded: {save_path} ({file_size:,} bytes)")
                return True
            else:
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def download_all_results(self, job_id, download_folder="api_downloads"):
        """Download all available results."""
        results = self.get_results(job_id)
        if not results:
            return False
        
        downloads = results.get('downloads', {})
        if not downloads:
            print("❌ No downloads available")
            return False
        
        os.makedirs(download_folder, exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        for output_id, files in downloads.items():
            output_folder = os.path.join(download_folder, output_id)
            
            for file_type, file_info in files.items():
                total_count += 1
                filename = file_info['filename']
                save_path = os.path.join(output_folder, filename)
                
                if self.download_file(job_id, output_id, file_type, save_path):
                    success_count += 1
        
        print(f"📊 Download summary: {success_count}/{total_count} files downloaded")
        return success_count > 0

def main():
    """Main test function."""
    print("🚀 Face Detection API Test Client")
    print("=" * 50)
    
    # Initialize client
    client = FaceDetectionAPIClient()
    
    # Step 1: Health check
    if not client.health_check():
        print("\n❌ API server not ready. Please start the server first:")
        print("   python api/run_server.py")
        return
    
    # Step 2: Find config file
    config_path = "../input_config.json"
    if not os.path.exists(config_path):
        config_path = "input_config.json"
    
    if not os.path.exists(config_path):
        print(f"\n❌ Config file not found. Please ensure input_config.json exists.")
        print("   Current directory:", os.getcwd())
        return
    
    print(f"\n📂 Using config file: {config_path}")
    
    # Step 3: Submit job
    job_id = client.submit_config(config_path)
    if not job_id:
        print("\n❌ Failed to submit job")
        return
    
    print(f"\n🎯 Job submitted: {job_id}")
    print("=" * 50)
    
    # Step 4: Monitor progress
    final_status = client.monitor_job(job_id)
    if not final_status or final_status['status'] != 'completed':
        print("\n❌ Job did not complete successfully")
        return
    
    print("\n" + "=" * 50)
    
    # Step 5: Download results
    if client.download_all_results(job_id):
        print(f"\n✅ All results downloaded successfully!")
        print(f"📁 Check the 'api_downloads' folder for your files:")
        print(f"   - face_data_export.json (same format as video_client.py)")
        print(f"   - processed_video.mp4 (with face tracking)")
    else:
        print(f"\n❌ Failed to download results")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
