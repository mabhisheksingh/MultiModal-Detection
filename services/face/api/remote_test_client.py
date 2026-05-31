"""
Remote Face Detection API Test Client
====================================
Python client to test the REST API on a remote GPU server.
"""

import requests
import json
import time
import os
import sys
from datetime import datetime

class RemoteFaceDetectionAPIClient:
    """Client for Remote Face Detection & ArcFace Embedding API."""
    
    def __init__(self, server_ip="192.168.x.89", server_port=5000):
        """
        Initialize client for remote server.
        
        Args:
            server_ip: IP address of machine 89 (replace x with actual IP)
            server_port: API server port (default: 5000)
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{server_ip}:{server_port}"
        self.api_base = f"{self.base_url}/api/v1"
        
        print(f"🌐 Connecting to remote GPU server: {self.base_url}")
    
    def health_check(self):
        """Check remote API health status."""
        try:
            print("🔍 Checking remote API health...")
            response = requests.get(f"{self.api_base}/health", timeout=30)
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Remote API is healthy!")
                print(f"   - Server: {self.server_ip}:{self.server_port}")
                print(f"   - Triton server: {health_data['services']['triton_server']}")
                print(f"   - ArcFace model: {health_data['services']['arcface_model']}")
                print(f"   - Active jobs: {health_data['services']['active_jobs']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to remote API server: {e}")
            print(f"Make sure the server is running on {self.server_ip}:5000")
            print("Check network connectivity and firewall settings")
            return False
    
    def submit_config(self, config_path):
        """Submit input_config.json to remote server for processing."""
        try:
            print(f"📤 Submitting configuration to remote server: {config_path}")
            
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
                timeout=60  # Longer timeout for remote requests
            )
            
            if response.status_code == 202:
                result = response.json()
                job_id = result['job_id']
                print(f"✅ Job submitted to GPU server successfully!")
                print(f"   - Job ID: {job_id}")
                print(f"   - Server: {self.server_ip}")
                print(f"   - Inputs: {len(config_data.get('inputs', []))}")
                return job_id
            else:
                print(f"❌ Failed to submit job: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Remote request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            return None
    
    def check_job_status(self, job_id):
        """Check job processing status on remote server."""
        try:
            response = requests.get(f"{self.api_base}/job/{job_id}/status", timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Remote request failed: {e}")
            return None
    
    def monitor_job(self, job_id, check_interval=20):
        """Monitor remote job until completion."""
        print(f"⏳ Monitoring remote GPU job: {job_id}")
        print(f"📊 Progress updates every {check_interval} seconds...")
        print(f"🖥️  GPU Server: {self.server_ip}")
        
        start_time = time.time()
        last_percentage = -1
        
        while True:
            status = self.check_job_status(job_id)
            
            if not status:
                print("❌ Failed to get job status from remote server")
                break
            
            current_status = status['status']
            percentage = status.get('progress_percentage', 0)
            
            # Print progress update if changed
            if percentage != last_percentage:
                elapsed = int(time.time() - start_time)
                print(f"🔄 [{elapsed:03d}s] GPU {current_status.upper()}: {percentage:3d}% - {status.get('current_stage', 'processing')}")
                last_percentage = percentage
            
            if current_status == 'completed':
                print(f"✅ Remote GPU job completed successfully!")
                print(f"   - Processing time: {status.get('processing_time', 'unknown')}")
                print(f"   - GPU Server: {self.server_ip}")
                return status
                
            elif current_status == 'failed':
                print(f"❌ Remote GPU job failed!")
                error = status.get('error', {})
                print(f"   - Error: {error.get('message', 'Unknown error')}")
                return status
                
            time.sleep(check_interval)
    
    def download_file(self, job_id, output_id, file_type, save_path):
        """Download a specific file from remote server."""
        try:
            print(f"⬇️  Downloading {file_type} from GPU server...")
            
            url = f"{self.api_base}/job/{job_id}/download/{output_id}/{file_type}"
            response = requests.get(url, stream=True, timeout=120)  # Longer timeout for downloads
            
            if response.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(save_path)
                print(f"✅ Downloaded from GPU server: {save_path} ({file_size:,} bytes)")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Download from remote server failed: {e}")
            return False
    
    def download_all_results(self, job_id, download_folder="remote_gpu_downloads"):
        """Download all results from remote GPU server."""
        try:
            print(f"📋 Getting results from GPU server...")
            response = requests.get(f"{self.api_base}/job/{job_id}/results", timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Failed to get results: {response.status_code}")
                return False
            
            results = response.json()
            downloads = results.get('downloads', {})
            
            if not downloads:
                print("❌ No downloads available")
                return False
            
            print(f"📥 Downloading all GPU results to: {download_folder}")
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
            
            print(f"📊 GPU Download summary: {success_count}/{total_count} files downloaded")
            return success_count > 0
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get results from remote server: {e}")
            return False

def main():
    """Main test function for remote GPU server."""
    print("🚀 Remote GPU Face Detection API Test Client")
    print("=" * 60)
    
    # IMPORTANT: Replace this with machine 89's actual IP address
    SERVER_IP = "192.168.x.89"  # ← UPDATE THIS!
    
    print("⚠️  IMPORTANT: Update SERVER_IP in this script!")
    print(f"   Current setting: {SERVER_IP}")
    
    if "x" in SERVER_IP:
        print("❌ Please update SERVER_IP with machine 89's actual IP address")
        return
    
    # Initialize remote client
    client = RemoteFaceDetectionAPIClient(server_ip=SERVER_IP)
    
    # Step 1: Health check
    if not client.health_check():
        print("\n❌ Remote GPU server not ready. Please check:")
        print("   1. Machine 89 is running the API server")
        print("   2. Network connectivity")
        print("   3. Firewall settings")
        return
    
    # Step 2: Find config file
    config_path = "../input_config.json"
    if not os.path.exists(config_path):
        config_path = "input_config.json"
    
    if not os.path.exists(config_path):
        print(f"\n❌ Config file not found. Please ensure input_config.json exists.")
        return
    
    print(f"\n📂 Using config file: {config_path}")
    
    # Step 3: Submit job to GPU server
    job_id = client.submit_config(config_path)
    if not job_id:
        print("\n❌ Failed to submit job to GPU server")
        return
    
    print(f"\n🎯 GPU Job submitted: {job_id}")
    print("=" * 60)
    
    # Step 4: Monitor GPU processing
    final_status = client.monitor_job(job_id, check_interval=20)  # Check every 20s for remote
    if not final_status or final_status['status'] != 'completed':
        print("\n❌ GPU job did not complete successfully")
        return
    
    print("\n" + "=" * 60)
    
    # Step 5: Download results from GPU server
    if client.download_all_results(job_id):
        print(f"\n✅ All results downloaded from GPU server!")
        print(f"📁 Check the 'remote_gpu_downloads' folder for your files:")
        print(f"   - face_data_export.json (ArcFace embeddings from GPU)")
        print(f"   - processed_video.mp4 (GPU-accelerated processing)")
    else:
        print(f"\n❌ Failed to download results from GPU server")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
