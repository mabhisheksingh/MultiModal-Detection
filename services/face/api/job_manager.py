"""
Job Manager for Face Detection API
=================================
Manages background job processing, status tracking, and result storage.
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from models import JobStatus

class JobManager:
    """Manages background processing jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def create_job(self, job_id: str, config_data: Dict) -> Dict:
        """Create a new processing job."""
        with self.lock:
            job = {
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "config": config_data,
                "progress_percentage": 0,
                "current_stage": "initializing",
                "results": {},
                "error": None,
                "processing_time": None
            }
            self.jobs[job_id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = status
                
                if status == JobStatus.PROCESSING and not self.jobs[job_id]["started_at"]:
                    self.jobs[job_id]["started_at"] = datetime.now().isoformat()
    
    def update_job_progress(self, job_id: str, percentage: int, stage: str = None) -> None:
        """Update job progress."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["progress_percentage"] = percentage
                if stage:
                    self.jobs[job_id]["current_stage"] = stage
    
    def complete_job(self, job_id: str, results: Dict) -> None:
        """Mark job as completed with results."""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job["status"] = JobStatus.COMPLETED
                job["completed_at"] = datetime.now().isoformat()
                job["progress_percentage"] = 100
                job["current_stage"] = "completed"
                job["results"] = results
                
                # Calculate processing time
                if job["started_at"]:
                    start_time = datetime.fromisoformat(job["started_at"])
                    end_time = datetime.fromisoformat(job["completed_at"])
                    duration = end_time - start_time
                    job["processing_time"] = str(duration).split('.')[0]  # Remove microseconds
    
    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed with error."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = JobStatus.FAILED
                self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
                self.jobs[job_id]["error"] = {
                    "message": error_message,
                    "timestamp": datetime.now().isoformat()
                }
    
    def get_active_jobs(self) -> List[str]:
        """Get list of active job IDs."""
        with self.lock:
            return [
                job_id for job_id, job in self.jobs.items()
                if job["status"] in [JobStatus.PENDING, JobStatus.PROCESSING]
            ]
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than specified hours. Returns count of removed jobs."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        with self.lock:
            old_jobs = []
            for job_id, job in self.jobs.items():
                job_time = datetime.fromisoformat(job["created_at"]).timestamp()
                if job_time < cutoff_time and job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    old_jobs.append(job_id)
            
            for job_id in old_jobs:
                del self.jobs[job_id]
            
            return len(old_jobs)
    
    def get_job_statistics(self) -> Dict:
        """Get statistics about jobs."""
        with self.lock:
            total_jobs = len(self.jobs)
            status_counts = {}
            
            for job in self.jobs.values():
                status = job["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_jobs": total_jobs,
                "status_breakdown": status_counts,
                "active_jobs": len(self.get_active_jobs())
            }
