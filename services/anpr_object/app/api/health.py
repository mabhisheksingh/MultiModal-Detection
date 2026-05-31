from datetime import datetime, timezone

from fastapi import APIRouter

from triton_server.app import settings
from triton_server.app import RequestIdHeader

router = APIRouter()


@router.get("/health")
def health(request_id: RequestIdHeader):
    return {
        "request_id": request_id,
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "triton_server": settings.triton_server_url,
        "vehicle_model": settings.vehicle_model_name,
        "plate_model": settings.plate_model_name,
        "time": datetime.now(timezone.utc).isoformat(),
    }
