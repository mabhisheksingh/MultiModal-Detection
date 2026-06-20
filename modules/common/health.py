from datetime import UTC, datetime

from fastapi import APIRouter, Header

health_router = APIRouter()


@health_router.get("/health")
def health(request_id: str | None = Header(None)):
    return {
        "request_id": request_id,
        "status": "ok",
        "time": datetime.now(UTC).isoformat(),
    }
