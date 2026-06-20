import asyncio
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.api_gateway import include_routers
from modules.common.logging import configure_logging, setup_logger

# Configure logging
configure_logging()
setup_logger()
logger = __import__("logging").getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

app = FastAPI(
    title="MultiModal Detection API",
    description="API for ANPR, Drone, Face detection and other multimodal services",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
include_routers(app)


# Request ID middleware
@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    incoming_request_id = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming_request_id or str(uuid.uuid4())
    if not incoming_request_id:
        request.scope["headers"] = list(request.scope.get("headers", [])) + [
            (b"x-request-id", request_id.encode("latin-1"))
        ]
    bind_contextvars(request_id=request_id)
    try:
        response: Response = await call_next(request)
    except asyncio.CancelledError:
        raise
    finally:
        clear_contextvars()
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def start_server():
    """Entry point for the main API gateway"""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_server()
