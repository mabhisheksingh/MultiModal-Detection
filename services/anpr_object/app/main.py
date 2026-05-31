import asyncio
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response
from structlog.contextvars import bind_contextvars, clear_contextvars

from triton_server.app import router as anpr_v2_router
from triton_server.app import router as health_router
from triton_server.app import settings
from triton_server.app import setup_logger
from triton_server.app import init_db
from triton_server.app import APIConstants

tags_metadata = [
    {"name": "Health", "description": "Service health checks"},
    {"name": "ANPR", "description": "Automatic Number Plate Recognition endpoints"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API for Automatic Number Plate Recognition using Triton Inference Server and Ultralytics",
    openapi_tags=tags_metadata,
    contact={
        "name": "ANPR Team",
        "url": "https://clear-trail.com",
        "email": "support@clear-trail.com",
    },
    license_info={"name": "clear-trail"},
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    incoming_request_id = request.headers.get(APIConstants.REQUEST_ID_HEADER)
    request_id = incoming_request_id or str(uuid.uuid4())
    if not incoming_request_id:
        request.scope["headers"] = list(request.scope.get("headers", [])) + [
            (
                APIConstants.REQUEST_ID_HEADER_PARAM.encode("latin-1"),
                request_id.encode("latin-1"),
            )
        ]
    bind_contextvars(request_id=request_id)
    try:
        response: Response = await call_next(request)
    except asyncio.CancelledError:
        raise
    finally:
        clear_contextvars()
    response.headers[APIConstants.REQUEST_ID_HEADER] = request_id
    return response


app.include_router(health_router, tags=["Health"])
app.include_router(anpr_v2_router, tags=["ANPR-v2"])


def start_server():
    """Entry point for the anpr_object-api command."""
    import uvicorn

    setup_logger()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9003,
        # reload=True
    )


def start_gunicorn():
    """Entry point for the anpr_object-prod command."""
    from gunicorn.app.wsgiapp import run
    import sys

    setup_logger()
    sys.argv = [
        "gunicorn",
        "app.main:app",
        "-k",
        "uvicorn.workers.UvicornWorker",
        "--bind",
        "0.0.0.0:9003",
        "--workers",
        "3",
        "--timeout",
        "120",
    ]
    run()


if __name__ == "__main__":
    start_server()
