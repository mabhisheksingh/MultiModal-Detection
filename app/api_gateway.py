import logging

from fastapi import FastAPI

from modules.anpr.api.anpr import anpr_router
from modules.common.health import health_router

logger = logging.getLogger(__name__)


class Tags:
    HEALTH_ROUTER: str = "Health"
    ANPR_ROUTER: str = "ANPR"


def include_routers(app: FastAPI):
    app.include_router(health_router, tags=[Tags.HEALTH_ROUTER])
    app.include_router(anpr_router, tags=[Tags.ANPR_ROUTER], prefix="/anpr")
