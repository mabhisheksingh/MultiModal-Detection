from functools import lru_cache
from typing import Annotated

import structlog
from fastapi import Body, Depends, File, Header, Query, UploadFile

from triton_server.app import (
    ANPRService,
    APIConstants,
    BehavioralPatternService,
    CommonVisionInputRequest,
    GlobalTrackingService,
    PaddleOCREngine,
    SpatiotemporalCorrelationService,
    TritonClient,
    settings,
)

logger = structlog.get_logger(__name__)

RequestIdHeader = Annotated[
    str,
    Header(alias=APIConstants.REQUEST_ID_HEADER_PARAM),
]

# 1. Define a global variable to hold the single instance
_ocr_engine_instance = None


def get_paddle_ocr_engine() -> PaddleOCREngine:
    global _ocr_engine_instance

    # 2. Only initialize it if it doesn't exist yet
    if _ocr_engine_instance is None:
        logger.info("Bootstrapping PaddleOCR Singleton instance...")
        _ocr_engine_instance = PaddleOCREngine()
    # 3. Return the cached instance for all requests
    return _ocr_engine_instance


@lru_cache
def get_vehicle_triton_client() -> TritonClient:
    return TritonClient(server_url=settings.triton_server_url, model_name=settings.vehicle_model_name)


@lru_cache
def get_plate_triton_client() -> TritonClient:
    return TritonClient(server_url=settings.triton_server_url, model_name=settings.plate_model_name)


@lru_cache
def get_object_triton_client() -> TritonClient:
    return TritonClient(server_url=settings.triton_server_url, model_name=settings.object_model_name)


@lru_cache
def get_global_tracking_service() -> GlobalTrackingService:
    return GlobalTrackingService()


@lru_cache
def get_spatial_correlation_service() -> SpatiotemporalCorrelationService:
    return SpatiotemporalCorrelationService()


@lru_cache
def get_behavioral_pattern_service() -> BehavioralPatternService:
    return BehavioralPatternService()


@lru_cache
def get_anpr_service() -> ANPRService:
    return ANPRService(
        ocr_service=get_paddle_ocr_engine(),
        vehicle_triton_client=get_vehicle_triton_client(),
        plate_triton_client=get_plate_triton_client(),
        object_triton_client=get_object_triton_client(),
        global_tracking_service=get_global_tracking_service(),
        spatial_correlation_service=get_spatial_correlation_service(),
        behavioral_pattern_service=get_behavioral_pattern_service(),
    )


ServiceDep = Annotated[ANPRService, Depends(get_anpr_service)]
CommonVisionInputBody = Annotated[CommonVisionInputRequest, Body(...)]
LegacyFiles = Annotated[list[UploadFile] | None, File(description="List of files to process")]
InputPathQuery = Annotated[
    str | None,
    Query(description="Optional server-side folder containing images/videos to process"),
]
VideoUrlQuery = Annotated[
    str | None,
    Query(description="Optional remote video URL (http/https) to process directly without copying locally"),
]
