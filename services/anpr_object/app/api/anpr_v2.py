import asyncio
import time
from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from triton_server.app import VisionInputItem, VisionInputOptions, VisionProcessingConfig
from triton_server.app import ANPRService
from triton_server.app import ERROR_RESPONSES
from triton_server.app import CommonVisionInputBody, RequestIdHeader, ServiceDep
from triton_server.app import MediaSourceUtils
from triton_server.app import anpr_supported_input_types, model_dump_compat

router = APIRouter(prefix="/v2/anpr_object")
logger = structlog.get_logger(__name__)


async def _process_common_anpr_input(
    *,
    service: ANPRService,
    request_id: str,
    input_item: VisionInputItem,
    processing_config: VisionProcessingConfig,
) -> Dict[str, Any]:
    options: VisionInputOptions = input_item.options
    metadata = input_item.metadata or {}
    processing_config_dict = model_dump_compat(processing_config)
    zones = [model_dump_compat(zone) for zone in (options.zones or [])]
    behavior_config = (
        model_dump_compat(options.behavior_config)
        if options.behavior_config is not None
        else None
    )
    result: Dict[str, Any] = {}
    if input_item.input_type not in anpr_supported_input_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported input_type for ANPR v2: {input_item.input_type}",
        )

    if input_item.input_type == "video_url" or input_item.input_type == "video_file":
        resolved_camera_id = MediaSourceUtils.resolve_camera_id(
            options.uri or "",
            options.camera_id,
        )
        result = await service.process_video_source(
            options.uri,
            output_dir="data/output",
            save_csv=True,
            request_id=request_id,
            camera_id=resolved_camera_id,
            lat=options.lat,
            lon=options.lon,
            pixels_per_meter=options.pixels_per_meter,
            zones=zones,
            behavior_config=behavior_config,
            processing_config=processing_config_dict,
        )

    if input_item.input_type == "image_url":
        annotated_image, detections, output_path = await service.process_image_url(
            options.uri,
            request_id=request_id,
            processing_config=processing_config_dict,
        )
        result = {
            "total_detections": len(detections),
            "detections": detections,
            "image_shape": (
                list(annotated_image.shape) if annotated_image is not None else None
            ),
            "output_path": output_path,
        }

    if input_item.input_type == "image_file":
        annotated_image, detections, output_path = await service.process_image_source(
            options.uri,
            request_id=request_id,
            processing_config=processing_config_dict,
        )
        result = {
            "total_detections": len(detections),
            "detections": detections,
            "image_shape": (
                list(annotated_image.shape) if annotated_image is not None else None
            ),
            "output_path": output_path,
        }

    result.update(
        {
            "input_id": input_item.id,
            "input_type": input_item.input_type,
            "source_path": options.uri,
            "metadata": metadata,
        }
    )

    return result


@router.post("/process", responses=ERROR_RESPONSES)
async def process_anpr(
    request_id: RequestIdHeader,
    payload: CommonVisionInputBody,
    service: ServiceDep,
) -> JSONResponse:
    logger.info(f"Processing ANPR request with payload: {payload}")
    api_start = time.perf_counter()
    try:
        results = []
        for input_item in payload.inputs:
            results.append(
                await _process_common_anpr_input(
                    service=service,
                    request_id=request_id,
                    input_item=input_item,
                    processing_config=payload.processing_config,
                )
            )

        response = {
            "request_id": request_id,
            "service": "anpr_object",
            "version": "v2",
            "endpoint": "process",
            "total_inputs": len(payload.inputs),
            "processed_inputs": len(results),
            "processing_config": payload.processing_config.model_dump(exclude_none=True),
            "results": results,
        }
        return JSONResponse(content=response, status_code=200)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("api.v2.process_anpr.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info(
            "api.v2.process_anpr",
            total_duration=int((time.perf_counter() - api_start) * 1000),
        )
