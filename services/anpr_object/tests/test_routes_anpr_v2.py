from unittest.mock import AsyncMock, Mock, patch

import json
import asyncio
import numpy as np
import pytest
from fastapi import HTTPException

from triton_server.app import anpr_v2 as anpr_v2_routes
from triton_server.app import (
    BehaviorConfig,
    CommonVisionInputRequest,
    VisionInputItem,
    VisionInputOptions,
    VisionProcessingConfig,
    Zone,
)


@pytest.mark.anyio
async def test_process_common_anpr_input_video_branch_uses_resolved_camera_id():
    service = Mock()
    service.process_video_source = AsyncMock(return_value={"video": True})
    processing_config = VisionProcessingConfig(
        confidence_threshold=0.25,
        ocr_confidence_threshold=0.5,
        ocr_match_confidence=0.88,
        global_id_match_score=0.76,
        frames_per_second=10,
        is_ocr_enabled=True,
        platform="anpr_object",
    )
    input_item = VisionInputItem(
        id="video_1",
        input_type="video_url",
        options=VisionInputOptions(
            uri="https://example.com/video.mp4",
            camera_id="cam_manual",
            lat=10.5,
            lon=20.5,
            pixels_per_meter=12.0,
            zones=[
                Zone(
                    zone_id="z1",
                    zone_type="restricted",
                    coordinates=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
                )
            ],
            behavior_config=BehaviorConfig(repeat_visit_threshold=5),
        ),
        metadata={"source": "front_gate"},
    )

    with patch.object(
        anpr_v2_routes.MediaSourceUtils,
        "resolve_camera_id",
        return_value="cam_resolved",
    ):
        result = await anpr_v2_routes._process_common_anpr_input(
            service=service,
            request_id="req_v2_video",
            input_item=input_item,
            processing_config=processing_config,
        )

    assert result["video"] is True
    assert result["input_id"] == "video_1"
    assert result["input_type"] == "video_url"
    assert result["source_path"] == "https://example.com/video.mp4"
    assert result["metadata"] == {"source": "front_gate"}

    kwargs = service.process_video_source.await_args.kwargs
    assert kwargs["camera_id"] == "cam_resolved"
    assert kwargs["lat"] == 10.5
    assert kwargs["lon"] == 20.5
    assert kwargs["pixels_per_meter"] == 12.0
    assert kwargs["zones"] == [
        {
            "zone_id": "z1",
            "zone_type": "restricted",
            "coordinates": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
        }
    ]
    assert kwargs["behavior_config"] == {
        "repeat_visit_threshold": 5,
        "linger_threshold_ms": 30000,
        "sensitive_zone_types": ["sensitive", "restricted"],
        "min_behavior_score": 0.6,
    }
    assert kwargs["processing_config"] == {
        "confidence_threshold": 0.25,
        "ocr_confidence_threshold": 0.5,
        "ocr_match_confidence": 0.88,
        "global_id_match_score": 0.76,
        "frames_per_second": 10,
        "ocr_plate_text_mode": "balanced",
        "extra": {},
        "is_ocr_enabled": True,
        "platform": "anpr_object",
    }


@pytest.mark.anyio
async def test_process_common_anpr_input_image_branches_build_summary_payloads():
    service = Mock()
    annotated = np.zeros((8, 6, 3), dtype=np.uint8)
    service.process_image_url = AsyncMock(
        return_value=(annotated, [{"id": 1}], "/tmp/annotated_url.jpg")
    )
    service.process_image_source = AsyncMock(
        return_value=(annotated, [{"id": 2}], "/tmp/annotated_file.jpg")
    )
    processing_config = VisionProcessingConfig(confidence_threshold=0.35, is_ocr_enabled=True, platform="anpr_object")

    image_url_item = VisionInputItem(
        id="img_url_1",
        input_type="image_url",
        options=VisionInputOptions(uri="https://example.com/frame.jpg"),
    )
    image_file_item = VisionInputItem(
        id="img_file_1",
        input_type="image_file",
        options=VisionInputOptions(uri="/tmp/frame.jpg"),
        metadata={"batch": "local"},
    )

    url_result = await anpr_v2_routes._process_common_anpr_input(
        service=service,
        request_id="req_img_url",
        input_item=image_url_item,
        processing_config=processing_config,
    )
    file_result = await anpr_v2_routes._process_common_anpr_input(
        service=service,
        request_id="req_img_file",
        input_item=image_file_item,
        processing_config=processing_config,
    )

    assert url_result["total_detections"] == 1
    assert url_result["detections"] == [{"id": 1}]
    assert url_result["image_shape"] == [8, 6, 3]
    assert url_result["output_path"] == "/tmp/annotated_url.jpg"

    assert file_result["total_detections"] == 1
    assert file_result["detections"] == [{"id": 2}]
    assert file_result["image_shape"] == [8, 6, 3]
    assert file_result["output_path"] == "/tmp/annotated_file.jpg"
    assert file_result["metadata"] == {"batch": "local"}
    assert (
        service.process_image_url.await_args.kwargs["processing_config"][
            "confidence_threshold"
        ]
        == 0.35
    )
    assert (
        service.process_image_source.await_args.kwargs["processing_config"][
            "confidence_threshold"
        ]
        == 0.35
    )


@pytest.mark.anyio
async def test_process_common_anpr_input_rejects_unsupported_type():
    service = Mock()
    input_item = VisionInputItem(
        id="bad_1",
        input_type="manifest",
        options=VisionInputOptions(uri="manifest.json"),
    )

    with pytest.raises(HTTPException) as exc:
        await anpr_v2_routes._process_common_anpr_input(
            service=service,
            request_id="req_bad",
            input_item=input_item,
            processing_config=VisionProcessingConfig(is_ocr_enabled=True, platform="anpr_object"),
        )

    assert exc.value.status_code == 400
    assert "Unsupported input_type" in exc.value.detail


@pytest.mark.anyio
async def test_process_anpr_v2_aggregates_results_and_processing_config():
    payload = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(
            confidence_threshold=0.3,
            platform="anpr_object",
            is_ocr_enabled=True,
        ),
        inputs=[
            VisionInputItem(
                id="input_1",
                input_type="image_url",
                options=VisionInputOptions(uri="https://example.com/1.jpg"),
            ),
            VisionInputItem(
                id="input_2",
                input_type="image_file",
                options=VisionInputOptions(uri="/tmp/2.jpg"),
            ),
        ],
    )
    service = Mock()

    with patch.object(
        anpr_v2_routes,
        "_process_common_anpr_input",
        new=AsyncMock(side_effect=[{"input_id": "input_1"}, {"input_id": "input_2"}]),
    ):
        response = await anpr_v2_routes.process_anpr(
            request_id="req_v2",
            payload=payload,
            service=service,
        )

    body = json.loads(response.body.decode("utf-8"))
    assert body["request_id"] == "req_v2"
    assert body["service"] == "anpr_object"
    assert body["version"] == "v2"
    assert body["processed_inputs"] == 2
    assert body["processing_config"]["confidence_threshold"] == 0.3


@pytest.mark.anyio
async def test_process_anpr_v2_wraps_unexpected_errors():
    payload = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(is_ocr_enabled=True, platform="anpr_object"),
        inputs=[
            VisionInputItem(
                id="input_1",
                input_type="image_url",
                options=VisionInputOptions(uri="https://example.com/1.jpg"),
            )
        ]
    )
    service = Mock()

    with patch.object(
        anpr_v2_routes,
        "_process_common_anpr_input",
        new=AsyncMock(side_effect=RuntimeError("route failed")),
    ):
        with pytest.raises(HTTPException) as exc:
            await anpr_v2_routes.process_anpr(
                request_id="req_v2_fail",
                payload=payload,
                service=service,
            )

    assert exc.value.status_code == 500
    assert exc.value.detail == "route failed"


@pytest.mark.anyio
async def test_process_anpr_v2_reraises_http_exception():
    payload = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(is_ocr_enabled=True, platform="anpr_object"),
        inputs=[
            VisionInputItem(
                id="input_1",
                input_type="image_url",
                options=VisionInputOptions(uri="https://example.com/1.jpg"),
            )
        ]
    )
    service = Mock()

    with patch.object(
        anpr_v2_routes,
        "_process_common_anpr_input",
        new=AsyncMock(side_effect=HTTPException(status_code=400, detail="bad input")),
    ):
        with pytest.raises(HTTPException) as exc:
            await anpr_v2_routes.process_anpr(
                request_id="req_v2_http",
                payload=payload,
                service=service,
            )

    assert exc.value.status_code == 400
    assert exc.value.detail == "bad input"


@pytest.mark.anyio
async def test_process_anpr_v2_reraises_cancelled_error():
    payload = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(is_ocr_enabled=True, platform="anpr_object"),
        inputs=[
            VisionInputItem(
                id="input_1",
                input_type="image_url",
                options=VisionInputOptions(uri="https://example.com/1.jpg"),
            )
        ]
    )
    service = Mock()

    with patch.object(
        anpr_v2_routes,
        "_process_common_anpr_input",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await anpr_v2_routes.process_anpr(
                request_id="req_v2_cancel",
                payload=payload,
                service=service,
            )
