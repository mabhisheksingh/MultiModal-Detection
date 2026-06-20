import pytest
from pydantic import ValidationError

from triton_server.app import (
    CommonVisionInputRequest,
    SourceMetadata,
    VisionInputItem,
    VisionInputOptions,
    VisionProcessingConfig,
)


def test_vision_input_options_accepts_uri_content_or_frames():
    uri_options = VisionInputOptions(uri="/tmp/input.mp4")
    assert uri_options.uri == "/tmp/input.mp4"
    assert uri_options.frames == []

    base64_options = VisionInputOptions(content_base64="ZmFrZQ==")
    assert base64_options.content_base64 == "ZmFrZQ=="

    frame_options = VisionInputOptions(frames=[{"frame_id": 1, "data": "abc"}])
    assert frame_options.frames == [{"frame_id": 1, "data": "abc"}]


def test_vision_input_options_rejects_missing_payload_sources():
    with pytest.raises(ValidationError) as exc:
        VisionInputOptions()

    assert "At least one of 'uri', 'content_base64', or 'frames' must be provided in options." in str(exc.value)


def test_vision_input_item_and_common_request_preserve_processing_config():
    request = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(
            confidence_threshold=0.4,
            ocr_confidence_threshold=0.6,
            ocr_match_confidence=0.88,
            global_id_match_score=0.76,
            frames_per_second=12,
            platform="anpr_object",
            is_ocr_enabled=True,
            extra={"batch_size": 4},
        ),
        inputs=[
            VisionInputItem(
                id="input_1",
                input_type="video_url",
                options=VisionInputOptions(
                    uri="https://example.com/video.mp4",
                    camera_id="cam_1",
                ),
                metadata={"site": "gate_1"},
            )
        ],
    )

    assert request.processing_config.confidence_threshold == 0.4
    assert request.processing_config.ocr_confidence_threshold == 0.6
    assert request.processing_config.ocr_match_confidence == 0.88
    assert request.processing_config.global_id_match_score == 0.76
    assert request.processing_config.frames_per_second == 12
    assert request.processing_config.platform == "anpr_object"
    assert request.processing_config.extra == {"batch_size": 4}
    assert request.inputs[0].options.camera_id == "cam_1"
    assert request.inputs[0].metadata == {"site": "gate_1"}


def test_vision_processing_config_defaults_for_anpr():
    config = VisionProcessingConfig(is_ocr_enabled=False)

    assert config.confidence_threshold == 0.2
    assert config.ocr_confidence_threshold == 0.5
    assert config.ocr_match_confidence == 0.85
    assert config.global_id_match_score == 0.7
    assert config.frames_per_second == 10


def test_common_request_requires_at_least_one_input():
    with pytest.raises(ValidationError):
        CommonVisionInputRequest(inputs=[])


def test_source_metadata_rejects_empty_string_numeric_fields():
    with pytest.raises(ValidationError) as lat_exc:
        SourceMetadata(url="https://example.com/video.mp4", lat="")
    assert "lat must be a valid number or null" in str(lat_exc.value)

    with pytest.raises(ValidationError) as lon_exc:
        SourceMetadata(url="https://example.com/video.mp4", lon="")
    assert "lon must be a valid number or null" in str(lon_exc.value)

    with pytest.raises(ValidationError) as ppm_exc:
        SourceMetadata(url="https://example.com/video.mp4", pixels_per_meter="")
    assert "pixels_per_meter must be a valid number or null" in str(ppm_exc.value)
