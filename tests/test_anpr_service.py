from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from triton_server.app import (
    BehavioralPatternService,
    CommonVisionInputRequest,
    SourceMetadata,
    VisionInputItem,
    VisionInputOptions,
    VisionProcessingConfig,
)


@pytest.fixture
def mock_repo():
    repo = Mock()
    repo.record_visit.return_value = None
    repo.record_behavioral_event.return_value = None
    return repo


def test_source_metadata_defaults_and_camera_id_generation_behavior():
    source = SourceMetadata(url="http://example.com/video.mp4")
    generated = SourceMetadata(url="http://example.com/video.mp4", camera_id="")

    assert source.camera_id is None
    assert generated.camera_id.startswith("cam_")
    assert source.pixels_per_meter == 25.0
    assert source.lat is None
    assert source.lon is None


def test_source_metadata_rejects_empty_lat_lon_and_pixels_per_meter():
    with pytest.raises(ValidationError) as lat_error:
        SourceMetadata(url="http://example.com/video.mp4", lat="")
    assert "lat must be a valid number or null" in str(lat_error.value)

    with pytest.raises(ValidationError) as lon_error:
        SourceMetadata(url="http://example.com/video.mp4", lon="")
    assert "lon must be a valid number or null" in str(lon_error.value)

    with pytest.raises(ValidationError) as ppm_error:
        SourceMetadata(url="http://example.com/video.mp4", pixels_per_meter="")
    assert "pixels_per_meter must be a valid number or null" in str(ppm_error.value)


def test_common_vision_input_request_requires_at_least_one_input():
    with pytest.raises(ValidationError):
        CommonVisionInputRequest(inputs=[])


def test_vision_input_options_requires_one_supported_payload_source():
    with pytest.raises(ValidationError):
        VisionInputOptions()


def test_common_vision_input_request_accepts_valid_image_input():
    payload = CommonVisionInputRequest(
        processing_config=VisionProcessingConfig(is_ocr_enabled=False),
        inputs=[
            VisionInputItem(
                id="img_1",
                input_type="image_url",
                options=VisionInputOptions(uri="https://example.com/frame.jpg"),
            )
        ],
    )

    assert len(payload.inputs) == 1
    assert payload.inputs[0].id == "img_1"


def test_behavioral_service_returns_empty_state_without_behavior_config(mock_repo):
    service = BehavioralPatternService(repository=mock_repo)

    detections = [{"track_id": "1", "frame_id": 1, "ts_ms": 1000}]
    enriched = service.enrich_detections_with_behavior_state(detections)

    assert len(enriched) == 1
    assert enriched[0]["behavior_state"]["behavior_label"] == "normal"
    mock_repo.record_visit.assert_not_called()
    mock_repo.record_behavioral_event.assert_not_called()


def test_behavioral_service_marks_repeat_linger_and_sensitive_zone(mock_repo):
    service = BehavioralPatternService(repository=mock_repo)

    detections = [
        {
            "global_id": "gid_1",
            "track_id": "1",
            "frame_id": 1,
            "ts_ms": 1000,
            "spatial_state": {
                "active_zone_id": "zone_a",
                "active_zone_type": "restricted",
                "is_inside_zone": True,
            },
        },
        {
            "global_id": "gid_1",
            "track_id": "1",
            "frame_id": 2,
            "ts_ms": 2000,
            "spatial_state": {
                "active_zone_id": "zone_a",
                "active_zone_type": "restricted",
                "is_inside_zone": True,
            },
        },
        {
            "global_id": "gid_1",
            "track_id": "1",
            "frame_id": 3,
            "ts_ms": 7000,
            "spatial_state": {
                "active_zone_id": "zone_a",
                "active_zone_type": "restricted",
                "is_inside_zone": True,
            },
        },
        {
            "global_id": "gid_1",
            "track_id": "1",
            "frame_id": 4,
            "ts_ms": 9000,
            "spatial_state": {
                "active_zone_id": "zone_a",
                "active_zone_type": "restricted",
                "is_inside_zone": True,
            },
        },
    ]

    config = {
        "repeat_visit_threshold": 1,
        "linger_threshold_ms": 1000,
        "sensitive_zone_types": ["restricted"],
        "min_behavior_score": 0.6,
        "reappearance_gap_ms": 3000,
    }

    enriched = service.enrich_detections_with_behavior_state(
        detections,
        behavior_config=config,
        camera_id="cam_1",
        request_id="req_1",
    )

    first_state = enriched[0]["behavior_state"]
    second_episode_start_state = enriched[2]["behavior_state"]

    assert first_state["visit_count"] == 1
    assert first_state["is_repeat_visit"] is False
    assert first_state["is_lingering"] is False
    assert first_state["is_sensitive_zone_presence"] is True
    assert first_state["behavior_label"] == "sensitive_zone_presence"

    assert second_episode_start_state["visit_count"] == 2
    assert second_episode_start_state["is_repeat_visit"] is True
    assert second_episode_start_state["is_lingering"] is True
    assert second_episode_start_state["is_sensitive_zone_presence"] is True
    assert second_episode_start_state["behavior_label"] == "repeated_presence"

    mock_repo.record_visit.assert_called_once()
    assert mock_repo.record_behavioral_event.call_count == 4


def test_behavioral_service_with_missing_identity_returns_normal(mock_repo):
    service = BehavioralPatternService(repository=mock_repo)

    detections = [
        {
            "track_id": "",
            "global_id": "",
            "frame_id": 1,
            "ts_ms": 1000,
            "spatial_state": {"is_inside_zone": False, "active_zone_type": ""},
        }
    ]

    enriched = service.enrich_detections_with_behavior_state(
        detections,
        behavior_config={"reappearance_gap_ms": 3000},
        camera_id="cam_1",
        request_id="req_1",
    )

    assert enriched[0]["behavior_state"]["behavior_label"] == "normal"
    assert enriched[0]["behavior_state"]["behavior_score"] == 0.0


def test_behavioral_service_min_behavior_score_can_force_normal(mock_repo):
    service = BehavioralPatternService(repository=mock_repo)

    detections = [
        {
            "global_id": "gid_min",
            "track_id": "1",
            "frame_id": 1,
            "ts_ms": 1000,
            "spatial_state": {
                "active_zone_id": "",
                "active_zone_type": "",
                "is_inside_zone": False,
            },
        },
        {
            "global_id": "gid_min",
            "track_id": "1",
            "frame_id": 2,
            "ts_ms": 5000,
            "spatial_state": {
                "active_zone_id": "",
                "active_zone_type": "",
                "is_inside_zone": False,
            },
        },
    ]

    config = {
        "repeat_visit_threshold": 1,
        "linger_threshold_ms": 100000,
        "sensitive_zone_types": ["restricted"],
        "min_behavior_score": 0.9,
        "reappearance_gap_ms": 3000,
    }

    enriched = service.enrich_detections_with_behavior_state(
        detections,
        behavior_config=config,
        camera_id="cam_1",
        request_id="req_1",
    )

    assert enriched[1]["behavior_state"]["visit_count"] == 2
    assert enriched[1]["behavior_state"]["is_repeat_visit"] is True
    assert enriched[1]["behavior_state"]["behavior_label"] == "normal"
    assert enriched[1]["behavior_state"]["behavior_score"] == 0.0


def test_behavioral_service_linger_branch_and_repo_exceptions_are_swallowed():
    mock_repo = Mock()
    mock_repo.record_visit.side_effect = RuntimeError("db visit error")
    mock_repo.record_behavioral_event.side_effect = RuntimeError("db event error")
    service = BehavioralPatternService(repository=mock_repo)

    detections = [
        {
            "global_id": "gid_linger",
            "track_id": "1",
            "frame_id": 1,
            "ts_ms": 1000,
            "spatial_state": {
                "active_zone_id": "",
                "active_zone_type": "",
                "is_inside_zone": False,
            },
        },
        {
            "global_id": "gid_linger",
            "track_id": "1",
            "frame_id": 2,
            "ts_ms": 3000,
            "spatial_state": {
                "active_zone_id": "",
                "active_zone_type": "",
                "is_inside_zone": False,
            },
        },
        {
            "global_id": "gid_linger",
            "track_id": "1",
            "frame_id": 3,
            "ts_ms": 7000,
            "spatial_state": {
                "active_zone_id": "",
                "active_zone_type": "",
                "is_inside_zone": False,
            },
        },
    ]

    config = {
        "repeat_visit_threshold": 999,
        "linger_threshold_ms": 1000,
        "sensitive_zone_types": ["restricted"],
        "min_behavior_score": 0.6,
        "reappearance_gap_ms": 3000,
    }

    enriched = service.enrich_detections_with_behavior_state(
        detections,
        behavior_config=config,
        camera_id="cam_2",
        request_id="req_2",
    )

    assert enriched[1]["behavior_state"]["is_lingering"] is True
    assert enriched[1]["behavior_state"]["behavior_label"] == "linger"
