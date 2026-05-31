from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from triton_server.app import health
from triton_server.app import (
    SpatiotemporalCorrelationService,
)
from triton_server.app import AnalyticsUtils
from triton_server.app import APIConstants, ERROR_RESPONSES
from triton_server.app import (
    FileSourceUtils,
    ImageAnalysisUtils,
    MediaSourceUtils,
    VideoSourceUtils,
)
from triton_server.app import OCRUtils
from triton_server.app import (
    aggregate_track_votes,
    build_detection_response_item,
    build_source_summary_rows,
    write_csv_rows,
    write_frame_detections_csv,
    write_track_summary_csv,
)
from triton_server.app import RequestTraceUtils
from triton_server.app import DetectionResponseItem


def test_derive_source_id_is_stable():
    source_a = AnalyticsUtils.derive_source_id(" HTTP://Example.com/Feed ")
    source_b = AnalyticsUtils.derive_source_id("http://example.com/feed")

    assert source_a == source_b
    assert len(source_a) == 12


def test_point_in_polygon_and_find_active_zone():
    zone = {
        "zone_id": "z1",
        "zone_type": "restricted",
        "coordinates": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]],
    }

    assert AnalyticsUtils.point_in_polygon((0.2, 0.2), zone["coordinates"]) is True
    assert AnalyticsUtils.point_in_polygon((0.9, 0.9), zone["coordinates"]) is False
    assert AnalyticsUtils.find_active_zone((0.2, 0.2), [zone]) == zone
    assert AnalyticsUtils.find_active_zone((0.9, 0.9), [zone]) is None


def test_velocity_and_direction_helpers():
    assert AnalyticsUtils.normalize_bbox_center([0, 0, 100, 200], 200, 400) == (
        0.25,
        0.25,
    )
    assert AnalyticsUtils.normalize_bbox_center([], 200, 400) == (0.5, 0.5)
    assert AnalyticsUtils.compute_direction_vector((0.0, 0.0), (3.0, 4.0)) == [
        0.6,
        0.8,
    ]
    assert AnalyticsUtils.compute_direction_vector((1.0, 1.0), (1.0, 1.0)) == [
        0.0,
        0.0,
    ]
    assert AnalyticsUtils.compute_speed_estimate((0.0, 0.0), (3.0, 4.0), 1000) == 5.0
    assert AnalyticsUtils.compute_speed_estimate((0.0, 0.0), (1.0, 1.0), 0) == 0.0
    assert AnalyticsUtils.compute_pixel_speed((0.0, 0.0), (12.0, 16.0), 4000) == 5.0
    assert AnalyticsUtils.compute_pixel_speed((100.0, 100.0), (107.0, 106.0), 1000) == 0.0
    assert AnalyticsUtils.convert_pixel_speed_to_kmph(10.0, 5.0) == 7.2
    assert AnalyticsUtils.convert_pixel_speed_to_kmph(10.0, None) == 0.0
    assert AnalyticsUtils.format_velocity_kmph(12.34) == ["12.3 km/h"]
    assert AnalyticsUtils.format_velocity_display(10.0, 5.0) == ["7.2 km/h"]


def test_empty_state_and_catalog_helpers():
    spatial_state = AnalyticsUtils.build_empty_spatial_state()
    behavior_state = AnalyticsUtils.build_empty_behavior_state()

    assert spatial_state == {
        "active_zone_id": "",
        "active_zone_type": "",
        "is_inside_zone": False,
        "spatial_label": "outside_zone",
        "spatial_score": 0.0,
    }
    assert behavior_state["behavior_label"] == "normal"
    assert "restricted" in AnalyticsUtils.get_sensitive_zone_type_catalog()
    assert AnalyticsUtils.get_behavior_label_meaning("linger")
    assert AnalyticsUtils.get_default_sensitive_zone_types() == [
        "sensitive",
        "restricted",
    ]


def test_build_vehicle_episodes_splits_by_gap():
    detections = [
        {"global_id": "gid_1", "track_id": "1", "ts_ms": 1000},
        {"global_id": "gid_1", "track_id": "1", "ts_ms": 2000},
        {"global_id": "gid_1", "track_id": "1", "ts_ms": 7000},
        {"track_id": "2", "ts_ms": 3000},
        {"track_id": "2", "ts_ms": 3500},
        {"ts_ms": 9999},
    ]

    episodes = AnalyticsUtils.build_vehicle_episodes(
        detections, reappearance_gap_ms=3000
    )

    assert len(episodes) == 3
    assert episodes[0]["identity_key"] == "gid_1"
    assert episodes[0]["duration_ms"] == 1000
    assert episodes[1]["episode_index"] == 1
    assert episodes[2]["identity_key"] == "2"


def test_spatiotemporal_correlation_service_enriches_spatial_state_and_motion():
    service = SpatiotemporalCorrelationService()
    zones = [
        {
            "zone_id": "fast",
            "zone_type": "emergency",
            "coordinates": [[0.1, 0.1], [0.7, 0.1], [0.7, 0.7], [0.1, 0.7]],
        }
    ]
    detections = [
        {
            "frame_id": 1,
            "ts_ms": 1000,
            "track_id": "t1",
            "bbox_xyxy": [100.0, 100.0, 300.0, 300.0],
            "center": [200.0, 200.0],
        },
        {
            "frame_id": 2,
            "ts_ms": 2000,
            "track_id": "t1",
            "bbox_xyxy": [200.0, 100.0, 400.0, 300.0],
            "center": [300.0, 200.0],
        },
        {
            "frame_id": 3,
            "ts_ms": 3000,
            "track_id": "",
            "bbox_xyxy": [1800.0, 900.0, 1900.0, 1000.0],
            "center": [1850.0, 950.0],
        },
    ]

    enriched = service.enrich_detections_with_spatial_state(
        detections,
        zones,
        pixels_per_meter=10.0,
        frame_width=1920,
        frame_height=1080,
    )

    assert enriched[0]["spatial_state"]["active_zone_id"] == "fast"
    assert enriched[0]["spatial_state"]["active_zone_type"] == "emergency"
    assert enriched[0]["velocity"] == ["0.0 km/h"]
    assert enriched[1]["direction_vector"][0] > 0
    assert enriched[1]["velocity"] == ["36.0 km/h"]
    assert enriched[2]["spatial_state"]["is_inside_zone"] is False


def test_spatiotemporal_correlation_service_suppresses_small_motion_jitter():
    service = SpatiotemporalCorrelationService()
    detections = [
        {
            "frame_id": 10,
            "ts_ms": 1000,
            "track_id": "t_jitter",
            "bbox_xyxy": [100.0, 100.0, 300.0, 300.0],
            "center": [200.0, 200.0],
        },
        {
            "frame_id": 11,
            "ts_ms": 2000,
            "track_id": "t_jitter",
            "bbox_xyxy": [107.0, 106.0, 307.0, 306.0],
            "center": [207.0, 206.0],
        },
    ]

    enriched = service.enrich_detections_with_spatial_state(
        detections,
        zones=[],
        pixels_per_meter=10.0,
        frame_width=1920,
        frame_height=1080,
    )

    assert enriched[1]["velocity"] == ["0.0 km/h"]
    assert enriched[1]["direction_vector"] == [0.0, 0.0]
    assert enriched[1]["direction"] == "stationary"
    assert enriched[1]["orientation"] == "stationary"


def test_media_source_validation_helpers(tmp_path: Path):
    image_file = tmp_path / "frame.jpg"
    image_file.write_bytes(b"fake")

    validated = FileSourceUtils.validate_file_path(str(image_file), {".jpg"})
    assert validated == image_file

    with pytest.raises(ValueError):
        FileSourceUtils.validate_file_path(str(tmp_path / "missing.jpg"), {".jpg"})

    with pytest.raises(ValueError):
        FileSourceUtils.validate_file_path(str(image_file), {".png"})

    assert (
        MediaSourceUtils.validate_remote_media_url(
            " https://example.com/video.mp4 ",
            source_name="video_url",
            allowed_extensions=[".mp4"],
        )
        == "https://example.com/video.mp4"
    )

    with pytest.raises(ValueError):
        MediaSourceUtils.validate_remote_media_url(
            "ftp://example.com/video.mp4",
            source_name="video_url",
            allowed_extensions=[".mp4"],
        )

    with pytest.raises(ValueError):
        MediaSourceUtils.validate_remote_media_url(
            "https://example.com/video.avi",
            source_name="video_url",
            allowed_extensions=[".mp4"],
        )

    VideoSourceUtils.validate_video_source("https://example.com/live_stream")

    with pytest.raises(ValueError):
        VideoSourceUtils.validate_video_source(
            "https://example.com/video.unsupported_ext"
        )

    request_folder, output_path, source_name = VideoSourceUtils.build_output_paths(
        "https://example.com/path/video.mp4", str(tmp_path), "req_123"
    )
    assert request_folder.exists() is True
    assert source_name == "video.mp4"
    assert output_path.endswith("annotated_video.mp4")


def test_api_constants_and_error_responses():
    assert APIConstants.REQUEST_ID_HEADER == "X-Request-ID"
    assert APIConstants.REQUEST_ID_HEADER_PARAM == "x-request-id"
    assert APIConstants.DEFAULT_REQUEST_ID == "request_without_id"
    assert ERROR_RESPONSES[400]["description"] == "Bad request"
    assert ERROR_RESPONSES[500]["description"] == "Internal server error"


def test_request_trace_utils_build_triton_request_id():
    assert RequestTraceUtils.build_triton_request_id(None, "cam/a", 1) is None
    assert (
        RequestTraceUtils.build_triton_request_id("req_1", "cam/a\\b test", 7)
        == "req_1:cam_a_b_test:7"
    )
    assert (
        RequestTraceUtils.build_triton_request_id("req_1", None, 0)
        == "req_1:unknown_source:0"
    )


def test_ocr_utils_parse_and_normalize():
    assert OCRUtils.parse_result("  abc123 ") == ("abc123", 0.0)
 
    parsed = OCRUtils.parse_result(
        [
            {"text": " AB ", "confidence": "0.8"},
            {"text": "12", "confidence": 1.0},
            {"text": "", "confidence": "bad"},
        ]
    )
    assert parsed[0] == "AB 12"
    assert parsed[1] == pytest.approx(0.9)
 
    assert OCRUtils.normalize_plate_text('"mp(09)$ab|1234"') == "MPC09SABI1234"
    
    normalized = OCRUtils.normalize_plate_text(' "mp-09 cm 0105" ')
    assert normalized == "MP09CM0105"
    assert OCRUtils.validate_plate_text(normalized, mode="strict") == True
    
    normalized = OCRUtils.normalize_plate_text(' "mp-09" ')
    assert normalized == "MP09"
    assert OCRUtils.validate_plate_text(normalized, mode="balanced") == True
    
    normalized = OCRUtils.normalize_plate_text(' "mp-09 cm 010" ')
    assert normalized == "MP09CM010"
    assert OCRUtils.validate_plate_text(normalized, mode="balanced") == True
    
    normalized = OCRUtils.normalize_plate_text("Mahadev")
    assert normalized == "MAHADEV"
    assert OCRUtils.validate_plate_text(normalized, mode="strict") == False
    assert OCRUtils.validate_plate_text(normalized, mode="balanced") == True
    
    assert OCRUtils.normalize_plate_text("") == ""
    assert OCRUtils.validate_plate_text("", mode="balanced") == False


def test_image_analysis_utils_remaining_color_branches():
    colorful_low_hist = np.zeros((80, 120, 3), dtype=np.uint8)
    colorful_low_hist[:] = (0, 255, 0)
    with patch("app.utils.media_utils.cv2.calcHist", return_value=np.zeros((180, 1), dtype=np.float32)):
        assert ImageAnalysisUtils.extract_dominant_color(colorful_low_hist, min_area=100) == ""

    grey_img = np.full((60, 120, 3), 140, dtype=np.uint8)
    with patch.object(ImageAnalysisUtils, "_VALID_PLATE_COLORS", {"white", "yellow", "green", "black"}), patch.object(
        ImageAnalysisUtils, "extract_dominant_color", return_value="grey"
    ):
        assert ImageAnalysisUtils.extract_plate_color(grey_img) == "white"

    cyan_img = np.zeros((60, 120, 3), dtype=np.uint8)
    cyan_img[:] = (255, 255, 0)
    assert ImageAnalysisUtils.extract_plate_color(cyan_img) == ""

    empty_roi = np.zeros((30, 30, 3), dtype=np.uint8)
    with patch("app.utils.media_utils.cv2.cvtColor", return_value=np.zeros((0, 0, 3), dtype=np.uint8)):
        assert ImageAnalysisUtils.extract_dominant_color(empty_roi, min_area=100) == ""


def test_analytics_utils_remaining_small_branches_again():
    assert AnalyticsUtils.direction_label_from_vector([1.0]) == "stationary"
    assert AnalyticsUtils.orientation_label_from_motion([1.0]) == "unknown"

    episodes = AnalyticsUtils.build_vehicle_episodes([
        {"track_id": "", "ts_ms": 1000},
        {"global_id": "gid_1", "ts_ms": 1000},
        {"global_id": "gid_1", "ts_ms": 6000},
    ])
    assert len(episodes) == 2


def test_output_serializers_aggregate_and_csv_writers(tmp_path: Path):
    detections = [
        {
            "frame_id": 1,
            "track_id": "10",
            "ocr_text": "MP09",
            "conf": 0.8,
            "ocr_confidence": 0.7,
        },
        {
            "frame_id": 2,
            "track_id": "10",
            "ocr_text": "MP09",
            "conf": 0.9,
            "ocr_confidence": 0.8,
        },
        {
            "frame_id": 3,
            "track_id": "10",
            "ocr_text": "OTHER",
            "conf": 0.99,
            "ocr_confidence": 0.2,
        },
        {
            "frame_id": 4,
            "track_id": "20",
            "ocr_text": "GJ01",
            "conf": 0.6,
            "ocr_confidence": 0.5,
        },
    ]

    rows = aggregate_track_votes(detections)
    assert len(rows) == 2
    row10 = next(r for r in rows if r["track_id"] == 10)
    assert row10["plate_text"] == "MP09"
    assert row10["votes"] == 2

    source_rows = build_source_summary_rows("cam_a", detections)
    assert all(r["source"] == "cam_a" for r in source_rows)

    frame_csv = tmp_path / "frame.csv"
    summary_header = [
        "track_id",
        "plate_text",
        "votes",
        "avg_confidence",
        "avg_confidence_ocr",
    ]
    frame_header = ["frame", "track_id", "plate_text", "confidence", "confidence_ocr"]

    assert write_frame_detections_csv(
        detections=detections,
        csv_path=frame_csv,
        header=frame_header,
    ) == str(frame_csv)
    assert frame_csv.exists() is True

    track_summary_path = write_track_summary_csv(
        detections=detections,
        frame_csv_path=frame_csv,
        header=summary_header,
    )
    assert track_summary_path is not None
    assert Path(track_summary_path).exists() is True

    empty_rows_result = write_csv_rows(
        csv_path=tmp_path / "empty.csv",
        rows=[],
        header=["a"],
    )
    assert empty_rows_result is None

    written_rows = write_csv_rows(
        csv_path=tmp_path / "generic.csv",
        rows=[{"a": 1}],
        header=["a"],
    )
    assert written_rows == str(tmp_path / "generic.csv")


def test_build_detection_response_item_and_skipped_csv_paths(tmp_path: Path):
    item = DetectionResponseItem(track_id="12", name="car", conf=0.88)
    dumped = build_detection_response_item(item)
    assert dumped["track_id"] == "12"
    assert dumped["name"] == "car"

    no_plate_rows = aggregate_track_votes(
        [{"track_id": "10", "ocr_text": "", "conf": 0.1, "ocr_confidence": 0.1}]
    )
    assert no_plate_rows == []

    assert (
        write_track_summary_csv(
            detections=[{"track_id": "1", "ocr_text": ""}],
            frame_csv_path=tmp_path / "frame.csv",
            header=[
                "track_id",
                "plate_text",
                "votes",
                "avg_confidence",
                "avg_confidence_ocr",
            ],
        )
        is None
    )
    assert (
        write_frame_detections_csv(
            detections=[],
            csv_path=tmp_path / "empty_frame.csv",
            header=["frame", "track_id", "plate_text", "confidence", "confidence_ocr"],
        )
        is None
    )


def test_output_serializers_error_paths_return_none(tmp_path: Path):
    csv_path = tmp_path / "error.csv"

    with patch("app.utils.output_serializers.open", side_effect=OSError("nope")):
        assert write_csv_rows(csv_path=csv_path, rows=[{"a": 1}], header=["a"]) is None
        assert (
            write_frame_detections_csv(
                detections=[{"frame_id": 1}],
                csv_path=csv_path,
                header=[
                    "frame",
                    "track_id",
                    "plate_text",
                    "confidence",
                    "confidence_ocr",
                ],
            )
            is None
        )
        assert (
            write_track_summary_csv(
                detections=[{"track_id": "1", "ocr_text": "ABC"}],
                frame_csv_path=csv_path,
                header=[
                    "track_id",
                    "plate_text",
                    "votes",
                    "avg_confidence",
                    "avg_confidence_ocr",
                ],
            )
            is None
        )


def test_health_route_returns_expected_shape():
    response = health("req-123")

    assert response["request_id"] == "req-123"
    assert response["status"] == "ok"
    assert "time" in response
    assert "app" in response
    assert "env" in response
    assert "triton_server" in response


def test_image_analysis_utils_branches():
    assert ImageAnalysisUtils.extract_dominant_color(np.array([])) == ""
    assert ImageAnalysisUtils.hue_to_color_name(5) == "red"
    assert ImageAnalysisUtils.hue_to_color_name(20) == "orange"
    assert ImageAnalysisUtils.hue_to_color_name(30) == "yellow"
    assert ImageAnalysisUtils.hue_to_color_name(70) == "green"
    assert ImageAnalysisUtils.hue_to_color_name(95) == "cyan"
    assert ImageAnalysisUtils.hue_to_color_name(110) == "blue"
    assert ImageAnalysisUtils.hue_to_color_name(150) == "purple"
    assert ImageAnalysisUtils.hue_to_color_name(165) == "pink"

    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(tiny, min_area=100) == ""

    white_img = np.full((60, 120, 3), 255, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_plate_color(white_img) == "white"

    black_img = np.zeros((60, 120, 3), dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(black_img, min_area=100) == ""

    colorful = np.zeros((80, 120, 3), dtype=np.uint8)
    colorful[:, :] = (0, 0, 255)
    extracted = ImageAnalysisUtils.extract_dominant_color(colorful, min_area=100)
    assert extracted in {
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "pink",
        "white",
        "black",
        "grey",
        "silver",
        "",
    }


def test_image_source_utils_url_validation_and_load_paths():
    from triton_server.app import ImageSourceUtils

    with pytest.raises(ValueError):
        ImageSourceUtils.validate_image_source_url("ftp://example.com/a.jpg")

    with pytest.raises(ValueError):
        ImageSourceUtils.validate_image_source_url("https://example.com/a.tiff")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"binary"

    with patch("app.utils.media_utils.urlopen", return_value=_Resp()), patch(
        "app.utils.media_utils.cv2.imdecode",
        return_value=np.zeros((10, 10, 3), dtype=np.uint8),
    ):
        img = ImageSourceUtils.load_image_from_url("https://example.com/a.jpg")
        assert img.shape == (10, 10, 3)

    with patch("app.utils.media_utils.urlopen", return_value=_Resp()), patch(
        "app.utils.media_utils.cv2.imdecode", return_value=None
    ):
        with pytest.raises(ValueError):
            ImageSourceUtils.load_image_from_url("https://example.com/a.jpg")
