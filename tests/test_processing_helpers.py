from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from triton_server.app import ANPRService, LiveVideoSourceProcessor, _make_tracker, settings


def _new_anpr_service_for_helpers() -> ANPRService:
    svc = ANPRService.__new__(ANPRService)
    svc.plate_class_offset = 1
    svc.class_names = ["car", "number_plate"]
    svc.vehicle_class_id_name_map = {0: "car"}
    svc.plate_class_id_name_map = {0: "number_plate"}
    svc.ocr_class_ids = {1}
    svc.plate_candidate_vehicle_classes = {"car", "vehicle"}
    svc.tracker = None
    svc.ocr_service = Mock()
    return svc


def test_anpr_helper_decode_and_nms_and_merge_paths():
    svc = _new_anpr_service_for_helpers()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert ANPRService._decode_inference_payload(
        frame,
        np.array([1, 2, 3]),
        preprocess_meta={},
        class_names=["car"],
        class_id_offset=0,
        confidence_threshold=0.2,
    )[0].shape == (0, 4)

    raw = np.array(
        [
            [0.5, 0.5, 0.4, 0.4, 0.95],
            [0.3, 0.3, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    boxes, conf, cls = ANPRService._decode_inference_payload(
        frame,
        raw,
        preprocess_meta={
            "input_size": (100, 100),
            "x_offset": 0,
            "y_offset": 0,
            "scale": 1.0,
        },
        class_names=["car"],
        class_id_offset=0,
        confidence_threshold=0.2,
    )
    assert len(boxes) == 1
    assert float(conf[0]) == pytest.approx(0.95)
    assert cls[0] == 0

    assert len(ANPRService._nms(np.empty((0, 4), dtype=np.float32), np.array([]), 0.5)) == 0

    merged = svc._merge_detections([], [], [], [])
    assert merged[0].shape == (0, 4)


def test_anpr_tracking_and_ocr_helper_paths():
    svc = _new_anpr_service_for_helpers()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    merged_xyxy = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    merged_conf = np.array([0.9], dtype=np.float32)
    merged_cls = np.array([0], dtype=int)
    merged_sources = np.array(["vehicle_model"])

    tracked = svc._track_detections(
        frame,
        merged_xyxy,
        merged_conf,
        merged_cls,
        merged_sources,
        tracker=None,
        enable_tracking=False,
    )
    assert len(tracked[0]) == 1
    assert tracked[1][0] == 0

    vehicle_record = {
        "result_item": {"cls": 0, "name": "car", "ocr_confidence": 0.0, "sources": []},
        "box": np.array([0.0, 0.0, 30.0, 30.0]),
        "base_label": "#1 car",
        "label": "#1 car",
    }
    plate_record = {
        "result_item": {
            "cls": 1,
            "name": "number_plate",
            "ocr_confidence": 0.8,
            "ocr_text": "MP09",
            "conf": 0.95,
            "area_px": 100,
            "bbox_xyxy": [5.0, 5.0, 15.0, 15.0],
            "plate_color": "white",
            "sources": ["plate_model"],
        },
        "box": np.array([5.0, 5.0, 15.0, 15.0]),
        "base_label": "#2 plate",
        "label": "#2 plate",
    }

    assert svc._is_plate_candidate_vehicle_record(vehicle_record) is True
    assert svc._is_plate_candidate_vehicle_record(plate_record) is False
    assert svc._bbox_containment_score(plate_record["box"], vehicle_record["box"]) > 0.5

    should_plate_ocr = svc._should_run_ocr_for_record(plate_record, [vehicle_record, plate_record])
    assert bool(should_plate_ocr) is True

    merged_records = svc._associate_plates_to_vehicles([vehicle_record, plate_record])
    assert len(merged_records) == 1
    assert merged_records[0]["result_item"]["ocr_text"] == "MP09"
    assert merged_records[0]["result_item"]["plate_conf"] == 0.95


def test_anpr_maybe_run_ocr_and_run_ocr_on_records():
    svc = _new_anpr_service_for_helpers()
    crop = np.ones((20, 20, 3), dtype=np.uint8)

    record = {
        "order": 0,
        "crop": crop,
        "confidence": 0.99,
        "blur_score": 999.0,
        "tracker_id": 1,
        "base_label": "#1 car",
        "label": "#1 car",
        "result_item": {
            "cls": 1,
            "name": "number_plate",
            "ocr_text": "",
            "ocr_confidence": 0.0,
        },
    }
    vehicle = {
        "order": 1,
        "crop": crop,
        "confidence": 0.99,
        "blur_score": 999.0,
        "tracker_id": 2,
        "base_label": "#2 car",
        "label": "#2 car",
        "box": np.array([0.0, 0.0, 40.0, 40.0]),
        "result_item": {"cls": 0, "name": "car", "ocr_text": "", "ocr_confidence": 0.0},
    }
    record["box"] = np.array([5.0, 5.0, 20.0, 20.0])

    svc.ocr_service.recognize.return_value = [{"text": "MP09AB1234", "confidence": 0.8}]
    with patch.object(settings, "plate_blur_threshold", 0.1):
        elapsed = svc._maybe_run_ocr(record, [vehicle, record], 0.5, plate_text_mode="balanced")
        assert elapsed >= 0.0
        assert record["result_item"]["ocr_text"] == "MP09AB1234"

    record["result_item"]["ocr_text"] = ""
    record["result_item"]["ocr_confidence"] = 0.0
    record["ocr_confidence"] = 0.0
    record["label"] = record["base_label"]
    svc.ocr_service.recognize.return_value = [{"text": "JW", "confidence": 0.95}]
    with patch.object(settings, "plate_blur_threshold", 0.1):
        svc._maybe_run_ocr(record, [vehicle, record], 0.5, plate_text_mode="balanced")
        assert record["result_item"]["ocr_text"] == "JW"
        assert record["label"] == f"{record['base_label']} JW"

    record["result_item"]["ocr_text"] = ""
    record["result_item"]["ocr_confidence"] = 0.0
    record["ocr_confidence"] = 0.0
    record["label"] = record["base_label"]
    svc.ocr_service.recognize.return_value = [{"text": "MP09", "confidence": 0.8}]
    with patch.object(settings, "plate_blur_threshold", 0.1):
        svc._maybe_run_ocr(record, [vehicle, record], 0.5, plate_text_mode="balanced")
        assert record["result_item"]["ocr_text"] == "MP09"
        assert record["label"] == f"{record['base_label']} MP09"

    record["result_item"]["ocr_text"] = ""
    record["result_item"]["ocr_confidence"] = 0.0
    record["ocr_confidence"] = 0.0
    record["label"] = record["base_label"]
    svc.ocr_service.recognize.return_value = [{"text": "MP09AB1234", "confidence": 0.4}]
    with patch.object(settings, "plate_blur_threshold", 0.1):
        svc._maybe_run_ocr(record, [vehicle, record], 0.5, plate_text_mode="balanced")
        assert record["result_item"]["ocr_text"] == ""
        assert record["label"] == record["base_label"]

    records = [record, vehicle]
    total = svc._run_ocr_on_records(records, ocr_confidence_threshold=0.5, plate_text_mode="balanced")
    assert total >= 0.0


def test_video_source_processor_helpers_and_result_building(tmp_path: Path):
    processor = LiveVideoSourceProcessor(service=Mock(), spatial_service=None, behavioral_service=None)

    csv_path, summary_path = processor._write_csv_outputs(
        output_path=str(tmp_path / "out.mp4"),
        detections=[],
        save_csv=True,
    )
    assert csv_path is None
    assert summary_path is None

    with (
        patch(
            "modules.anpr.services.video_source_processor.output_serializers.write_frame_detections_csv",
            return_value=str(tmp_path / "a.csv"),
        ),
        patch(
            "modules.anpr.services.video_source_processor.output_serializers.write_track_summary_csv",
            return_value=str(tmp_path / "b.csv"),
        ),
    ):
        csv_path, summary_path = processor._write_csv_outputs(
            output_path=str(tmp_path / "out.mp4"),
            detections=[{"track_id": "1", "ocr_text": "X"}],
            save_csv=True,
        )
    assert csv_path and summary_path

    detections = [{"track_id": "1"}]
    enriched = processor._enrich_detections_with_analytics(
        detections,
        camera_id="cam_1",
        lat=10.0,
        lon=20.0,
        pixels_per_meter=15.0,
        zones=[],
        behavior_config=None,
        frame_width=100,
        frame_height=100,
        request_id="req_1",
    )
    assert enriched[0]["camera_id"] == "cam_1"

    result = processor._build_result(
        source="https://example.com/v.mp4",
        source_name="v.mp4",
        output_path=str(tmp_path / "out.mp4"),
        csv_path=None,
        summary_csv_path=None,
        source_fps=25.0,
        target_fps=5.0,
        sample_interval=5,
        frame_count=100,
        processed_count=20,
        all_detections=[{"track_id": "1"}],
        t_start=0.0,
        camera_id="cam_1",
        lat=11.1,
        lon=22.2,
        pixels_per_meter=12.0,
        zones=[{"zone_id": "z1"}],
        behavior_config={"repeat_visit_threshold": 3},
        is_ocr_enabled=True,
        platform="anpr_object",
    )
    assert result["camera_id"] == "cam_1"
    assert result["lat"] == 11.1
    assert result["lon"] == 22.2
    assert result["pixels_per_meter"] == 12.0
    assert result["zones"]
    assert result["behavior_config"]


def test_process_frame_after_inference_success_and_error_paths():
    svc = _new_anpr_service_for_helpers()
    svc.global_tracking_service = Mock()
    svc.plate_class_id_name_map = {0: "number_plate"}
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    svc._collect_model_detections = Mock(return_value=([], [], [], []))
    svc._merge_detections = Mock(
        return_value=(
            np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([0], dtype=int),
            np.array(["vehicle_model"]),
        )
    )
    svc._track_detections = Mock(
        return_value=(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    )
    record = {
        "result_item": {"cls": 0, "name": "car", "global_id": ""},
        "label": "car",
        "box": np.array([0, 0, 5, 5]),
    }
    svc._build_detection_records = Mock(return_value=[record])
    svc._run_ocr_on_records = Mock()
    svc._associate_plates_to_vehicles = Mock(return_value=[record])
    svc._annotate_detection_records = Mock(return_value=frame)

    annotated, results = svc.process_frame_after_inference(
        frame,
        [{"x": 1}],
        tracker=None,
        enable_tracking=False,
        confidence_threshold=0.2,
        ocr_confidence_threshold=0.5,
        ocr_match_confidence=0.85,
        global_id_match_score=0.7,
        plate_text_mode="balanced",
        is_ocr_enabled=True,
    )
    assert annotated is frame
    assert results == [record["result_item"]]
    svc.global_tracking_service.resolve_detections.assert_called_once()

    svc._collect_model_detections = Mock(side_effect=RuntimeError("boom"))
    annotated, results = svc.process_frame_after_inference(
        frame,
        [{"x": 1}],
        confidence_threshold=0.2,
        ocr_confidence_threshold=0.5,
        ocr_match_confidence=0.85,
        global_id_match_score=0.7,
        plate_text_mode="balanced",
        is_ocr_enabled=True,
    )
    assert results == []
    assert annotated is frame


def test_process_frame_and_image_delegates(tmp_path: Path):
    svc = _new_anpr_service_for_helpers()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    svc.infer_frame_payloads = AsyncMock(return_value=[{"payload": 1}])
    svc.process_frame_after_inference = Mock(return_value=(frame, [{"ok": 1}]))

    import anyio

    async def _run_process_frame_success():
        return await svc.process_frame(
            frame,
            0,
            camera_id="cam1",
            request_id="req1",
            source_name="a.jpg",
            processing_config={"is_ocr_enabled": True, "platform": "anpr_object"},
        )

    annotated, results = anyio.run(_run_process_frame_success)
    assert annotated is frame
    assert results == [{"ok": 1}]

    svc.infer_frame_payloads = AsyncMock(side_effect=RuntimeError("infer fail"))

    async def _run_process_frame_error():
        return await svc.process_frame(frame, 1, processing_config={"is_ocr_enabled": True, "platform": "anpr_object"})

    annotated, results = anyio.run(_run_process_frame_error)
    assert results == []
    assert annotated is frame

    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"x")
    svc.process_frame = AsyncMock(return_value=(frame, [{"img": 1}]))
    with (
        patch(
            "modules.anpr.services.anpr_service.ImageSourceUtils.validate_image_source_path",
            return_value=image_path,
        ),
        patch("modules.anpr.services.anpr_service.cv2.imread", return_value=frame),
    ):

        async def _run_process_image_source_success():
            return await svc.process_image_source(str(image_path), "req")

        annotated, results, output_path = anyio.run(_run_process_image_source_success)
        assert annotated is frame
        assert results == [{"img": 1}]
        assert output_path is not None

    with (
        patch(
            "modules.anpr.services.anpr_service.ImageSourceUtils.validate_image_source_path",
            return_value=image_path,
        ),
        patch("modules.anpr.services.anpr_service.cv2.imread", return_value=None),
    ):

        async def _run_process_image_source_error():
            return await svc.process_image_source(str(image_path), "req")

        with pytest.raises(ValueError):
            anyio.run(_run_process_image_source_error)

    svc.process_frame = AsyncMock(return_value=(frame, [{"url": 1}]))
    with patch("modules.anpr.services.anpr_service.asyncio.to_thread", new=AsyncMock(return_value=frame)):

        async def _run_process_image_url():
            return await svc.process_image_url("https://example.com/a.jpg", 2, "req")

        annotated, results, output_path = anyio.run(_run_process_image_url)
        assert annotated is frame
        assert results == [{"url": 1}]
        assert output_path is not None


def test_video_source_processor_static_helpers_and_analytics_branches(tmp_path: Path):
    fake_capture = Mock()
    fake_capture.isOpened.return_value = True
    with patch(
        "modules.anpr.services.video_source_processor.cv2.VideoCapture",
        return_value=fake_capture,
    ):
        assert LiveVideoSourceProcessor._open_capture("src") is fake_capture

    bad_capture = Mock()
    bad_capture.isOpened.return_value = False
    with patch("modules.anpr.services.video_source_processor.cv2.VideoCapture", return_value=bad_capture):
        with pytest.raises(ValueError):
            LiveVideoSourceProcessor._open_capture("src")

    capture = Mock()
    capture.get.side_effect = [0.0, 0.0, 0.0, 0.0]
    source_fps, target_fps, w, h, total = LiveVideoSourceProcessor._resolve_video_properties(
        capture, np.zeros((12, 24, 3), dtype=np.uint8), 10.0
    )
    assert source_fps == 25.0
    assert target_fps == 10.0
    assert (w, h) == (24, 12)
    assert total is None

    opened_writer = Mock()
    opened_writer.isOpened.return_value = True
    with (
        patch(
            "modules.anpr.services.video_source_processor.cv2.VideoWriter",
            return_value=opened_writer,
        ),
        patch("modules.anpr.services.video_source_processor.cv2.VideoWriter.fourcc", return_value=123),
    ):
        writer = LiveVideoSourceProcessor._create_writer(str(tmp_path / "a.mp4"), 25.0, 10, 10, capture)
        assert writer is opened_writer

    closed_writer = Mock()
    closed_writer.isOpened.return_value = False
    with (
        patch(
            "modules.anpr.services.video_source_processor.cv2.VideoWriter",
            return_value=closed_writer,
        ),
        patch("modules.anpr.services.video_source_processor.cv2.VideoWriter.fourcc", return_value=123),
        pytest.raises(ValueError),
    ):
        LiveVideoSourceProcessor._create_writer(str(tmp_path / "a.mp4"), 25.0, 10, 10, capture)

    tracker = _make_tracker(0.2)
    assert tracker is not None

    service = Mock()
    processor = LiveVideoSourceProcessor(service=service, spatial_service=Mock(), behavioral_service=Mock())
    processor.spatial_service.enrich_detections_with_spatial_state.side_effect = RuntimeError("spatial fail")
    processor.behavioral_service.enrich_detections_with_behavior_state.side_effect = RuntimeError("behavior fail")
    detections = [{"track_id": "1"}]
    enriched = processor._enrich_detections_with_analytics(detections, "cam", None, None, None, [], {}, 100, 100, "req")
    assert enriched[0]["camera_id"] == "cam"

    processor = LiveVideoSourceProcessor(service=Mock(), spatial_service=Mock(), behavioral_service=Mock())
    processor.spatial_service.enrich_detections_with_spatial_state.return_value = [
        {"track_id": "1", "spatial_state": {}}
    ]
    processor.behavioral_service.enrich_detections_with_behavior_state.return_value = [
        {"track_id": "1", "behavior_state": {}}
    ]
    enriched = processor._enrich_detections_with_analytics(
        [{"track_id": "1"}], "cam", None, None, None, [], {}, 100, 100, "req"
    )
    assert "behavior_state" in enriched[0] or "spatial_state" in enriched[0]


def test_video_source_processor_batch_helpers(tmp_path: Path):
    service = Mock()
    service.infer_frame_payloads = AsyncMock(side_effect=[[{"a": 1}], [{"b": 2}]])
    processor = LiveVideoSourceProcessor(service=service)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)]

    import anyio

    batch = anyio.run(processor._run_inference_batch, frames, [1, 2], "req", "src", 0.5, True, "anpr_object")
    assert len(batch) == 2

    writer = Mock()
    service.process_frame_after_inference.return_value = (
        frames[0],
        [{"track_id": "1"}],
    )
    with patch("modules.anpr.services.video_source_processor.cv2.imwrite", return_value=True):
        processed = processor._process_inference_batch(
            frames=frames,
            indices=[0, 10],
            payloads_batch=[[{"a": 1}], [{"b": 2}]],
            debug_dir=tmp_path,
            tracker=Mock(),
            writer=writer,
            source_fps=10.0,
            all_detections=[],
            request_id="req",
            camera_id="cam1",
            confidence_threshold=0.2,
            ocr_confidence_threshold=0.5,
            ocr_match_confidence=0.85,
            global_id_match_score=0.7,
            plate_text_mode="balanced",
            is_ocr_enabled=True,
        )
    assert processed == 2


def test_anpr_service_init_infer_and_record_annotation_helpers():
    ocr = Mock()
    vehicle_client = Mock()
    plate_client = Mock()
    global_tracking = Mock()

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    vehicle_client.infer = AsyncMock(return_value=(np.array([[1.0]], dtype=np.float32), {"meta": 1}))
    plate_client.infer = AsyncMock(return_value=(np.array([[2.0]], dtype=np.float32), {"meta": 2}))

    fake_video_processor = Mock()
    fake_tracker = Mock()
    with (
        patch("modules.anpr.services.anpr_service.BYTETracker", return_value=fake_tracker),
        patch(
            "modules.anpr.services.anpr_service.LiveVideoSourceProcessor",
            return_value=fake_video_processor,
        ),
        patch.object(settings, "vehicle_class_id_map", '{"0":"car","2":"bus"}'),
        patch.object(settings, "plate_class_id_map", '{"0":"number_plate"}'),
    ):
        service = ANPRService(
            ocr_service=ocr,
            vehicle_triton_client=vehicle_client,
            plate_triton_client=plate_client,
            global_tracking_service=global_tracking,
        )

    assert service.vehicle_class_names == ["car", "class_1", "bus"]
    assert service.plate_class_names == ["number_plate"]
    assert service.video_source_processor is fake_video_processor

    import anyio

    async def _run_infer():
        return await service.infer_frame_payloads(
            frame,
            3,
            request_id="req",
            source_name="cam1",
            ocr_confidence_threshold=0.5,
            is_ocr_enabled=True,
            platform="anpr_object",
        )

    payloads = anyio.run(_run_infer)
    assert len(payloads) == 2
    assert payloads[0]["source"] == "vehicle"
    assert payloads[1]["source"] == "plate"

    with (
        patch("modules.anpr.services.anpr_service.ImageAnalysisUtils.calculate_blur", return_value=12.0),
        patch(
            "modules.anpr.services.anpr_service.ImageAnalysisUtils.extract_dominant_color",
            return_value="white",
        ),
        patch(
            "modules.anpr.services.anpr_service.ImageAnalysisUtils.extract_plate_color",
            return_value="white",
        ),
    ):
        vehicle_record = service._build_detection_record(
            frame,
            np.array([1.0, 1.0, 10.0, 10.0]),
            class_id=0,
            tracker_id=5,
            confidence=0.9,
            model_source="vehicle_model",
            order_idx=0,
            camera_id="cam1",
        )
        plate_record = service._build_detection_record(
            frame,
            np.array([2.0, 2.0, 8.0, 8.0]),
            class_id=service.plate_class_offset,
            tracker_id=6,
            confidence=0.8,
            model_source="plate_model",
            order_idx=1,
            camera_id="cam1",
        )

    assert vehicle_record["result_item"]["color"] == "white"
    assert plate_record["result_item"]["plate_color"] == "white"

    class _Annotator:
        def __init__(self, image):
            self.image = image
            self.calls = []

        def box_label(self, box, label, color=None):
            self.calls.append((box, label, color))

        def result(self):
            return self.image

    vehicle_record["result_item"]["global_id"] = "gid1"
    vehicle_record["label"] = "#5 car"
    vehicle_record["result_item"]["plate_bbox_xyxy"] = [2, 2, 8, 8]
    vehicle_record["result_item"]["ocr_text"] = "MP09"
    with patch("modules.anpr.services.anpr_service.Annotator", _Annotator):
        annotated = service._annotate_detection_records(frame, [vehicle_record, plate_record])
    assert annotated.shape == frame.shape


def test_infer_frame_payloads_runs_sequential_plate_stage_for_valid_vehicle():
    svc = _new_anpr_service_for_helpers()
    svc.vehicle_class_names = ["car"]
    svc.plate_class_names = ["number_plate"]
    svc.plate_class_offset = 1
    svc.vehicle_triton_client = Mock()
    svc.vehicle_triton_client.infer = AsyncMock(return_value=(np.array([[1.0]], dtype=np.float32), {"meta": 1}))

    expected_plate_payload = {
        "source": "plate",
        "decoded_boxes": np.empty((0, 4), dtype=np.float32),
        "decoded_confs": np.empty((0,), dtype=np.float32),
        "decoded_cls": np.empty((0,), dtype=int),
        "preprocess_meta": {"meta": 1},
        "class_names": ["number_plate"],
        "class_id_offset": 1,
    }

    import anyio

    with (
        patch.object(
            svc,
            "_filter_valid_vehicle_boxes",
            return_value=[np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)],
        ),
        patch.object(
            svc,
            "_detect_plates_in_vehicles",
            new=AsyncMock(return_value=expected_plate_payload),
        ),
    ):

        async def _run_infer_payloads():
            return await svc.infer_frame_payloads(
                np.zeros((16, 16, 3), dtype=np.uint8),
                frame_idx=3,
                request_id="req",
                source_name="cam",
                ocr_confidence_threshold=0.5,
                is_ocr_enabled=True,
                platform="anpr_object",
            )

        payloads = anyio.run(_run_infer_payloads)

    assert len(payloads) == 2
    assert payloads[0]["source"] == "vehicle"
    assert payloads[1]["source"] == "plate"


def test_detect_plates_collect_merge_and_track_paths_for_new_flow():
    svc = _new_anpr_service_for_helpers()
    svc.plate_class_names = ["number_plate"]
    svc.plate_class_offset = 1
    svc.plate_triton_client = Mock()
    svc.plate_triton_client.infer = AsyncMock(return_value=(np.array([[1.0]], dtype=np.float32), {"meta": 2}))

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    vehicle_boxes = [np.array([1.0, 2.0, 11.0, 12.0], dtype=np.float32)]

    import anyio

    with patch.object(
        svc,
        "_decode_inference_payload",
        return_value=(
            np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([1], dtype=int),
        ),
    ):

        async def _run_detect_plates():
            return await svc._detect_plates_in_vehicles(
                frame,
                vehicle_boxes,
                frame_idx=0,
                request_id="req",
                base_preprocess_meta={"meta": 1},
                confidence_threshold=0.5,
            )

        plate_payload = anyio.run(_run_detect_plates)

    assert plate_payload["source"] == "plate"
    assert plate_payload["decoded_boxes"].shape == (1, 4)
    assert plate_payload["decoded_boxes"][0].tolist() == [1.0, 2.0, 3.0, 4.0]

    all_xyxy, all_conf, all_cls, all_src = svc._collect_model_detections(
        frame,
        [plate_payload],
        confidence_threshold=0.2,
    )
    assert len(all_xyxy) == 1
    assert all_src[0] == settings.plate_model_name

    with patch.object(svc, "_nms", return_value=np.array([0], dtype=int)):
        merged_xyxy, merged_conf, merged_cls, merged_src = svc._merge_detections(
            all_xyxy,
            all_conf,
            all_cls,
            all_src,
        )

    tracker = Mock()
    tracker.update.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 9.0, 0.9, 0.0, 0.0]], dtype=np.float32)
    t_xyxy, t_ids, t_conf, t_cls, t_sources = svc._track_detections(
        frame,
        merged_xyxy,
        merged_conf,
        merged_cls,
        merged_src,
        tracker=tracker,
        enable_tracking=True,
    )
    assert t_xyxy.shape[0] == 1
    assert int(t_ids[0]) == 9
    assert len(t_sources) == 1


def test_detect_plates_returns_empty_payload_for_empty_crops():
    svc = _new_anpr_service_for_helpers()
    svc.plate_class_names = ["number_plate"]
    svc.plate_class_offset = 1
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    import anyio

    async def _run_empty_case():
        return await svc._detect_plates_in_vehicles(
            frame,
            [np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)],
            frame_idx=0,
            request_id="req",
            base_preprocess_meta={"meta": 1},
            confidence_threshold=0.5,
        )

    payload = anyio.run(_run_empty_case)
    assert payload["source"] == "plate"
    assert payload["decoded_boxes"].shape == (0, 4)


def test_video_source_processor_process_success_and_initial_read_failure(
    tmp_path: Path,
):
    service = Mock()
    processor = LiveVideoSourceProcessor(service=service, spatial_service=None, behavioral_service=None)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class _Capture:
        def __init__(self, reads, props=None):
            self._reads = list(reads)
            self._props = props or {}
            self.released = False

        def read(self):
            if self._reads:
                return self._reads.pop(0)
            return False, None

        def get(self, key):
            return self._props.get(key, 0)

        def release(self):
            self.released = True

    class _Writer:
        def __init__(self):
            self.writes = 0
            self.released = False

        def write(self, frame):
            self.writes += 1

        def release(self):
            self.released = True

    import anyio

    request_folder = tmp_path / "req"
    output_path = str(tmp_path / "out.mp4")

    success_capture = _Capture(
        reads=[(True, frame), (True, frame), (False, None)],
        props={5: 10.0, 3: 4.0, 4: 4.0, 7: 2.0},
    )
    writer = _Writer()

    with (
        patch("modules.anpr.services.video_source_processor.VideoSourceUtils.validate_video_source"),
        patch(
            "modules.anpr.services.video_source_processor.VideoSourceUtils.build_output_paths",
            return_value=(request_folder, output_path, "src.mp4"),
        ),
        patch.object(processor, "_open_capture", return_value=success_capture),
        patch.object(processor, "_create_writer", return_value=writer),
        patch.object(
            processor,
            "_run_inference_batch",
            new=AsyncMock(return_value=[[{"p": 1}], [{"p": 2}]]),
        ),
        patch.object(processor, "_process_inference_batch", return_value=2),
        patch.object(processor, "_enrich_detections_with_analytics", return_value=[{"track_id": "1"}]),
        patch.object(processor, "_write_csv_outputs", return_value=(None, None)),
    ):

        async def _run_success_process():
            return await processor.process(
                "src.mp4",
                str(tmp_path),
                True,
                "req1",
                "cam1",
                None,
                None,
                None,
                [],
                None,
                frames_per_second=10.0,
                confidence_threshold=0.2,
                ocr_confidence_threshold=0.5,
                ocr_match_confidence=0.85,
                global_id_match_score=0.7,
                plate_text_mode="balanced",
                is_ocr_enabled=True,
                platform="anpr_object",
            )

        result = anyio.run(_run_success_process)

    assert result["video_path"] == "src.mp4"
    assert result["processed_frames"] == 2
    assert success_capture.released is True
    assert writer.released is True

    failed_capture = _Capture(reads=[(False, None)])
    writer2 = _Writer()
    with (
        patch("modules.anpr.services.video_source_processor.VideoSourceUtils.validate_video_source"),
        patch(
            "modules.anpr.services.video_source_processor.VideoSourceUtils.build_output_paths",
            return_value=(request_folder, output_path, "src.mp4"),
        ),
        patch.object(processor, "_open_capture", return_value=failed_capture),
        patch.object(processor, "_create_writer", return_value=writer2),
        pytest.raises(ValueError),
    ):

        async def _run_failed_process():
            return await processor.process(
                "src.mp4",
                str(tmp_path),
                True,
                "req2",
                "cam1",
                None,
                None,
                None,
                [],
                None,
                frames_per_second=10.0,
                confidence_threshold=0.2,
                ocr_confidence_threshold=0.5,
                ocr_match_confidence=0.85,
                global_id_match_score=0.7,
                plate_text_mode="balanced",
                is_ocr_enabled=True,
                platform="anpr_object",
            )

        anyio.run(_run_failed_process)
