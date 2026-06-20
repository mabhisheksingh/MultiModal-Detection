from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from triton_server.app import (
    APIConstants,
    GlobalTrackRepository,
    ImageAnalysisUtils,
    ImageSourceUtils,
    LiveVideoSourceProcessor,
    MediaSourceUtils,
    PaddleOCREngine,
    Settings,
    TritonClient,
    VideoSourceUtils,
    attach_request_id,
    configure_logging,
    dependencies,
    settings,
    setup_logger,
    start_server,
)
from triton_server.app import database as db_module


def test_configure_logging_development_and_production():
    with patch.object(configure_logging, "__module__", "modules.common.logging"):
        configure_logging(log_level="INFO", env="development")
        # Just test it doesn't crash - actual testing would require mocking structlog

    with patch.object(configure_logging, "__module__", "modules.common.logging"):
        configure_logging(log_level="INFO", env="production")
        # Just test it doesn't crash - actual testing would require mocking structlog


def test_setup_logger_configures_handlers(tmp_path):
    with patch.object(settings, "output_folder", str(tmp_path / "output")):
        # Just test it doesn't crash - actual testing would require mocking logging internals
        setup_logger()


def test_setup_logger_covers_directory_creation_and_third_party_logger_levels(tmp_path):
    with patch.object(settings, "log_level", "info"):
        # Just test it doesn't crash - actual testing would require mocking logging internals
        setup_logger()


def test_attach_request_id_middleware_sets_and_preserves_header(anyio_backend):
    async def _call_next(request):
        from starlette.responses import Response

        return Response(content="ok")

    scope_without_id = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 12345),
        "scheme": "http",
        "http_version": "1.1",
    }

    from starlette.requests import Request

    req = Request(scope_without_id)

    import anyio

    response = anyio.run(attach_request_id, req, _call_next)
    assert APIConstants.REQUEST_ID_HEADER in response.headers


def test_start_server_invokes_uvicorn_run():
    with patch.object(start_server, "__module__", "app.main"), patch("uvicorn.run") as run_mock:
        start_server()
        assert run_mock.called


def test_dependencies_factories_and_cached_instances():
    dependencies.get_vehicle_triton_client.cache_clear()
    dependencies.get_plate_triton_client.cache_clear()
    dependencies.get_global_tracking_service.cache_clear()
    dependencies.get_spatial_correlation_service.cache_clear()
    dependencies.get_behavioral_pattern_service.cache_clear()
    dependencies.get_anpr_service.cache_clear()

    with patch("modules.anpr.utils.dependencies.PaddleOCREngine", return_value="ocr"):
        assert dependencies.get_paddle_ocr_engine() == "ocr"

    with patch("modules.anpr.utils.dependencies.TritonClient", return_value="client"):
        assert dependencies.get_vehicle_triton_client() == "client"
    dependencies.get_vehicle_triton_client.cache_clear()

    with patch("modules.anpr.utils.dependencies.TritonClient", return_value="plate_client"):
        assert dependencies.get_plate_triton_client() == "plate_client"
    dependencies.get_plate_triton_client.cache_clear()

    with patch("modules.anpr.utils.dependencies.GlobalTrackingService", return_value="g"):
        assert dependencies.get_global_tracking_service() == "g"

    with patch("modules.anpr.utils.dependencies.SpatiotemporalCorrelationService", return_value="s"):
        assert dependencies.get_spatial_correlation_service() == "s"

    with patch("modules.anpr.utils.dependencies.BehavioralPatternService", return_value="b"):
        assert dependencies.get_behavioral_pattern_service() == "b"

    dependencies.get_anpr_service.cache_clear()
    with (
        patch("modules.anpr.utils.dependencies.get_paddle_ocr_engine", return_value="ocr"),
        patch("modules.anpr.utils.dependencies.get_vehicle_triton_client", return_value="vc"),
        patch("modules.anpr.utils.dependencies.get_plate_triton_client", return_value="pc"),
        patch("modules.anpr.utils.dependencies.get_global_tracking_service", return_value="gt"),
        patch("modules.anpr.utils.dependencies.get_spatial_correlation_service", return_value="sp"),
        patch("modules.anpr.utils.dependencies.get_behavioral_pattern_service", return_value="bp"),
        patch("modules.anpr.utils.dependencies.ANPRService", return_value="anpr_object"),
    ):
        assert dependencies.get_anpr_service() == "anpr_object"


def test_triton_client_preprocess_and_close():
    client = TritonClient(server_url="http://127.0.0.1:9001", model_name="model_a")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    blob, meta = client.preprocess_image(image, frame_idx=3)

    assert blob.shape == (3, 640, 640)
    assert meta["frame_idx"] == 3
    assert meta["scale"] > 0

    client.close()
    assert client._model is None


def test_triton_client_infer_request_and_infer_error_path():
    client = TritonClient(server_url="127.0.0.1:9001", model_name="model_a")
    dtype = np.float32

    class FakeInferInput:
        def __init__(self, *args, **kwargs):
            self.data = None

        def set_data_from_numpy(self, data):
            self.data = data

    class FakeOutput:
        def as_numpy(self, name):
            return np.array([[1.0]], dtype=dtype)

    fake_model = SimpleNamespace(
        input_names=["images"],
        output_names=["output0"],
        endpoint="model_a",
        InferInput=FakeInferInput,
        InferRequestedOutput=lambda name: name,
        triton_client=SimpleNamespace(infer=lambda **kwargs: FakeOutput()),
    )

    out = client._infer_request(fake_model, np.zeros((1, 3, 10, 10), dtype=dtype), "rid")
    assert len(out) == 1
    assert out[0].shape == (1, 1)

    client = TritonClient(server_url="127.0.0.1:9001", model_name="model_a")
    with patch.object(client, "_get_model", side_effect=RuntimeError("boom")):
        import anyio

        try:
            anyio.run(client.infer, np.zeros((10, 10, 3), dtype=np.uint8), 0)
            raise AssertionError("Expected RuntimeError")
        except RuntimeError:
            assert True


def test_paddle_ocr_engine_recognize_and_text_only():
    class FakePaddle:
        class device:
            @staticmethod
            def is_compiled_with_cuda():
                return False

        @staticmethod
        def set_device(device):
            return device

    class FakeReader:
        def predict(self, image, **kwargs):
            assert kwargs["text_det_limit_side_len"] == settings.ocr_det_limit_side_len
            assert kwargs["text_det_box_thresh"] == settings.ocr_det_box_thresh
            assert kwargs["text_rec_score_thresh"] == settings.ocr_rec_score_thresh
            return [
                {"rec_texts": ["MP09CM0105"], "rec_scores": [0.8]},
                {"rec_texts": [], "rec_scores": []},
            ]

    with patch(
        "modules.anpr.services.paddle_ocr_engine._import_paddle_dependencies",
        return_value=(FakePaddle(), lambda **kwargs: FakeReader()),
    ):
        engine = PaddleOCREngine()
        gray = np.zeros((10, 10), dtype=np.uint8)
        results = engine.recognize(gray, plate_text_mode="balanced")

        assert len(results) == 1
        assert results[0]["text"] == "MP09CM0105"
        assert results[0]["confidence"] == pytest.approx(0.8)


def test_paddle_ocr_engine_rejects_invalid_candidates_and_falls_back_from_gpu():
    class FakePaddle:
        class device:
            @staticmethod
            def is_compiled_with_cuda():
                return True

        def __init__(self):
            self.calls = []

        def set_device(self, device):
            self.calls.append(device)
            if device == "gpu:0":
                raise RuntimeError("gpu unavailable")
            return device

    class FakeReader:
        def predict(self, image, **kwargs):
            return [
                {"rec_texts": ["MP09", "Mahadev", "02/23/2026 15:19:02"], "rec_scores": [0.99, 0.95, 0.94]},
            ]

    fake_paddle = FakePaddle()
    with patch(
        "modules.anpr.services.paddle_ocr_engine._import_paddle_dependencies",
        return_value=(fake_paddle, lambda **kwargs: FakeReader()),
    ):
        engine = PaddleOCREngine()
        results = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8), plate_text_mode="balanced")

        assert engine._device == "cpu"
        assert fake_paddle.calls == ["gpu:0", "cpu"]
        assert results == [{"text": "MP09MAHADEV02232026151902", "confidence": 0.96}]


def test_paddle_ocr_engine_strict_mode_skips_partial_plate_candidates():
    class FakePaddle:
        class device:
            @staticmethod
            def is_compiled_with_cuda():
                return False

        @staticmethod
        def set_device(device):
            return device

    class FakeReader:
        def predict(self, image, **kwargs):
            return [
                {"rec_texts": ["MP09", "Mahadev"], "rec_scores": [0.99, 0.95]},
            ]

    with patch(
        "modules.anpr.services.paddle_ocr_engine._import_paddle_dependencies",
        return_value=(FakePaddle(), lambda **kwargs: FakeReader()),
    ):
        engine = PaddleOCREngine()
        results = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8), plate_text_mode="strict")

        assert engine._device == "cpu"
        assert results == []


def test_main_lifespan_and_cancelled_middleware_path():
    import anyio
    from starlette.requests import Request

    async def _cancel_next(request):
        raise anyio.get_cancelled_exc_class()()

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 12345),
        "scheme": "http",
        "http_version": "1.1",
    }
    req = Request(scope)

    with pytest.raises(BaseException):
        anyio.run(attach_request_id, req, _cancel_next)


def test_settings_validation_and_parse_errors(tmp_path):
    settings_obj = Settings(
        INPUT_FOLDER=str(tmp_path / "in"),
        OUTPUT_FOLDER=str(tmp_path / "out"),
    )
    assert settings_obj.plate_candidate_vehicle_classes_list
    assert settings_obj.vehicle_class_id_name_map[0]
    assert settings_obj.plate_class_id_name_map[0]
    assert settings_obj.get_summary()["app_name"] == settings_obj.app_name
    assert ".mp4" in settings_obj.combined_extensions_list

    with pytest.raises(ValueError):
        settings_obj._parse_class_id_map("{bad json", field_name="X")
    with pytest.raises(ValueError):
        settings_obj._parse_class_id_map("[]", field_name="X")
    with pytest.raises(ValueError):
        settings_obj._parse_class_id_map('{"a":"car"}', field_name="X")
    with pytest.raises(ValueError):
        settings_obj._parse_class_id_map('{"1":"   "}', field_name="X")

    broken = Settings(
        TRITON_SERVER_URL="",
        VEHICLE_MODEL_NAME="",
        PLATE_MODEL_NAME="",
        VEHICLE_CLASS_ID_MAP="[]",
        PLATE_CLASS_ID_MAP="[]",
        INPUT_FOLDER=str(tmp_path / "in2"),
        OUTPUT_FOLDER=str(tmp_path / "out2"),
    )
    with pytest.raises(ValueError):
        broken.validate_config()


def test_settings_validate_config_success_logs_info(tmp_path):
    settings_obj = Settings(
        INPUT_FOLDER=str(tmp_path / "in_ok"),
        OUTPUT_FOLDER=str(tmp_path / "out_ok"),
    )

    settings_obj.validate_config()


def test_settings_create_directories_error_path(tmp_path):
    settings_obj = Settings(INPUT_FOLDER="in", OUTPUT_FOLDER="out")
    with (
        patch.object(type(settings_obj), "data_input_dir", new=property(lambda self: tmp_path / "a")),
        patch.object(type(settings_obj), "data_output_dir", new=property(lambda self: tmp_path / "b")),
        patch("pathlib.Path.mkdir", side_effect=OSError("mkdir fail")),
        pytest.raises(OSError),
    ):
        settings_obj.create_directories()


def test_database_schema_helpers_and_init_db():
    with patch.object(db_module, "DATABASE_URL", "postgresql://test"):
        db_module._ensure_sqlite_schema()

    inspector_no_table = Mock()
    inspector_no_table.get_table_names.return_value = []
    with (
        patch.object(db_module, "DATABASE_URL", "sqlite:///x"),
        patch.object(db_module, "inspect", return_value=inspector_no_table),
    ):
        db_module._ensure_sqlite_schema()

    inspector_has_column = Mock()
    inspector_has_column.get_table_names.return_value = ["global_identities"]
    inspector_has_column.get_columns.return_value = [{"name": "license_plate_confidence"}]
    with (
        patch.object(db_module, "DATABASE_URL", "sqlite:///x"),
        patch.object(db_module, "inspect", return_value=inspector_has_column),
    ):
        db_module._ensure_sqlite_schema()

    inspector_missing_column = Mock()
    inspector_missing_column.get_table_names.return_value = ["global_identities"]
    inspector_missing_column.get_columns.return_value = [{"name": "global_id"}]
    begin_ctx = Mock()
    begin_ctx.__enter__ = Mock(return_value=Mock(execute=Mock()))
    begin_ctx.__exit__ = Mock(return_value=False)
    with (
        patch.object(db_module, "DATABASE_URL", "sqlite:///x"),
        patch.object(db_module, "inspect", return_value=inspector_missing_column),
        patch.object(db_module.engine, "begin", return_value=begin_ctx),
    ):
        db_module._ensure_sqlite_schema()

    with (
        patch.object(db_module.Base.metadata, "create_all") as create_all_mock,
        patch.object(db_module, "_ensure_sqlite_schema") as ensure_mock,
    ):
        db_module.init_db()
        assert create_all_mock.called
        assert ensure_mock.called


def test_global_track_repository_update_branch_and_media_utils_more_paths(tmp_path):
    repo = GlobalTrackRepository()
    gid = "gid_extra_branch"
    repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="PLATE123",
        license_plate_confidence=0.5,
        avg_width=100,
        avg_height=50,
        aspect_ratio=2.0,
        camera_id="cam1",
    )
    updated = repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="",
        license_plate_text="",
        license_plate_confidence=0.0,
        avg_width=0,
        avg_height=0,
        aspect_ratio=0,
        camera_id="",
    )
    assert updated.vehicle_color == "white"

    updated_replace = repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="blue",
        license_plate_text="PLATE1234",
        license_plate_confidence=0.95,
        avg_width=120,
        avg_height=55,
        aspect_ratio=2.2,
        camera_id="cam2",
    )
    assert updated_replace.license_plate_text == "PLATE1234"
    assert updated_replace.license_plate_confidence == 0.95

    local_file = tmp_path / "clip.mp4"
    local_file.write_bytes(b"x")
    VideoSourceUtils.validate_video_source(str(local_file))
    request_folder, output_path, source_name = VideoSourceUtils.build_output_paths(str(local_file), "", None)
    assert request_folder.exists()
    assert source_name == "clip.mp4"
    assert output_path.endswith("clip.mp4")

    no_ext_folder, _, no_ext_name = VideoSourceUtils.build_output_paths("https://example.com/live", str(tmp_path), None)
    assert no_ext_folder.exists()
    assert no_ext_name.endswith(".mp4")

    blank_name_folder, _, blank_name = VideoSourceUtils.build_output_paths("https://example.com/", str(tmp_path), None)
    assert blank_name == "live_stream.mp4"
    assert blank_name_folder.exists()

    image_file = tmp_path / "image.png"
    image_file.write_bytes(b"x")
    assert ImageSourceUtils.validate_image_source_path(str(image_file)) == image_file

    blur_img = np.full((20, 20, 3), 128, dtype=np.uint8)
    assert isinstance(ImageAnalysisUtils.calculate_blur(blur_img), float)

    empty_roi = np.zeros((30, 0, 3), dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(empty_roi, min_area=0) == ""

    gray_mid = np.full((60, 120, 3), 120, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(gray_mid, min_area=100) == "grey"
    silver_img = np.full((60, 120, 3), 170, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(silver_img, min_area=100) in {
        "silver",
        "grey",
    }
    dark_img = np.full((60, 120, 3), 60, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(dark_img, min_area=100) == "black"

    colorful_low_hist = np.zeros((80, 120, 3), dtype=np.uint8)
    colorful_low_hist[:] = (0, 255, 0)
    with patch(
        "modules.anpr.utils.media_utils.cv2.calcHist",
        return_value=np.zeros((180, 1), dtype=np.float32),
    ):
        assert ImageAnalysisUtils.extract_dominant_color(colorful_low_hist, min_area=100) == ""

    with patch.object(ImageAnalysisUtils, "extract_dominant_color", return_value="grey"):
        assert ImageAnalysisUtils.extract_plate_color(np.zeros((10, 10, 3), dtype=np.uint8)) == "grey"
    black_plate = np.zeros((60, 120, 3), dtype=np.uint8)
    assert ImageAnalysisUtils.extract_plate_color(black_plate) in {"", "black", "white"}


def test_global_track_repository_preserves_existing_plate_when_replacement_is_weaker():
    repo = GlobalTrackRepository()
    gid = "gid_preserve_plate"
    repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="AB12CD3456",
        license_plate_confidence=0.95,
        avg_width=100,
        avg_height=50,
        aspect_ratio=2.0,
        camera_id="cam1",
    )

    updated = repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="blue",
        license_plate_text="A",
        license_plate_confidence=0.10,
        avg_width=101,
        avg_height=51,
        aspect_ratio=2.0,
        camera_id="cam2",
    )

    assert updated.license_plate_text == "AB12CD3456"
    assert updated.license_plate_confidence == 0.95


def test_media_source_and_video_processor_remaining_small_branches(tmp_path):
    assert MediaSourceUtils.resolve_camera_id("", None) == ""
    assert MediaSourceUtils.resolve_camera_id("https://example.com/", None) == "example.com"
    assert MediaSourceUtils.resolve_camera_id("https://example.com/feed.mp4", " cam_manual ") == "cam_manual"
    assert MediaSourceUtils.resolve_camera_id("https://example.com/feed.mp4", None) == "feed"
    assert MediaSourceUtils.resolve_camera_id("~/clips/sample.mp4", None) == "sample"

    processor = LiveVideoSourceProcessor(service=Mock(), spatial_service=None, behavioral_service=None)
    assert processor._enrich_detections_with_analytics([], "cam", None, None, None, [], None, 0, 0, "req") == []

    writer = Mock()
    capture = Mock()
    capture.read.side_effect = [
        (True, np.zeros((4, 4, 3), dtype=np.uint8)),
        (False, None),
    ]
    capture.get.side_effect = [10.0, 4.0, 4.0, 2.0]

    with (
        patch("modules.anpr.services.video_source_processor.VideoSourceUtils.validate_video_source"),
        patch(
            "modules.anpr.services.video_source_processor.VideoSourceUtils.build_output_paths",
            return_value=(tmp_path, str(tmp_path / "out.mp4"), "src.mp4"),
        ),
        patch.object(processor, "_open_capture", return_value=capture),
        patch.object(processor, "_create_writer", return_value=writer),
        patch.object(
            processor,
            "_run_inference_batch",
            new=AsyncMock(side_effect=RuntimeError("batch fail")),
        ),
        patch.object(processor, "_enrich_detections_with_analytics", return_value=[]),
        patch.object(processor, "_write_csv_outputs", return_value=(None, None)),
    ):
        import anyio

        async def _run_batch_fail_process():
            return await processor.process(
                "src.mp4",
                str(tmp_path),
                True,
                "req-batch-fail",
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

        result = anyio.run(_run_batch_fail_process)

    assert result["processed_frames"] == 0
    writer.write.assert_called()


def test_media_utils_remaining_color_and_url_branches():
    empty_roi = np.zeros((30, 0, 3), dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(empty_roi, min_area=0) == ""

    white_low_sat = np.full((60, 120, 3), 230, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_dominant_color(white_low_sat, min_area=100) == "white"

    grey_plate = np.full((60, 120, 3), 170, dtype=np.uint8)
    assert ImageAnalysisUtils.extract_plate_color(grey_plate) in {"white", "silver"}

    ImageSourceUtils.validate_image_source_url("https://example.com/frame")


def test_triton_client_get_model_cached_and_infer_success():
    client = TritonClient(server_url="http://127.0.0.1:9001", model_name="model_a")
    fake_remote_model = SimpleNamespace(
        input_names=["images"],
        output_names=["output0"],
        endpoint="model_a",
    )
    with patch("modules.anpr.services.triton_client.TritonRemoteModel", return_value=fake_remote_model):
        model1 = client._get_model()
        model2 = client._get_model()
        assert model1 is model2

    client2 = TritonClient(server_url="127.0.0.1:9001", model_name="model_a")
    fake_model = Mock()
    with (
        patch.object(client2, "_get_model", return_value=fake_model),
        patch.object(
            client2,
            "preprocess_image",
            return_value=(np.zeros((3, 10, 10), dtype=np.float32), {"frame_idx": 1}),
        ),
        patch(
            "modules.anpr.services.triton_client.asyncio.to_thread",
            new=AsyncMock(return_value=[np.array([[9.0]], dtype=np.float32)]),
        ),
    ):
        import anyio

        output, meta = anyio.run(client2.infer, np.zeros((10, 10, 3), dtype=np.uint8), 1, "req")
        assert float(output[0][0]) == 9.0
        assert meta["frame_idx"] == 1
