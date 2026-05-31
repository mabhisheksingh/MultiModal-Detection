from triton_server.app import SourceMetadata
from triton_server.app import AnalyticsUtils
from triton_server.app import RequestTraceUtils, model_dump_compat


class _DictLike:
    def __iter__(self):
        return iter({"x": 1, "y": 2}.items())


def test_request_trace_utils_builds_and_skips_request_ids():
    assert RequestTraceUtils.build_triton_request_id(None, "cam 1/source", 7) is None
    assert (
        RequestTraceUtils.build_triton_request_id("req_1", "cam 1/source", 7)
        == "req_1:cam_1_source:7"
    )


def test_model_dump_compat_handles_none_model_and_dict_like_objects():
    assert model_dump_compat(None) == {}

    source = SourceMetadata(url="https://example.com/video.mp4", lat=12.3, lon=45.6)
    dumped = model_dump_compat(source)
    assert dumped["url"] == source.url
    assert dumped["lat"] == 12.3
    assert dumped["lon"] == 45.6

    assert model_dump_compat(_DictLike()) == {"x": 1, "y": 2}


def test_analytics_utils_remaining_small_branches():
    assert AnalyticsUtils.normalize_bbox_center([0.0, 0.0, 4.0, 8.0], 0, 0) == (
        2.0,
        4.0,
    )
    assert AnalyticsUtils.compute_pixel_speed((0.0, 0.0), (4.0, 3.0), 0) == 0.0
    assert "normal" in AnalyticsUtils.get_behavior_label_catalog()
    assert AnalyticsUtils.build_vehicle_episodes([]) == []


def test_analytics_direction_and_orientation_small_branches():
    assert AnalyticsUtils.direction_label_from_vector([0.0, 0.0]) == "stationary"
    assert AnalyticsUtils.direction_label_from_vector([0.2, 0.1]) == "left_to_right"
    assert AnalyticsUtils.direction_label_from_vector([0.1, -0.3]) == "towards_top"
    assert AnalyticsUtils.orientation_label_from_motion([0.0, 0.0]) == "stationary"
    assert AnalyticsUtils.orientation_label_from_motion([0.1, -0.3]) == "receding"
    assert AnalyticsUtils.orientation_label_from_motion([0.3, 0.1]) == "left_to_right"


def test_source_metadata_accepts_valid_numeric_fields():
    source = SourceMetadata(
        url="https://example.com/video.mp4",
        lat=12.3,
        lon=45.6,
        pixels_per_meter=20.0,
    )

    assert source.lat == 12.3
    assert source.lon == 45.6
    assert source.pixels_per_meter == 20.0
