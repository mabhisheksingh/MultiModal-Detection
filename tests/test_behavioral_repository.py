import time

from triton_server.app import BehavioralAnalyticsRepository, init_db


def test_camera_config_crud_and_dict_conversion():
    init_db()
    repo = BehavioralAnalyticsRepository()

    camera_id = f"cam_cfg_{int(time.time() * 1000)}"
    assert repo.get_camera_config(camera_id) is None
    assert repo.get_camera_config_as_dict(camera_id) is None

    created = repo.upsert_camera_config(
        camera_id=camera_id,
        pixels_per_meter=12.5,
        repeat_visit_threshold=4,
        linger_threshold_ms=45000,
        sensitive_zone_types="restricted,vip",
        min_behavior_score=0.7,
    )
    assert created.camera_id == camera_id

    cfg = repo.get_camera_config(camera_id)
    assert cfg is not None
    assert cfg.pixels_per_meter == 12.5

    as_dict = repo.get_camera_config_as_dict(camera_id)
    assert as_dict is not None
    assert as_dict["repeat_visit_threshold"] == 4
    assert as_dict["linger_threshold_ms"] == 45000
    assert as_dict["sensitive_zone_types"] == ["restricted", "vip"]

    updated = repo.upsert_camera_config(
        camera_id=camera_id,
        pixels_per_meter=20.0,
        repeat_visit_threshold=2,
        linger_threshold_ms=30000,
        sensitive_zone_types="sensitive,restricted",
        min_behavior_score=0.6,
    )
    assert updated.pixels_per_meter == 20.0


def test_visit_history_crud_queries():
    init_db()
    repo = BehavioralAnalyticsRepository()

    gid = f"gid_visit_{int(time.time() * 1000)}"
    camera_id = "cam_visit"
    now = time.time()

    repo.record_visit(
        global_id=gid,
        camera_id=camera_id,
        request_id="req_1",
        visit_timestamp=now - 10,
        zone_id="zone_a",
        dwell_duration_ms=1500,
    )
    repo.record_visit(
        global_id=gid,
        camera_id=camera_id,
        request_id="req_2",
        visit_timestamp=now,
        zone_id="zone_b",
        dwell_duration_ms=2500,
    )

    assert repo.get_visit_count(gid) >= 2
    assert repo.get_visit_count(gid, camera_id=camera_id) >= 2
    assert repo.get_visit_count(gid, since_timestamp=now - 1) >= 1

    timestamps = repo.get_visit_timestamps(gid, camera_id=camera_id, limit=10)
    assert len(timestamps) >= 2
    assert timestamps[0] >= timestamps[1]


def test_behavioral_event_crud_queries():
    init_db()
    repo = BehavioralAnalyticsRepository()

    gid = f"gid_behavior_{int(time.time() * 1000)}"
    camera_id = "cam_behavior"
    now = time.time()

    repo.record_behavioral_event(
        global_id=gid,
        camera_id=camera_id,
        request_id="req_1",
        behavior_label="repeat_visit",
        behavior_score=0.7,
        visit_count=3,
        dwell_time_ms=1200,
        is_sensitive_zone=False,
        zone_id="zone_a",
        zone_type="entry",
        detected_at=now - 10,
        frame_id=10,
    )
    repo.record_behavioral_event(
        global_id=gid,
        camera_id=camera_id,
        request_id="req_2",
        behavior_label="repeated_presence",
        behavior_score=0.9,
        visit_count=5,
        dwell_time_ms=6000,
        is_sensitive_zone=True,
        zone_id="zone_b",
        zone_type="restricted",
        detected_at=now,
        frame_id=20,
    )

    events = repo.get_behavioral_events(global_id=gid, camera_id=camera_id, limit=10)
    assert len(events) >= 2

    filtered = repo.get_behavioral_events(
        global_id=gid,
        camera_id=camera_id,
        behavior_label="repeated_presence",
        since_timestamp=now - 1,
        limit=10,
    )
    assert len(filtered) >= 1
    assert filtered[0].behavior_label == "repeated_presence"
