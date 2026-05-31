import time
from dataclasses import dataclass

import pytest

from triton_server.app import init_db
from triton_server.app import GlobalTrackRepository
from triton_server.app import (
    GlobalMatchResult,
    GlobalTrackingService,
    TrackFeatures,
    get_global_tracking_service,
)


@dataclass
class _Identity:
    global_id: str
    vehicle_class: str = "car"
    vehicle_color: str = "white"
    license_plate_text: str = "MP09AB1234"
    license_plate_confidence: float = 0.9
    avg_width: float = 100.0
    avg_height: float = 50.0
    aspect_ratio: float = 2.0


@dataclass
class _Association:
    id: int
    global_id: str
    match_score: float
    match_reason: str


class _FakeRepo:
    def __init__(self):
        self.identity = _Identity(global_id="gid_existing")
        self.identities = None
        self.association = None
        self.created_assocs = []
        self.updated_assocs = []
        self.upserts = []

    def find_association(self, **kwargs):
        return self.association

    def get_recent_identities(self, since_epoch):
        if self.identities is not None:
            return self.identities
        return [self.identity]

    def get_identity_by_global_id(self, global_id):
        if self.identity.global_id == global_id:
            return self.identity
        return None

    def upsert_identity(self, **kwargs):
        self.upserts.append(kwargs)
        return self.identity

    def create_association(self, **kwargs):
        self.created_assocs.append(kwargs)
        return kwargs

    def update_association(self, **kwargs):
        self.updated_assocs.append(kwargs)
        return kwargs


def test_plate_similarity_and_confusable_characters():
    assert GlobalTrackingService._plate_similarity("MP09AB1234", "MP09AB1234") == pytest.approx(1.0)
    assert GlobalTrackingService._plate_similarity("", "MP09AB1234") == pytest.approx(0.0)
    assert GlobalTrackingService._plate_similarity("DL12CS1256", "DL12(S1256") > 0.8
    assert GlobalTrackingService._chars_confusable("1", "I") is True
    assert GlobalTrackingService._chars_confusable("X", "Y") is False
    assert GlobalTrackingService._color_similarity("silver", "white") == pytest.approx(1.0)


def test_score_with_adaptive_weights_and_class_mismatch():
    features = TrackFeatures(
        local_track_id="1",
        camera_id="cam1",
        request_id="req1",
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
    )
    identity = _Identity(global_id="gid1")

    score, reasons = GlobalTrackingService._score(features, identity)
    assert score >= 0.9
    assert "class" in reasons

    mismatch_identity = _Identity(global_id="gid2", vehicle_class="truck")
    mismatch_score, mismatch_reasons = GlobalTrackingService._score(
        features, mismatch_identity
    )
    assert mismatch_score == pytest.approx(0.0)
    assert mismatch_reasons == []


def test_resolve_requires_configured_plate_similarity_for_plate_led_match():
    repo = _FakeRepo()
    repo.identity = _Identity(global_id="gid_existing", license_plate_text="MP09AB1234")
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="13",
        camera_id="cam_b",
        request_id="req_plate_threshold",
        vehicle_class="car",
        vehicle_color="silver",
        license_plate_text="MH12XY9876",
        license_plate_confidence=0.94,
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
        ocr_match_confidence=0.95,
        global_id_match_score=0.70,
    )

    result = service.resolve(features)

    assert result.global_id != "gid_existing"
    assert result.match_reason == "new_identity"


def test_resolve_creates_new_identity_when_no_match():
    repo = _FakeRepo()
    repo.identity = _Identity(global_id="gid_low", vehicle_class="truck")
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="10",
        camera_id="cam_a",
        request_id="req_a",
        vehicle_class="car",
        vehicle_color="blue",
        license_plate_text="XX00YY1234",
        bbox_width=120,
        bbox_height=60,
        aspect_ratio=2.0,
    )

    result = service.resolve(features)

    assert isinstance(result, GlobalMatchResult)
    assert result.global_id.startswith("gid_")
    assert result.match_reason == "new_identity"
    assert len(repo.created_assocs) == 1


def test_resolve_matches_existing_identity():
    repo = _FakeRepo()
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="11",
        camera_id="cam_b",
        request_id="req_b",
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
    )

    result = service.resolve(features)

    assert result.global_id == "gid_existing"
    assert result.match_score >= 0.7
    assert len(repo.created_assocs) == 1
    assert len(repo.upserts) == 1


def test_resolve_prefers_latest_high_confidence_identity_for_strong_plate_match():
    repo = _FakeRepo()
    older = _Identity(global_id="gid_old", license_plate_confidence=0.72)
    older.last_seen_epoch = time.time() - 20
    newer = _Identity(global_id="gid_new", license_plate_confidence=0.95)
    newer.last_seen_epoch = time.time() - 5
    repo.identities = [older, newer]
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="12",
        camera_id="cam_b",
        request_id="req_future",
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        license_plate_confidence=0.98,
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
    )

    result = service.resolve(features)

    assert result.global_id == "gid_new"
    assert repo.created_assocs[0]["global_id"] == "gid_new"
    assert repo.upserts[0]["global_id"] == "gid_new"


def test_refresh_existing_association_path():
    repo = _FakeRepo()
    repo.association = _Association(
        id=99, global_id="gid_existing", match_score=0.8, match_reason="class+plate"
    )
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="11",
        camera_id="cam_b",
        request_id="req_b",
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
    )

    result = service.resolve(features)

    assert result.global_id == "gid_existing"
    assert len(repo.updated_assocs) == 1


def test_refresh_existing_association_when_identity_missing_returns_existing_values():
    repo = _FakeRepo()
    repo.association = _Association(
        id=1, global_id="gid_missing", match_score=0.5, match_reason="existing"
    )
    repo.identity = _Identity(global_id="gid_other")
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(local_track_id="1", camera_id="cam", request_id="req")
    result = service.resolve(features)

    assert result.global_id == "gid_missing"
    assert result.match_score == pytest.approx(0.5)
    assert result.match_reason == "existing"


def test_score_with_no_available_signals_and_factory_helper():
    features = TrackFeatures()
    identity = _Identity(
        global_id="gid1",
        vehicle_class="",
        vehicle_color="",
        license_plate_text="",
        aspect_ratio=0.0,
    )

    score, reasons = GlobalTrackingService._score(features, identity)
    assert score == pytest.approx(0.0)
    assert reasons == []
    assert GlobalTrackingService._chars_confusable("A", "A") is True
    assert isinstance(get_global_tracking_service(), GlobalTrackingService)


def test_resolve_detections_assigns_or_skips_global_tracking():
    repo = _FakeRepo()
    service = GlobalTrackingService(repository=repo)

    detections = [
        {
            "name": "car",
            "track_id": "1",
            "bbox_xyxy": [0.0, 0.0, 100.0, 50.0],
            "color": "white",
            "ocr_text": "MP09AB1234",
            "ocr_confidence": 0.9,
        },
        {
            "name": "number_plate",
            "track_id": "2",
            "bbox_xyxy": [0.0, 0.0, 50.0, 20.0],
        },
    ]

    service.resolve_detections(
        detections,
        camera_id="cam_x",
        request_id="req_x",
        skip_classes={"number_plate"},
        ocr_match_confidence=0.85,
        global_id_match_score=0.70,
    )

    assert detections[0]["global_id"]
    assert detections[0]["match_score"] >= 0.0
    assert detections[1]["global_id"] == ""
    assert detections[1]["match_score"] == pytest.approx(0.0)


def test_resolve_bypasses_request_association_cache_when_disabled():
    repo = _FakeRepo()
    repo.association = _Association(
        id=7,
        global_id="gid_cached",
        match_score=0.9,
        match_reason="cached",
    )
    repo.identity = _Identity(global_id="gid_existing", license_plate_text="MP09AB1234")
    service = GlobalTrackingService(repository=repo)

    features = TrackFeatures(
        local_track_id="0",
        camera_id="",
        request_id="",
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        license_plate_confidence=0.95,
        bbox_width=100,
        bbox_height=50,
        aspect_ratio=2.0,
        allow_association_cache=False,
    )

    result = service.resolve(features)

    assert result.global_id == "gid_existing"
    assert repo.updated_assocs == []
    assert repo.created_assocs == []


def test_resolve_detections_no_association_creation_when_cache_disabled():
    repo = _FakeRepo()
    service = GlobalTrackingService(repository=repo)

    detections = [
        {
            "name": "car",
            "track_id": "0",
            "bbox_xyxy": [0.0, 0.0, 100.0, 50.0],
            "color": "white",
            "ocr_text": "MP09AB1234",
            "ocr_confidence": 0.9,
        }
    ]

    service.resolve_detections(
        detections,
        camera_id="",
        request_id="",
        skip_classes={"number_plate"},
        ocr_match_confidence=0.85,
        global_id_match_score=0.70,
        allow_association_cache=False,
    )

    assert detections[0]["global_id"]
    assert repo.created_assocs == []


def test_global_track_repository_helpers_and_crud():
    init_db()
    repo = GlobalTrackRepository()

    assert repo._plate_pattern_score("") == 0
    assert repo._plate_pattern_score("abc") == 1
    assert repo._plate_pattern_score("MP09AB1234") == 2

    assert repo._should_replace_plate("", 0.0, "MP09AB1234", 0.7) is True
    assert repo._should_replace_plate("MP09AB123", 0.5, "MP09AB1234", 0.2) is True
    assert repo._should_replace_plate("MP09AB1234", 0.5, "MP09AB1234", 0.9) is True
    assert repo._should_replace_plate("MP09AB1234", 0.9, "", 0.9) is False

    gid = f"gid_test_{int(time.time() * 1000)}"
    identity = repo.upsert_identity(
        global_id=gid,
        vehicle_class="car",
        vehicle_color="white",
        license_plate_text="MP09AB1234",
        license_plate_confidence=0.8,
        avg_width=100,
        avg_height=50,
        aspect_ratio=2.0,
        camera_id="cam_1",
    )
    assert identity.global_id == gid
    assert identity.sighting_count >= 1

    assoc = repo.create_association(
        global_id=gid,
        camera_id="cam_1",
        local_track_id="11",
        request_id="req_1",
        match_score=0.8,
        match_reason="class+plate",
    )
    assert assoc.id is not None

    found = repo.find_association(
        camera_id="cam_1", local_track_id="11", request_id="req_1"
    )
    assert found is not None

    updated = repo.update_association(
        association_id=assoc.id,
        match_score=0.9,
        match_reason="class+plate+color",
    )
    assert updated is not None
    assert updated.match_score == pytest.approx(0.9)

    missing_update = repo.update_association(
        association_id=999999,
        match_score=0.1,
        match_reason="none",
    )
    assert missing_update is None

    assert repo.get_identity_by_global_id(gid) is not None
    assert len(repo.get_recent_identities(time.time() - 3600)) >= 1
    assert len(repo.get_all_identities()) >= 1
