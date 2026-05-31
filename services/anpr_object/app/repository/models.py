from sqlalchemy import Column, Integer, Float, String, Index
from triton_server.app import Base


class GlobalIdentity(Base):
    """Canonical record for a unique real-world object seen across cameras/videos."""

    __tablename__ = "global_identities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(String, unique=True, nullable=False, index=True)
    vehicle_class = Column(String, nullable=False, default="")
    vehicle_color = Column(String, nullable=False, default="")
    license_plate_text = Column(String, nullable=False, default="")
    license_plate_confidence = Column(Float, nullable=False, default=0.0)
    avg_width = Column(Float, nullable=False, default=0.0)
    avg_height = Column(Float, nullable=False, default=0.0)
    aspect_ratio = Column(Float, nullable=False, default=0.0)
    last_camera_id = Column(String, nullable=False, default="")
    last_seen_epoch = Column(Float, nullable=False, default=0.0)
    sighting_count = Column(Integer, nullable=False, default=1)


class TrackAssociation(Base):
    """Maps a per-camera/video local track to a global identity."""

    __tablename__ = "track_associations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False, default="")
    local_track_id = Column(String, nullable=False)
    request_id = Column(String, nullable=False, default="")
    match_score = Column(Float, nullable=False, default=0.0)
    match_reason = Column(String, nullable=False, default="")
    created_epoch = Column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_assoc_camera_track", "camera_id", "local_track_id", "request_id"),
    )


class CameraConfig(Base):
    """Per-camera behavior configuration and calibration settings."""

    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, unique=True, nullable=False, index=True)
    pixels_per_meter = Column(Float, nullable=True)
    repeat_visit_threshold = Column(Integer, nullable=False, default=3)
    linger_threshold_ms = Column(Integer, nullable=False, default=30000)
    sensitive_zone_types = Column(
        String, nullable=False, default="sensitive,restricted"
    )
    min_behavior_score = Column(Float, nullable=False, default=0.6)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)


class BehavioralEvent(Base):
    """Detected behavioral patterns with timestamps for audit and analytics."""

    __tablename__ = "behavioral_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    request_id = Column(String, nullable=False)
    behavior_label = Column(String, nullable=False, index=True)
    behavior_score = Column(Float, nullable=False)
    visit_count = Column(Integer, nullable=False, default=1)
    dwell_time_ms = Column(Integer, nullable=False, default=0)
    is_sensitive_zone = Column(Integer, nullable=False, default=0)
    zone_id = Column(String, nullable=False, default="")
    zone_type = Column(String, nullable=False, default="")
    detected_at = Column(Float, nullable=False, index=True)
    frame_id = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_behavior_global_camera", "global_id", "camera_id"),
        Index("ix_behavior_label_time", "behavior_label", "detected_at"),
    )


class VisitHistory(Base):
    """Cross-session visit tracking per global_id and camera."""

    __tablename__ = "visit_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    request_id = Column(String, nullable=False)
    visit_timestamp = Column(Float, nullable=False, index=True)
    zone_id = Column(String, nullable=False, default="")
    dwell_duration_ms = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_visit_global_camera", "global_id", "camera_id"),
        Index("ix_visit_time", "visit_timestamp"),
    )
