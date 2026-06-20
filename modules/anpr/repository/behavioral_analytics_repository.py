"""Repository layer for behavioral analytics persistence."""

import time
from typing import Any

from sqlalchemy.orm import Session

from triton_server.app import BehavioralEvent, CameraConfig, SessionLocal, VisitHistory


class BehavioralAnalyticsRepository:
    """CRUD operations for camera config, behavioral events, and visit history."""

    def __init__(self, session: Session | None = None):
        self.session = session

    def _get_session(self) -> Session:
        return self.session or SessionLocal()

    # ------------------------------------------------------------------
    # CameraConfig operations
    # ------------------------------------------------------------------
    def get_camera_config(self, camera_id: str) -> CameraConfig | None:
        """Fetch camera config by camera_id."""
        session = self._get_session()
        try:
            return session.query(CameraConfig).filter(CameraConfig.camera_id == camera_id).first()
        finally:
            if not self.session:
                session.close()

    def upsert_camera_config(
        self,
        camera_id: str,
        pixels_per_meter: float | None = None,
        repeat_visit_threshold: int = 3,
        linger_threshold_ms: int = 30000,
        sensitive_zone_types: str = "sensitive,restricted",
        min_behavior_score: float = 0.6,
    ) -> CameraConfig:
        """Create or update camera config."""
        session = self._get_session()
        try:
            existing = session.query(CameraConfig).filter(CameraConfig.camera_id == camera_id).first()

            now = time.time()
            if existing:
                existing.pixels_per_meter = pixels_per_meter
                existing.repeat_visit_threshold = repeat_visit_threshold
                existing.linger_threshold_ms = linger_threshold_ms
                existing.sensitive_zone_types = sensitive_zone_types
                existing.min_behavior_score = min_behavior_score
                existing.updated_at = now
                session.commit()
                session.refresh(existing)
                return existing
            else:
                new_config = CameraConfig(
                    camera_id=camera_id,
                    pixels_per_meter=pixels_per_meter,
                    repeat_visit_threshold=repeat_visit_threshold,
                    linger_threshold_ms=linger_threshold_ms,
                    sensitive_zone_types=sensitive_zone_types,
                    min_behavior_score=min_behavior_score,
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_config)
                session.commit()
                session.refresh(new_config)
                return new_config
        finally:
            if not self.session:
                session.close()

    def get_camera_config_as_dict(self, camera_id: str) -> dict[str, Any] | None:
        """Get camera config as dict for behavioral service."""
        config = self.get_camera_config(camera_id)
        if not config:
            return None

        sensitive_types = [t.strip() for t in config.sensitive_zone_types.split(",") if t.strip()]

        return {
            "repeat_visit_threshold": config.repeat_visit_threshold,
            "linger_threshold_ms": config.linger_threshold_ms,
            "sensitive_zone_types": sensitive_types,
            "min_behavior_score": config.min_behavior_score,
        }

    # ------------------------------------------------------------------
    # VisitHistory operations
    # ------------------------------------------------------------------
    def record_visit(
        self,
        global_id: str,
        camera_id: str,
        request_id: str,
        visit_timestamp: float,
        zone_id: str = "",
        dwell_duration_ms: int = 0,
    ) -> VisitHistory:
        """Record a visit to the database."""
        session = self._get_session()
        try:
            visit = VisitHistory(
                global_id=global_id,
                camera_id=camera_id,
                request_id=request_id,
                visit_timestamp=visit_timestamp,
                zone_id=zone_id,
                dwell_duration_ms=dwell_duration_ms,
            )
            session.add(visit)
            session.commit()
            session.refresh(visit)
            return visit
        finally:
            if not self.session:
                session.close()

    def get_visit_count(
        self,
        global_id: str,
        camera_id: str | None = None,
        since_timestamp: float | None = None,
    ) -> int:
        """Get total visit count for a global_id, optionally filtered by camera and time."""
        session = self._get_session()
        try:
            query = session.query(VisitHistory).filter(VisitHistory.global_id == global_id)
            if camera_id:
                query = query.filter(VisitHistory.camera_id == camera_id)
            if since_timestamp:
                query = query.filter(VisitHistory.visit_timestamp >= since_timestamp)
            return query.count()
        finally:
            if not self.session:
                session.close()

    def get_visit_timestamps(
        self,
        global_id: str,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[float]:
        """Get recent visit timestamps for a global_id."""
        session = self._get_session()
        try:
            query = session.query(VisitHistory.visit_timestamp).filter(VisitHistory.global_id == global_id)
            if camera_id:
                query = query.filter(VisitHistory.camera_id == camera_id)

            results = query.order_by(VisitHistory.visit_timestamp.desc()).limit(limit).all()
            return [r[0] for r in results]
        finally:
            if not self.session:
                session.close()

    # ------------------------------------------------------------------
    # BehavioralEvent operations
    # ------------------------------------------------------------------
    def record_behavioral_event(
        self,
        global_id: str,
        camera_id: str,
        request_id: str,
        behavior_label: str,
        behavior_score: float,
        visit_count: int,
        dwell_time_ms: int,
        is_sensitive_zone: bool,
        zone_id: str,
        zone_type: str,
        detected_at: float,
        frame_id: int,
    ) -> BehavioralEvent:
        """Record a detected behavioral pattern."""
        session = self._get_session()
        try:
            event = BehavioralEvent(
                global_id=global_id,
                camera_id=camera_id,
                request_id=request_id,
                behavior_label=behavior_label,
                behavior_score=behavior_score,
                visit_count=visit_count,
                dwell_time_ms=dwell_time_ms,
                is_sensitive_zone=1 if is_sensitive_zone else 0,
                zone_id=zone_id,
                zone_type=zone_type,
                detected_at=detected_at,
                frame_id=frame_id,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event
        finally:
            if not self.session:
                session.close()

    def get_behavioral_events(
        self,
        global_id: str | None = None,
        camera_id: str | None = None,
        behavior_label: str | None = None,
        since_timestamp: float | None = None,
        limit: int = 100,
    ) -> list[BehavioralEvent]:
        """Query behavioral events with filters."""
        session = self._get_session()
        try:
            query = session.query(BehavioralEvent)

            if global_id:
                query = query.filter(BehavioralEvent.global_id == global_id)
            if camera_id:
                query = query.filter(BehavioralEvent.camera_id == camera_id)
            if behavior_label:
                query = query.filter(BehavioralEvent.behavior_label == behavior_label)
            if since_timestamp:
                query = query.filter(BehavioralEvent.detected_at >= since_timestamp)

            return query.order_by(BehavioralEvent.detected_at.desc()).limit(limit).all()
        finally:
            if not self.session:
                session.close()
