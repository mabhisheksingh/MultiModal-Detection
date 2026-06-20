import re
import time

import structlog
from sqlalchemy.orm import Session

from triton_server.app import GlobalIdentity, SessionLocal, TrackAssociation

logger = structlog.get_logger(__name__)
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{8,12}$")


class GlobalTrackRepository:
    """Data-access layer for global tracking identity store."""

    @staticmethod
    def _plate_pattern_score(text: str) -> int:
        plate = str(text or "").strip().upper()
        if not plate:
            return 0
        return 2 if _PLATE_PATTERN.fullmatch(plate) else 1

    @classmethod
    def _should_replace_plate(
        cls,
        current_text: str,
        current_confidence: float,
        new_text: str,
        new_confidence: float,
    ) -> bool:
        current = str(current_text or "").strip().upper()
        new = str(new_text or "").strip().upper()
        if not new:
            return False
        if not current:
            return True
        if len(new) > len(current):
            return True
        if len(new) == len(current) and float(new_confidence) > float(current_confidence):
            return True
        return cls._plate_pattern_score(new) > cls._plate_pattern_score(current)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _session() -> Session:
        return SessionLocal()

    # ------------------------------------------------------------------
    # GlobalIdentity CRUD
    # ------------------------------------------------------------------
    def get_all_identities(self) -> list[GlobalIdentity]:
        with self._session() as session:
            return session.query(GlobalIdentity).all()

    def get_recent_identities(self, since_epoch: float) -> list[GlobalIdentity]:
        with self._session() as session:
            return session.query(GlobalIdentity).filter(GlobalIdentity.last_seen_epoch >= since_epoch).all()

    def get_identity_by_global_id(self, global_id: str) -> GlobalIdentity | None:
        with self._session() as session:
            return session.query(GlobalIdentity).filter(GlobalIdentity.global_id == global_id).first()

    def upsert_identity(
        self,
        *,
        global_id: str,
        vehicle_class: str,
        vehicle_color: str,
        license_plate_text: str,
        license_plate_confidence: float,
        avg_width: float,
        avg_height: float,
        aspect_ratio: float,
        camera_id: str,
    ) -> GlobalIdentity:
        with self._session() as session:
            identity = session.query(GlobalIdentity).filter(GlobalIdentity.global_id == global_id).first()
            now = time.time()
            if identity is None:
                identity = GlobalIdentity(
                    global_id=global_id,
                    vehicle_class=vehicle_class,
                    vehicle_color=vehicle_color,
                    license_plate_text=license_plate_text,
                    license_plate_confidence=license_plate_confidence,
                    avg_width=avg_width,
                    avg_height=avg_height,
                    aspect_ratio=aspect_ratio,
                    last_camera_id=camera_id,
                    last_seen_epoch=now,
                    sighting_count=1,
                )
                session.add(identity)
            else:
                identity.vehicle_color = vehicle_color or identity.vehicle_color
                if self._should_replace_plate(
                    identity.license_plate_text,
                    identity.license_plate_confidence,
                    license_plate_text,
                    license_plate_confidence,
                ):
                    identity.license_plate_text = license_plate_text
                    identity.license_plate_confidence = float(license_plate_confidence or 0.0)
                identity.avg_width = avg_width or identity.avg_width
                identity.avg_height = avg_height or identity.avg_height
                identity.aspect_ratio = aspect_ratio or identity.aspect_ratio
                identity.last_camera_id = camera_id or identity.last_camera_id
                identity.last_seen_epoch = now
                identity.sighting_count += 1
            session.commit()
            session.refresh(identity)
            return identity

    # ------------------------------------------------------------------
    # TrackAssociation CRUD
    # ------------------------------------------------------------------
    def find_association(
        self,
        *,
        camera_id: str,
        local_track_id: str,
        request_id: str,
    ) -> TrackAssociation | None:
        with self._session() as session:
            return (
                session.query(TrackAssociation)
                .filter(
                    TrackAssociation.camera_id == camera_id,
                    TrackAssociation.local_track_id == local_track_id,
                    TrackAssociation.request_id == request_id,
                )
                .first()
            )

    def create_association(
        self,
        *,
        global_id: str,
        camera_id: str,
        local_track_id: str,
        request_id: str,
        match_score: float,
        match_reason: str,
    ) -> TrackAssociation:
        with self._session() as session:
            assoc = TrackAssociation(
                global_id=global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                request_id=request_id,
                match_score=match_score,
                match_reason=match_reason,
                created_epoch=time.time(),
            )
            session.add(assoc)
            session.commit()
            session.refresh(assoc)
            return assoc

    def update_association(
        self,
        *,
        association_id: int,
        match_score: float,
        match_reason: str,
    ) -> TrackAssociation | None:
        with self._session() as session:
            assoc = session.query(TrackAssociation).filter(TrackAssociation.id == association_id).first()
            if assoc is None:
                return None

            assoc.match_score = match_score
            assoc.match_reason = match_reason
            session.commit()
            session.refresh(assoc)
            return assoc
