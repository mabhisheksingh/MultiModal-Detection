"""Cross-camera / cross-video global identity assignment.

Matching uses vehicle visual features (class, color, dimensions) and
fuzzy license-plate text comparison. When some signals are absent their
weight is redistributed so the remaining signals can still contribute to
matching.

The service is designed to run after per-camera local tracking assigns a
``local_track_id``. Matched identities are refreshed over time so better
plate text observations can replace weaker partial OCR reads.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import structlog

from triton_server.app import settings
from triton_server.app import GlobalTrackRepository

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Match thresholds
# ---------------------------------------------------------------------------
MATCH_ACCEPT_THRESHOLD = 0.70

# Base feature weights (sum = 1.0).  When a signal is *absent* on both
# sides its budget is redistributed proportionally to the remaining signals.
W_CLASS = 0.20
W_COLOR = 0.10
W_DIMENSION = 0.15
W_PLATE = 0.55

# Characters frequently confused by OCR engines.  Each set is a group of
# visually similar glyphs; substitution between members costs less.
_PLATE_CONFUSION_GROUPS = (
    {"0", "O", "Q", "D"},
    {"1", "I", "L", "T", "7"},
    {"2", "Z"},
    {"5", "S", "$"},
    {"6", "G"},
    {"8", "B"},
    {"C", "(", "{"},
    {"4", "A"},
)

# Pre-computed lookup for O(1) confusion check
_CONFUSION_LOOKUP: Dict[str, set] = {}
for _grp in _PLATE_CONFUSION_GROUPS:
    for _ch in _grp:
        _CONFUSION_LOOKUP.setdefault(_ch, set()).update(_grp - {_ch})


@dataclass
class TrackFeatures:
    """Lightweight carrier for features extracted from a local track.

    ``license_plate_text`` and ``license_plate_confidence`` are used both
    for fuzzy matching and for deciding whether a newly observed plate text
    should replace the stored identity plate text.
    """

    local_track_id: str = ""
    camera_id: str = ""
    request_id: str = ""
    vehicle_class: str = ""
    vehicle_color: str = ""
    license_plate_text: str = ""
    license_plate_confidence: float = 0.0
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    aspect_ratio: float = 0.0
    ocr_match_confidence: float = 0.85
    global_id_match_score: float = 0.70
    allow_association_cache: bool = True


@dataclass
class GlobalMatchResult:
    """Result of a global-ID resolution attempt."""

    global_id: str = ""
    match_score: float = 0.0
    match_reason: str = ""


class GlobalTrackingService:
    """Core logic for assigning a cross-video global identity.

    Keeps state in the repository-backed global identity store.

    Existing local-track associations are refreshed on subsequent frames so
    improved OCR, color, and dimension signals can update both the identity
    record and the stored association score/reason.
    """

    def __init__(self, repository: Optional[GlobalTrackRepository] = None) -> None:
        self.repo = repository or GlobalTrackRepository()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, features: TrackFeatures) -> GlobalMatchResult:
        """Resolve a local track to a global identity.

        1.  If we already mapped this (camera, local_track, request) → refresh
            the existing association and identity using the newest features.
        2.  Otherwise score all recent identities, pick the best match, or
            create a new identity.
        """
        # Fast path — only for stable local-track flows (e.g. video tracking)
        if features.allow_association_cache:
            existing = self.repo.find_association(
                camera_id=features.camera_id,
                local_track_id=features.local_track_id,
                request_id=features.request_id,
            )
            if existing is not None:
                return self._refresh_existing_association(existing, features)

        candidate_since_epoch = time.time() - settings.global_track_lookback_seconds
        candidates = self.repo.get_recent_identities(candidate_since_epoch)
        best_score = 0.0
        best_id: Optional[str] = None
        best_reasons: List[str] = []
        best_plate_confidence = -1.0
        best_last_seen_epoch = 0.0
        match_threshold = max(0.0, min(features.global_id_match_score, 1.0))
        plate_threshold = max(0.0, min(features.ocr_match_confidence, 1.0))

        for identity in candidates:
            score, reasons = self._score(features, identity)
            plate_similarity = 0.0
            if features.license_plate_text and identity.license_plate_text:
                plate_similarity = self._plate_similarity(
                    features.license_plate_text,
                    identity.license_plate_text,
                )
            if features.license_plate_text and identity.license_plate_text:
                if plate_similarity < plate_threshold:
                    continue
            identity_plate_confidence = float(
                getattr(identity, "license_plate_confidence", 0.0) or 0.0
            )
            identity_last_seen_epoch = float(
                getattr(identity, "last_seen_epoch", 0.0) or 0.0
            )
            should_replace = score > best_score
            if not should_replace and score >= match_threshold and best_id is not None:
                same_score = abs(score - best_score) <= 1e-6
                strong_plate_match = plate_similarity >= plate_threshold
                if same_score and strong_plate_match:
                    if identity_plate_confidence > best_plate_confidence:
                        should_replace = True
                    elif (
                        abs(identity_plate_confidence - best_plate_confidence) <= 1e-6
                        and identity_last_seen_epoch > best_last_seen_epoch
                    ):
                        should_replace = True
                elif strong_plate_match and "plate" in reasons and "plate" not in best_reasons:
                    should_replace = True
            if should_replace:
                best_score = score
                best_id = identity.global_id
                best_reasons = reasons
                best_plate_confidence = identity_plate_confidence
                best_last_seen_epoch = identity_last_seen_epoch

        if best_score >= match_threshold and best_id is not None:
            match_reason = "+".join(best_reasons)
            self.repo.upsert_identity(
                global_id=best_id,
                vehicle_class=features.vehicle_class,
                vehicle_color=features.vehicle_color,
                license_plate_text=features.license_plate_text,
                license_plate_confidence=features.license_plate_confidence,
                avg_width=features.bbox_width,
                avg_height=features.bbox_height,
                aspect_ratio=features.aspect_ratio,
                camera_id=features.camera_id,
            )
            if features.allow_association_cache:
                self.repo.create_association(
                    global_id=best_id,
                    camera_id=features.camera_id,
                    local_track_id=features.local_track_id,
                    request_id=features.request_id,
                    match_score=round(best_score, 4),
                    match_reason=match_reason,
                )
            logger.info(
                "global_id_matched",
                global_id=best_id,
                local_track_id=features.local_track_id,
                camera_id=features.camera_id,
                match_score=round(best_score, 4),
                match_reason=match_reason,
            )
            return GlobalMatchResult(
                global_id=best_id,
                match_score=round(best_score, 4),
                match_reason=match_reason,
            )

        # No acceptable match — create a new global identity
        return self._create_new_identity(features)

    def _refresh_existing_association(
        self, association, features: TrackFeatures
    ) -> GlobalMatchResult:
        """Refresh an existing association and its identity with newer features.

        This keeps the stored global identity up to date when later frames
        produce better OCR text, stronger confidence, or more stable visual
        measurements than the frame that created the original association.
        """
        identity = self.repo.get_identity_by_global_id(association.global_id)
        if identity is None:
            return GlobalMatchResult(
                global_id=association.global_id,
                match_score=association.match_score,
                match_reason=association.match_reason,
            )

        score, reasons = self._score(features, identity)
        match_score = round(score, 4) if score > 0 else association.match_score
        match_reason = "+".join(reasons) if reasons else association.match_reason
        self.repo.upsert_identity(
            global_id=association.global_id,
            vehicle_class=features.vehicle_class or identity.vehicle_class,
            vehicle_color=features.vehicle_color,
            license_plate_text=features.license_plate_text,
            license_plate_confidence=features.license_plate_confidence,
            avg_width=features.bbox_width,
            avg_height=features.bbox_height,
            aspect_ratio=features.aspect_ratio,
            camera_id=features.camera_id,
        )
        self.repo.update_association(
            association_id=association.id,
            match_score=match_score,
            match_reason=match_reason,
        )
        return GlobalMatchResult(
            global_id=association.global_id,
            match_score=match_score,
            match_reason=match_reason,
        )

    def resolve_detections(
        self,
        detections: List[Dict[str, Any]],
        *,
        camera_id: str = "",
        request_id: str = "",
        skip_classes: Optional[set] = None,
        ocr_match_confidence: float,
        global_id_match_score: float,
        allow_association_cache: bool = True,
    ) -> None:
        """Enrich a list of detection dicts in-place with global tracking fields.

        Detections whose ``name`` is in *skip_classes* are left untouched
        (no global ID assigned).  This is used to exclude plate-only
        detections from global tracking.
        """
        _skip = skip_classes or set()
        for det in detections:
            if det.get("name", "") in _skip:
                det.setdefault("global_id", "")
                det.setdefault("match_score", 0.0)
                det.setdefault("match_reason", "")
                continue

            bbox = det.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
            w = max(bbox[2] - bbox[0], 0.0) if len(bbox) >= 4 else 0.0
            h = max(bbox[3] - bbox[1], 0.0) if len(bbox) >= 4 else 0.0

            features = TrackFeatures(
                local_track_id=str(det.get("track_id", "")),
                camera_id=camera_id or det.get("camera_id", ""),
                request_id=request_id,
                vehicle_class=det.get("name", ""),
                vehicle_color=det.get("color", ""),
                license_plate_text=str(det.get("ocr_text", "") or ""),
                license_plate_confidence=float(det.get("ocr_confidence", 0.0) or 0.0),
                bbox_width=w,
                bbox_height=h,
                aspect_ratio=round(w / h, 4) if h > 0 else 0.0,
                ocr_match_confidence=ocr_match_confidence,
                global_id_match_score=global_id_match_score,
                allow_association_cache=allow_association_cache,
            )
            result = self.resolve(features)
            det["global_id"] = result.global_id
            det["match_score"] = result.match_score
            det["match_reason"] = result.match_reason

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _score(
        features: TrackFeatures,
        identity,
    ) -> tuple[float, List[str]]:
        """Compute a weighted similarity score between a track and a stored identity.

        Weights are *adaptive*: when a signal is absent on both sides its
        budget is redistributed proportionally to the present signals.
        """
        reasons: List[str] = []
        raw: Dict[str, float] = {}  # signal → [0..1] similarity
        available: Dict[str, float] = {}  # signal → base weight

        # --- class (exact) ---
        has_class = bool(features.vehicle_class and identity.vehicle_class)
        if has_class:
            if features.vehicle_class == identity.vehicle_class:
                raw["class"] = 1.0
            else:
                return 0.0, []  # hard reject on class mismatch
            available["class"] = W_CLASS

        # --- color ---
        has_color = bool(features.vehicle_color and identity.vehicle_color)
        if has_color:
            raw["color"] = GlobalTrackingService._color_similarity(
                features.vehicle_color,
                identity.vehicle_color,
            )
            available["color"] = W_COLOR

        # --- dimension / aspect-ratio ---
        has_dim = features.aspect_ratio > 0 and identity.aspect_ratio > 0
        if has_dim:
            ratio_diff = abs(features.aspect_ratio - identity.aspect_ratio) / max(
                features.aspect_ratio, identity.aspect_ratio, 1e-6
            )
            raw["dimension"] = max(0.0, 1.0 - ratio_diff)
            available["dimension"] = W_DIMENSION

        # --- plate (fuzzy) ---
        has_plate = bool(features.license_plate_text and identity.license_plate_text)
        if has_plate:
            raw["plate"] = GlobalTrackingService._plate_similarity(
                features.license_plate_text,
                identity.license_plate_text,
            )
            available["plate"] = W_PLATE

        if not available:
            return 0.0, []

        # Redistribute absent weights proportionally
        total_available = sum(available.values())
        scale = 1.0 / total_available if total_available > 0 else 0.0

        score = 0.0
        for signal, sim in raw.items():
            weighted = available[signal] * scale * sim
            score += weighted
            if sim >= 0.6:
                reasons.append(signal)

        return round(score, 4), reasons

    @staticmethod
    def _chars_confusable(a: str, b: str) -> bool:
        if a == b:
            return True
        return b in _CONFUSION_LOOKUP.get(a, set())

    @staticmethod
    def _normalize_color_group(color: str) -> str:
        value = str(color or "").strip().lower().replace("-", " ").replace("_", " ")
        if not value:
            return ""
        if value in {"white", "silver", "light gray", "light grey", "gray", "grey"}:
            return "light"
        if value in {"black", "dark gray", "dark grey", "charcoal"}:
            return "dark"
        if value in {"blue", "navy", "cyan"}:
            return "blue"
        if value in {"red", "maroon", "burgundy"}:
            return "red"
        if value in {"green", "olive"}:
            return "green"
        if value in {"yellow", "gold", "beige", "brown", "orange"}:
            return "warm"
        return value

    @staticmethod
    def _color_similarity(left_color: str, right_color: str) -> float:
        left_group = GlobalTrackingService._normalize_color_group(left_color)
        right_group = GlobalTrackingService._normalize_color_group(right_color)
        if not left_group or not right_group:
            return 0.0
        if left_group == right_group:
            return 1.0
        if {left_group, right_group} <= {"light", "dark"}:
            return 0.35
        return 0.1

    @staticmethod
    def _plate_similarity(left_plate: str, right_plate: str) -> float:
        """OCR-confusion-aware Levenshtein similarity in [0..1].

        Substitution of visually confusable characters costs 0.3 instead of
        1.0, so ``DL12CS1568`` vs ``DL12(S125`` still scores high.
        """
        a = str(left_plate or "")
        b = str(right_plate or "")
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        n, m = len(a), len(b)
        # Full-matrix Levenshtein with confusion-aware substitution cost
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = float(i)
        for j in range(m + 1):
            dp[0][j] = float(j)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    sub_cost = 0.0
                elif GlobalTrackingService._chars_confusable(a[i - 1], b[j - 1]):
                    sub_cost = 0.3
                else:
                    sub_cost = 1.0
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,  # deletion
                    dp[i][j - 1] + 1.0,  # insertion
                    dp[i - 1][j - 1] + sub_cost,  # substitution
                )

        max_len = max(n, m)
        return round(1.0 - dp[n][m] / max_len, 4)

    def _create_new_identity(self, features: TrackFeatures) -> GlobalMatchResult:
        new_global_id = f"gid_{uuid.uuid4().hex[:12]}"
        self.repo.upsert_identity(
            global_id=new_global_id,
            vehicle_class=features.vehicle_class,
            vehicle_color=features.vehicle_color,
            license_plate_text=features.license_plate_text,
            license_plate_confidence=features.license_plate_confidence,
            avg_width=features.bbox_width,
            avg_height=features.bbox_height,
            aspect_ratio=features.aspect_ratio,
            camera_id=features.camera_id,
        )
        if features.allow_association_cache:
            self.repo.create_association(
                global_id=new_global_id,
                camera_id=features.camera_id,
                local_track_id=features.local_track_id,
                request_id=features.request_id,
                match_score=0.0,
                match_reason="new_identity",
            )
        logger.info(
            "global_id_created",
            global_id=new_global_id,
            local_track_id=features.local_track_id,
            camera_id=features.camera_id,
        )
        return GlobalMatchResult(
            global_id=new_global_id,
            match_score=0.0,
            match_reason="new_identity",
        )


def get_global_tracking_service() -> GlobalTrackingService:
    return GlobalTrackingService()
