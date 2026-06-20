from typing import Any

import structlog

from triton_server.app import AnalyticsUtils, BehavioralAnalyticsRepository

logger = structlog.get_logger(__name__)


class BehavioralPatternService:

    def __init__(self, repository: BehavioralAnalyticsRepository | None = None):
        self.repo = repository or BehavioralAnalyticsRepository()

    def enrich_detections_with_behavior_state(
        self,
        detections: list[dict[str, Any]],
        behavior_config: dict[str, Any] | None = None,
        camera_id: str = "",
        request_id: str = "",
    ) -> list[dict[str, Any]]:
        if not behavior_config:
            enriched = []
            for det in detections:
                enriched_det = det.copy()
                enriched_det["behavior_state"] = AnalyticsUtils.build_empty_behavior_state()
                enriched.append(enriched_det)
            return enriched

        enriched = []
        config = behavior_config or {}
        reappearance_gap_ms = int(config.get("reappearance_gap_ms", 3000) or 3000)
        episodes = AnalyticsUtils.build_vehicle_episodes(detections, reappearance_gap_ms=reappearance_gap_ms)

        episode_map: dict[str, dict[str, Any]] = {}
        for ep in episodes:
            identity_key = ep["identity_key"]
            for episode_det in ep.get("detections", []):
                detection_key = self._build_detection_key(identity_key, episode_det)
                episode_map[detection_key] = ep

        for det in detections:
            enriched_det = det.copy()

            global_id = det.get("global_id", "")
            track_id = det.get("track_id", "")
            identity_key = global_id or track_id
            ts_ms = det.get("ts_ms", 0)
            frame_id = det.get("frame_id", 0)
            detection_key = self._build_detection_key(identity_key, det)

            behavior_state = self._compute_behavior_state(
                identity_key=identity_key,
                ts_ms=ts_ms,
                frame_id=frame_id,
                episode=episode_map.get(detection_key),
                spatial_state=det.get("spatial_state", {}),
                behavior_config=config,
                camera_id=camera_id,
                request_id=request_id,
            )

            enriched_det["behavior_state"] = behavior_state
            enriched.append(enriched_det)

        return enriched

    @staticmethod
    def _build_detection_key(identity_key: str, det: dict[str, Any]) -> str:
        return f"{identity_key}|{det.get('frame_id', 0)}|{det.get('ts_ms', 0)}|{det.get('track_id', '')}"

    def _compute_behavior_state(
        self,
        identity_key: str,
        ts_ms: int,
        frame_id: int,
        episode: dict[str, Any] | None,
        spatial_state: dict[str, Any],
        behavior_config: dict[str, Any],
        camera_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        behavior_state = AnalyticsUtils.build_empty_behavior_state()

        if not identity_key:
            return behavior_state

        repeat_visit_threshold = int(behavior_config.get("repeat_visit_threshold", 3) or 3)
        linger_threshold_ms = int(behavior_config.get("linger_threshold_ms", 30000) or 30000)
        sensitive_zone_types = {
            str(zone_type).strip().lower()
            for zone_type in behavior_config.get("sensitive_zone_types", ["sensitive", "restricted"])
            if str(zone_type).strip()
        }
        min_behavior_score = float(behavior_config.get("min_behavior_score", 0.6) or 0.6)
        visit_timestamp = ts_ms / 1000.0
        zone_id = spatial_state.get("active_zone_id", "")

        visit_count = 1
        if episode:
            visit_count = int(episode.get("episode_index", 0) or 0) + 1

        behavior_state["visit_count"] = visit_count
        behavior_state["is_repeat_visit"] = visit_count > repeat_visit_threshold

        duration_ms = 0
        if episode:
            duration_ms = int(episode.get("duration_ms", 0) or 0)
            behavior_state["dwell_time_ms"] = duration_ms
        behavior_state["is_lingering"] = duration_ms > linger_threshold_ms

        is_inside_zone = spatial_state.get("is_inside_zone", False)
        active_zone_type = str(spatial_state.get("active_zone_type", "")).strip().lower()

        if is_inside_zone and active_zone_type in sensitive_zone_types:
            behavior_state["is_sensitive_zone_presence"] = True

        if behavior_state["is_sensitive_zone_presence"] and behavior_state["is_repeat_visit"]:
            behavior_state["behavior_label"] = "repeated_presence"
            behavior_state["behavior_score"] = 0.85
        elif behavior_state["is_sensitive_zone_presence"]:
            behavior_state["behavior_label"] = "sensitive_zone_presence"
            behavior_state["behavior_score"] = 0.7
        elif behavior_state["is_lingering"]:
            behavior_state["behavior_label"] = "linger"
            behavior_state["behavior_score"] = 0.75
        elif behavior_state["is_repeat_visit"]:
            behavior_state["behavior_label"] = "repeat_visit"
            behavior_state["behavior_score"] = 0.65

        if behavior_state["behavior_score"] < min_behavior_score:
            behavior_state["behavior_label"] = "normal"
            behavior_state["behavior_score"] = 0.0

        behavior_state["is_repeat_visit"] = behavior_state["visit_count"] > repeat_visit_threshold
        behavior_state["is_lingering"] = behavior_state["dwell_time_ms"] > linger_threshold_ms

        behavior_state["behavior_label_meaning"] = AnalyticsUtils.get_behavior_label_meaning(
            behavior_state["behavior_label"]
        )

        if episode and int(episode.get("episode_index", 0) or 0) > 0:
            episode_detections = episode.get("detections", [])
            first_episode_det = episode_detections[0] if episode_detections else None
            is_episode_start = bool(
                first_episode_det
                and int(first_episode_det.get("frame_id", 0) or 0) == int(frame_id or 0)
                and int(first_episode_det.get("ts_ms", 0) or 0) == int(ts_ms or 0)
            )

            if is_episode_start:
                try:
                    self.repo.record_visit(
                        global_id=identity_key,
                        camera_id=camera_id,
                        request_id=request_id,
                        visit_timestamp=visit_timestamp,
                        zone_id=zone_id,
                        dwell_duration_ms=behavior_state["dwell_time_ms"],
                    )
                except Exception as e:
                    logger.warning(
                        "failed_to_record_visit",
                        error=str(e),
                        identity_key=identity_key,
                    )

        # Record behavioral event in DB if not normal
        if behavior_state["behavior_label"] != "normal":
            try:
                self.repo.record_behavioral_event(
                    global_id=identity_key,
                    camera_id=camera_id,
                    request_id=request_id,
                    behavior_label=behavior_state["behavior_label"],
                    behavior_score=behavior_state["behavior_score"],
                    visit_count=visit_count,
                    dwell_time_ms=behavior_state["dwell_time_ms"],
                    is_sensitive_zone=behavior_state["is_sensitive_zone_presence"],
                    zone_id=zone_id,
                    zone_type=spatial_state.get("active_zone_type", ""),
                    detected_at=visit_timestamp,
                    frame_id=frame_id,
                )
            except Exception as e:
                logger.warning(
                    "failed_to_record_behavioral_event",
                    error=str(e),
                    identity_key=identity_key,
                )

        return behavior_state
