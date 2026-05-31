from typing import Any, Dict, List, Optional, Tuple
import structlog

from triton_server.app import AnalyticsUtils

logger = structlog.get_logger(__name__)


class SpatiotemporalCorrelationService:

    def enrich_detections_with_spatial_state(
        self,
        detections: List[Dict[str, Any]],
        zones: List[Dict[str, Any]],
        pixels_per_meter: Optional[float] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> List[Dict[str, Any]]:
        enriched = []
        track_history: Dict[
            str, List[Tuple[Tuple[float, float], int, int, Tuple[float, float]]]
        ] = {}

        for det in detections:
            enriched_det = det.copy()

            bbox_xyxy = det.get("bbox_xyxy", [])
            ts_ms = det.get("ts_ms", 0)
            track_id = det.get("track_id", "")
            center_norm = AnalyticsUtils.normalize_bbox_center(
                bbox_xyxy, frame_width, frame_height
            )
            center_px = (
                tuple(det.get("center", [0.0, 0.0])[:2])
                if len(det.get("center", [])) >= 2
                else (0.0, 0.0)
            )

            direction_vector = [0.0, 0.0]
            pixel_speed = 0.0
            velocity_display = AnalyticsUtils.format_velocity_display(
                pixel_speed, pixels_per_meter
            )

            if track_id:
                if track_id in track_history:
                    prev_center_norm, prev_ts, _, prev_center_px = track_history[
                        track_id
                    ][-1]
                    direction_vector = AnalyticsUtils.compute_direction_vector(
                        prev_center_norm, center_norm
                    )
                    time_delta = ts_ms - prev_ts
                    pixel_speed = AnalyticsUtils.compute_pixel_speed(
                        prev_center_px, center_px, time_delta
                    )
                    if pixel_speed <= 0.0:
                        direction_vector = [0.0, 0.0]
                    velocity_display = AnalyticsUtils.format_velocity_display(
                        pixel_speed, pixels_per_meter
                    )

                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(
                    (center_norm, ts_ms, det.get("frame_id", 0), center_px)
                )

            enriched_det["direction_vector"] = direction_vector
            enriched_det["velocity"] = velocity_display
            enriched_det["direction"] = AnalyticsUtils.direction_label_from_vector(
                direction_vector
            )
            enriched_det["orientation"] = AnalyticsUtils.orientation_label_from_motion(
                direction_vector
            )

            spatial_state = self._compute_spatial_state(center_norm, zones)
            enriched_det["spatial_state"] = spatial_state

            enriched.append(enriched_det)

        return enriched

    def _compute_spatial_state(
        self, center_norm: Tuple[float, float], zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        spatial_state = AnalyticsUtils.build_empty_spatial_state()

        active_zone = AnalyticsUtils.find_active_zone(center_norm, zones)

        if active_zone:
            spatial_state["active_zone_id"] = active_zone.get("zone_id", "")
            spatial_state["active_zone_type"] = active_zone.get("zone_type", "")
            spatial_state["is_inside_zone"] = True
            spatial_state["spatial_label"] = "inside_zone"
            spatial_state["spatial_score"] = 0.9

        return spatial_state
