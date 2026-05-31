import hashlib
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

MIN_PIXEL_DISPLACEMENT_FOR_MOTION = 12.0

BehaviorLabel = Literal[
    "normal",
    "repeat_visit",
    "linger",
    "sensitive_zone_presence",
    "repeated_presence",
]

SensitiveZoneType = Literal[
    "sensitive",
    "restricted",
    "no_parking",
    "vip",
    "parking",
    "entry",
    "exit",
]

BEHAVIOR_LABEL_CATALOG: Dict[BehaviorLabel, str] = {
    "normal": "No significant behavioral pattern detected for the configured camera rules.",
    "repeat_visit": "The object has been observed repeatedly above the configured repeat threshold.",
    "linger": "The object remained in view longer than the configured linger threshold.",
    "sensitive_zone_presence": "The object was detected inside a configured sensitive zone type.",
    "repeated_presence": "The object repeatedly visited or stayed inside a sensitive zone.",
}

SENSITIVE_ZONE_TYPE_CATALOG: Dict[SensitiveZoneType, str] = {
    "sensitive": "General high-sensitivity area where presence should be flagged.",
    "restricted": "Access-controlled area where unauthorized presence may matter.",
    "no_parking": "Area where stopping or standing vehicles should be monitored.",
    "vip": "Protected or priority access area requiring additional scrutiny.",
    "parking": "Parking or waiting area used for stationary vehicle behavior rules.",
    "entry": "Ingress zone used to detect entering movement or gate approach behavior.",
    "exit": "Egress zone used to detect leaving movement or gate departure behavior.",
}

DEFAULT_SENSITIVE_ZONE_TYPES: Tuple[SensitiveZoneType, ...] = (
    "sensitive",
    "restricted",
)


class AnalyticsUtils:

    @staticmethod
    def derive_source_id(source_url: str) -> str:
        normalized = source_url.strip().lower()
        hash_obj = hashlib.sha1(normalized.encode("utf-8"))
        return hash_obj.hexdigest()[:12]

    @staticmethod
    def point_in_polygon(
        point: Tuple[float, float], polygon: List[List[float]]
    ) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def normalize_bbox_center(
        bbox_xyxy: List[float], frame_width: int, frame_height: int
    ) -> Tuple[float, float]:
        if not bbox_xyxy or len(bbox_xyxy) < 4:
            return (0.5, 0.5)
        x1, y1, x2, y2 = bbox_xyxy[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if frame_width > 0 and frame_height > 0:
            cx_norm = cx / frame_width
            cy_norm = cy / frame_height
            return (cx_norm, cy_norm)
        return (cx, cy)

    @staticmethod
    def compute_direction_vector(
        prev_center: Tuple[float, float], curr_center: Tuple[float, float]
    ) -> List[float]:
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude > 1e-6:
            return [dx / magnitude, dy / magnitude]
        return [0.0, 0.0]

    @staticmethod
    def direction_label_from_vector(direction_vector: List[float]) -> str:
        if len(direction_vector) < 2:
            return "stationary"
        dx, dy = float(direction_vector[0]), float(direction_vector[1])
        motion_threshold = 0.15
        if abs(dx) < motion_threshold and abs(dy) < motion_threshold:
            return "stationary"
        if abs(dx) >= abs(dy):
            return "left_to_right" if dx > 0 else "right_to_left"
        return "towards_bottom" if dy > 0 else "towards_top"

    @staticmethod
    def orientation_label_from_motion(direction_vector: List[float]) -> str:
        if len(direction_vector) < 2:
            return "unknown"
        dx, dy = float(direction_vector[0]), float(direction_vector[1])
        motion_threshold = 0.15
        if abs(dx) < motion_threshold and abs(dy) < motion_threshold:
            return "stationary"
        if abs(dy) > abs(dx):
            return "approaching" if dy > 0 else "receding"
        return "left_to_right" if dx > 0 else "right_to_left"

    @staticmethod
    def compute_speed_estimate(
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
        time_delta_ms: float,
    ) -> float:
        if time_delta_ms <= 0:
            return 0.0
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        distance_px = math.sqrt(dx * dx + dy * dy)
        return distance_px / (time_delta_ms / 1000.0)

    @staticmethod
    def compute_pixel_speed(
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
        time_delta_ms: float,
    ) -> float:
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        if math.sqrt(dx * dx + dy * dy) < MIN_PIXEL_DISPLACEMENT_FOR_MOTION:
            return 0.0
        return AnalyticsUtils.compute_speed_estimate(
            prev_center, curr_center, time_delta_ms
        )

    @staticmethod
    def convert_pixel_speed_to_kmph(
        pixel_speed: float, pixels_per_meter: Optional[float]
    ) -> float:
        if pixel_speed <= 0 or not pixels_per_meter or pixels_per_meter <= 0:
            return 0.0
        meters_per_second = pixel_speed / pixels_per_meter
        return meters_per_second * 3.6

    @staticmethod
    def format_velocity_kmph(speed_kmph: float) -> List[str]:
        return [f"{speed_kmph:.1f} km/h"]

    @staticmethod
    def format_velocity_display(
        pixel_speed: float,
        pixels_per_meter: Optional[float],
    ) -> List[str]:
        speed_kmph = AnalyticsUtils.convert_pixel_speed_to_kmph(
            pixel_speed, pixels_per_meter
        )
        return [f"{speed_kmph:.1f} km/h"]

    @staticmethod
    def get_behavior_label_meaning(label: str) -> str:
        return BEHAVIOR_LABEL_CATALOG.get(label, BEHAVIOR_LABEL_CATALOG["normal"])

    @staticmethod
    def get_behavior_label_catalog() -> Dict[BehaviorLabel, str]:
        return dict(BEHAVIOR_LABEL_CATALOG)

    @staticmethod
    def get_sensitive_zone_type_catalog() -> Dict[SensitiveZoneType, str]:
        return dict(SENSITIVE_ZONE_TYPE_CATALOG)

    @staticmethod
    def get_default_sensitive_zone_types() -> List[str]:
        return list(DEFAULT_SENSITIVE_ZONE_TYPES)

    @staticmethod
    def find_active_zone(
        center: Tuple[float, float], zones: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for zone in zones:
            coords = zone.get("coordinates", [])
            if coords and AnalyticsUtils.point_in_polygon(center, coords):
                return zone
        return None

    @staticmethod
    def build_empty_spatial_state() -> Dict[str, Any]:
        return {
            "active_zone_id": "",
            "active_zone_type": "",
            "is_inside_zone": False,
            "spatial_label": "outside_zone",
            "spatial_score": 0.0,
        }

    @staticmethod
    def build_empty_behavior_state() -> Dict[str, Any]:
        return {
            "is_repeat_visit": False,
            "is_lingering": False,
            "is_sensitive_zone_presence": False,
            "visit_count": 0,
            "dwell_time_ms": 0,
            "behavior_label": "normal",
            "behavior_score": 0.0,
            "behavior_label_meaning": BEHAVIOR_LABEL_CATALOG["normal"],
        }

    @staticmethod
    def build_vehicle_episodes(
        detections: List[Dict[str, Any]],
        reappearance_gap_ms: int = 3000,
    ) -> List[Dict[str, Any]]:
        episodes = []
        track_map: Dict[str, List[Dict[str, Any]]] = {}

        for det in detections:
            identity_key = det.get("global_id") or det.get("track_id") or ""
            if not identity_key:
                continue
            if identity_key not in track_map:
                track_map[identity_key] = []
            track_map[identity_key].append(det)

        for identity_key, dets in track_map.items():
            if not dets:
                continue
            dets_sorted = sorted(dets, key=lambda d: d.get("ts_ms", 0))
            current_episode: List[Dict[str, Any]] = []
            episode_index = 0

            for det in dets_sorted:
                if not current_episode:
                    current_episode = [det]
                    continue

                prev_det = current_episode[-1]
                prev_ts = int(prev_det.get("ts_ms", 0) or 0)
                curr_ts = int(det.get("ts_ms", 0) or 0)
                time_gap = curr_ts - prev_ts

                if time_gap > reappearance_gap_ms:
                    start_ts = current_episode[0].get("ts_ms", 0)
                    end_ts = current_episode[-1].get("ts_ms", 0)
                    episodes.append(
                        {
                            "identity_key": identity_key,
                            "episode_index": episode_index,
                            "start_ts_ms": start_ts,
                            "end_ts_ms": end_ts,
                            "duration_ms": end_ts - start_ts,
                            "detections": current_episode,
                            "detection_count": len(current_episode),
                        }
                    )
                    episode_index += 1
                    current_episode = [det]
                    continue

                current_episode.append(det)

            if current_episode:
                start_ts = current_episode[0].get("ts_ms", 0)
                end_ts = current_episode[-1].get("ts_ms", 0)
                episodes.append(
                    {
                        "identity_key": identity_key,
                        "episode_index": episode_index,
                        "start_ts_ms": start_ts,
                        "end_ts_ms": end_ts,
                        "duration_ms": end_ts - start_ts,
                        "detections": current_episode,
                        "detection_count": len(current_episode),
                    }
                )

        return episodes
