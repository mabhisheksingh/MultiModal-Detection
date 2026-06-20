"""Triton app package re-exports.

Lazy re-exports are used to avoid circular imports with the ANPR module tree.
"""

from __future__ import annotations

import importlib

# Map of names that code historically imported from ``triton_server.app`` to the
# actual module that defines them.  ``__getattr__`` resolves these lazily so that
# circular imports are avoided.
_IMPORT_MAP = {
    # app
    "app": "app.main",
    "start_server": "app.main",
    "attach_request_id": "app.main",
    # schemas
    "BehaviorConfig": "modules.anpr.schemas.anpr",
    "CommonVisionInputRequest": "modules.anpr.schemas.anpr",
    "DetectionResponseItem": "modules.anpr.schemas.anpr",
    "SourceMetadata": "modules.anpr.schemas.anpr",
    "VisionInputItem": "modules.anpr.schemas.anpr",
    "VisionInputOptions": "modules.anpr.schemas.anpr",
    "VisionProcessingConfig": "modules.anpr.schemas.anpr",
    "Zone": "modules.anpr.schemas.anpr",
    # services
    "ANPRService": "modules.anpr.services.anpr_service",
    "BehavioralPatternService": "modules.anpr.services.behavioral_pattern_service",
    "GlobalTrackingService": "modules.anpr.services.global_tracking_service",
    "LiveVideoSourceProcessor": "modules.anpr.services.video_source_processor",
    "PaddleOCREngine": "modules.anpr.services.paddle_ocr_engine",
    "SpatiotemporalCorrelationService": "modules.anpr.services.spatiotemporal_correlation_service",
    "TritonClient": "modules.anpr.services.triton_client",
    "_make_tracker": "modules.anpr.services.video_source_processor",
    # utils
    "AnalyticsUtils": "modules.anpr.utils.analytics_utils",
    "APIConstants": "modules.anpr.utils.constants",
    "ERROR_RESPONSES": "modules.anpr.utils.constants",
    "FileSourceUtils": "modules.anpr.utils.media_utils",
    "ImageAnalysisUtils": "modules.anpr.utils.media_utils",
    "ImageSourceUtils": "modules.anpr.utils.media_utils",
    "MediaSourceUtils": "modules.anpr.utils.media_utils",
    "VideoSourceUtils": "modules.anpr.utils.media_utils",
    "OCRUtils": "modules.anpr.utils.ocr_utils",
    "RequestTraceUtils": "modules.anpr.utils.request_utils",
    "aggregate_track_votes": "modules.anpr.utils.output_serializers",
    "build_detection_response_item": "modules.anpr.utils.output_serializers",
    "build_source_summary_rows": "modules.anpr.utils.output_serializers",
    "get_global_tracking_service": "modules.anpr.utils.dependencies",
    "model_dump_compat": "modules.anpr.utils.request_utils",
    "write_csv_rows": "modules.anpr.utils.output_serializers",
    "write_frame_detections_csv": "modules.anpr.utils.output_serializers",
    "write_track_summary_csv": "modules.anpr.utils.output_serializers",
    # repository
    "Base": "modules.anpr.repository.database",
    "BehavioralAnalyticsRepository": "modules.anpr.repository.behavioral_analytics_repository",
    "BehavioralEvent": "modules.anpr.repository.models",
    "CameraConfig": "modules.anpr.repository.models",
    "GlobalIdentity": "modules.anpr.repository.models",
    "GlobalMatchResult": "modules.anpr.services.global_tracking_service",
    "GlobalTrackRepository": "modules.anpr.repository.global_track_repository",
    "SessionLocal": "modules.anpr.repository.database",
    "TrackAssociation": "modules.anpr.repository.models",
    "TrackFeatures": "modules.anpr.services.global_tracking_service",
    "VisitHistory": "modules.anpr.repository.models",
    "database": "modules.anpr.repository.database",
    "init_db": "modules.anpr.repository.database",
    # config
    "Settings": "modules.anpr.core.config",
    "settings": "modules.anpr.core.config",
    # common
    "configure_logging": "modules.common.logging",
    "health": "modules.common.health",
    "health_router": "modules.common.health",
    "setup_logger": "modules.common.logging",
    # dependencies module
    "dependencies": "modules.anpr.utils.dependencies",
    # output_serializers module
    "output_serializers": "modules.anpr.utils.output_serializers",
    # route modules exposed by older tests
    "anpr_v1": "modules.anpr.api.anpr",
    "anpr_v2": "modules.anpr.api.anpr",
    "anpr_v1_routes": "modules.anpr.api.anpr",
    "anpr_v2_routes": "modules.anpr.api.anpr",
}


def __getattr__(name: str):
    module_path = _IMPORT_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_path)
    try:
        value = getattr(module, name)
    except AttributeError:
        # Some historical imports treated the module itself as the export
        # (e.g. ``output_serializers``, ``dependencies``, ``database``).
        value = module

    globals()[name] = value
    return value
