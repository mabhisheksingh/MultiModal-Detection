import json
from typing import Dict, List
from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Centralized configuration management with validation and robust error handling."""

    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    app_name: str = Field("ANPR API", validation_alias="APP_NAME")
    app_env: str = Field("development", validation_alias="APP_ENV")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    # ============================================================================
    # TRITON SERVER CONFIGURATION
    # ============================================================================
    triton_server_url: str = Field(
        "localhost:8001",
        validation_alias="TRITON_SERVER_URL",
        description="Triton server URL (gRPC port, typically 8001)",
    )
    triton_protocol: str = Field(
        "grpc",
        validation_alias="TRITON_PROTOCOL",
        description="Protocol: 'grpc' (recommended) or 'http'",
    )

    # Model configuration
    vehicle_model_name: str = Field(
        "vehicle_detection_rt_detr", validation_alias="VEHICLE_MODEL_NAME"
    )
    vehicle_class_id_map: str = Field(
        '{"0":"vehicle"}', validation_alias="VEHICLE_CLASS_ID_MAP"
    )
    object_model_name: str = Field(
        "object_detection", validation_alias="OBJECT_MODEL_NAME"
    )
    object_class_id_map: str = Field(
        '{"0":"person"}', validation_alias="OBJECT_CLASS_ID_MAP"
    )
    plate_model_name: str = Field(
        "plate_region_detection_rt_detr", validation_alias="PLATE_MODEL_NAME"
    )
    plate_class_id_map: str = Field(
        '{"0":"number_plate"}', validation_alias="PLATE_CLASS_ID_MAP"
    )
    plate_candidate_vehicle_classes: str = Field(
        "autorickshaw,bicycle,bus,car,caravan,motorcycle,truck,vehicle fallback",
        validation_alias="PLATE_CANDIDATE_VEHICLE_CLASSES",
        description="Comma-separated vehicle class names that are allowed to receive plate association and OCR",
    )

    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    batch_size: int = Field(4, validation_alias="BATCH_SIZE", ge=1, le=128)
    input_folder: str = Field("data/input", validation_alias="INPUT_FOLDER")
    output_folder: str = Field("data/output", validation_alias="OUTPUT_FOLDER")

    # ============================================================================
    # DETECTION SETTINGS
    # ============================================================================
    iou_threshold: float = Field(0.5, validation_alias="IOU_THRESHOLD", ge=0.0, le=1.0)
    blur_threshold: float = Field(40.0, validation_alias="BLUR_THRESHOLD", ge=0.0)
    plate_blur_threshold: float = Field(
        60.0,
        validation_alias="PLATE_BLUR_THRESHOLD",
        ge=0.0,
        description="Minimum Laplacian variance required on the cropped plate before triggering OCR",
    )

    # ============================================================================
    # FILE EXTENSIONS
    # ============================================================================
    video_extensions: str = Field(
        ".mp4,.avi,.mov,.mkv,.webm", validation_alias="VIDEO_EXTENSIONS"
    )
    image_extensions: str = Field(
        ".jpg,.jpeg,.png,.bmp,.JPG,.JPEG,.PNG", validation_alias="IMAGE_EXTENSIONS"
    )

    # ============================================================================
    # OCR CONFIGURATION
    # ============================================================================
    ocr_timeout: int = Field(
        3,
        validation_alias="OCR_TIMEOUT",
        ge=1,
        le=30,
        description="OCR timeout in seconds",
    )
    ocr_languages: str = Field("en", validation_alias="OCR_LANGUAGES")
    x_padding: float = Field(0.20, validation_alias="X_PADDING", ge=0.0, le=1.0)
    y_padding: float = Field(0.20, validation_alias="Y_PADDING", ge=0.0, le=1.0)
    ocr_cleanup_regex: str = Field(r"[^A-Z0-9]", validation_alias="OCR_CLEANUP_REGEX")
    
    # OCR Detection Parameters (PaddleOCR tuning)
    ocr_det_limit_side_len: int = Field(
        960,
        validation_alias="OCR_DET_LIMIT_SIDE_LEN",
        ge=320,
        le=2048,
        description="Max side length for text detection. Higher = better for small text but slower. Default: 960",
    )
    ocr_det_box_thresh: float = Field(
        0.5,
        validation_alias="OCR_DET_BOX_THRESH",
        ge=0.1,
        le=0.9,
        description="Detection box confidence threshold. Lower = more detections. Default: 0.5",
    )
    ocr_rec_score_thresh: float = Field(
        0.6,
        validation_alias="OCR_REC_SCORE_THRESH",
        ge=0.1,
        le=0.95,
        description="Recognition score threshold. Lower = more candidates. Default: 0.6",
    )
    ocr_det_thresh: float = Field(
        0.25,
        validation_alias="OCR_DET_THRESH",
        ge=0.1,
        le=0.9,
        description="Overall detection confidence threshold for PaddleOCR detector. Default: 0.3",
    )
    ocr_det_unclip_ratio: float = Field(
        1.8,
        validation_alias="OCR_DET_UNCLIP_RATIO",
        ge=1.0,
        le=3.0,
        description="Unclip ratio used to expand detection boxes (higher = larger boxes). Default: 1.6",
    )
    
    # OCR Preprocessing Parameters
    ocr_enable_preprocessing: bool = Field(
        True,
        validation_alias="OCR_ENABLE_PREPROCESSING",
        description="Enable CLAHE and denoising preprocessing for better OCR. Default: True",
    )
    ocr_clahe_clip_limit: float = Field(
        2.0,
        validation_alias="OCR_CLAHE_CLIP_LIMIT",
        ge=1.0,
        le=10.0,
        description="CLAHE contrast enhancement clip limit. Higher = more contrast. Default: 2.0",
    )
    ocr_denoise_strength: int = Field(
        10,
        validation_alias="OCR_DENOISE_STRENGTH",
        ge=3,
        le=30,
        description="Denoising filter strength. Higher = more smoothing. Default: 10",
    )

    # ============================================================================
    # GLOBAL TRACKING
    # ============================================================================
    global_track_lookback_seconds: int = Field(
        300,
        validation_alias="GLOBAL_TRACK_LOOKBACK_SECONDS",
        ge=1,
        description="How far back in time to search for reusable global identities",
    )
    global_track_database_url: str = Field(
        "sqlite:///./global_tracking.db",
        validation_alias="GLOBAL_TRACK_DATABASE_URL",
        description="Database URL for global tracking persistence",
    )

    # ============================================================================
    # CEPH/S3 STORAGE (Optional)
    # ============================================================================
    ceph_access_key: str = Field("", validation_alias="CEPH_ACCESS_KEY")
    ceph_secret_access_key: str = Field("", validation_alias="CEPH_SECRET_ACCESS_KEY")
    ceph_region: str = Field("us-east-1", validation_alias="CEPH_REGION")
    ceph_bucket_name: str = Field("", validation_alias="CEPH_BUCKET_NAME")
    ceph_endpoint_url: str = Field("", validation_alias="CEPH_ENDPOINT_URL")

    model_config = SettingsConfigDict(
        # Resolve .env relative to project root so running from repo root loads it correctly
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================

    @computed_field
    @property
    def video_extensions_list(self) -> List[str]:
        """Parse video extensions into a list."""
        return [ext.strip() for ext in self.video_extensions.split(",") if ext.strip()]

    @computed_field
    @property
    def image_extensions_list(self) -> List[str]:
        """Parse image extensions into a list."""
        return [ext.strip() for ext in self.image_extensions.split(",") if ext.strip()]

    @computed_field
    @property
    def combined_extensions_list(self) -> List[str]:
        """Parse extensions into a list."""
        extensions = self.video_extensions.split(",") + self.image_extensions.split(",")
        return [ext.strip() for ext in extensions if ext.strip()]

    @computed_field
    @property
    def plate_candidate_vehicle_classes_list(self) -> List[str]:
        return [
            name.strip().lower()
            for name in self.plate_candidate_vehicle_classes.split(",")
            if name.strip()
        ]

    def _parse_class_id_map(self, raw_map: str, *, field_name: str) -> Dict[int, str]:
        try:
            parsed = json.loads(raw_map)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must be a JSON object")

        normalized: Dict[int, str] = {}
        for key, value in parsed.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} keys must be numeric class IDs"
                ) from exc
            class_name = str(value).strip()
            if not class_name:
                raise ValueError(f"{field_name} values must be non-empty class names")
            normalized[class_id] = class_name
        return normalized

    @computed_field
    @property
    def vehicle_class_id_name_map(self) -> Dict[int, str]:
        return self._parse_class_id_map(
            self.vehicle_class_id_map, field_name="VEHICLE_CLASS_ID_MAP"
        )

    @computed_field
    @property
    def object_class_id_name_map(self) -> Dict[int, str]:
        return self._parse_class_id_map(
            self.object_class_id_map, field_name="OBJECT_CLASS_ID_MAP"
        )

    @computed_field
    @property
    def plate_class_id_name_map(self) -> Dict[int, str]:
        return self._parse_class_id_map(
            self.plate_class_id_map, field_name="PLATE_CLASS_ID_MAP"
        )

    @computed_field
    @property
    def data_input_dir(self) -> Path:
        """Get absolute path to input directory."""
        return (PROJECT_ROOT / self.input_folder).resolve()

    @computed_field
    @property
    def data_output_dir(self) -> Path:
        """Get absolute path to output directory."""
        return (PROJECT_ROOT / self.output_folder).resolve()

    # ============================================================================
    # CONSTANTS
    # ============================================================================

    DETECT_CLASSES: List[int] = [0]  # YOLO class IDs to keep
    CSV_FRAME_HEADER: List[str] = [
        "frame",
        "track_id",
        "plate_text",
        "confidence",
        "confidence_ocr",
    ]
    CSV_TRACK_SUMMARY_HEADER: List[str] = [
        "track_id",
        "plate_text",
        "votes",
        "avg_confidence",
        "avg_confidence_ocr",
    ]
    CSV_CONSOLIDATED_HEADER: List[str] = [
        "source",
        "track_id",
        "plate_text",
        "votes",
        "avg_confidence",
        "avg_confidence_ocr",
    ]
    LABELS_SUBDIR: str = "labels"
    OCR_RESULT_SUBDIR: str = "ocr_result"

    # ============================================================================
    # VALIDATION & UTILITY METHODS
    # ============================================================================

    def create_directories(self) -> None:
        """Create all required directories with error handling."""
        directories = [
            self.data_input_dir,
            self.data_output_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                raise

    def validate_config(self) -> None:
        """Validate critical configuration values."""
        errors = []

        if not self.triton_server_url:
            errors.append("TRITON_SERVER_URL is required")

        if not self.vehicle_model_name:
            errors.append("VEHICLE_MODEL_NAME is required")
        if not self.plate_model_name:
            errors.append("PLATE_MODEL_NAME is required")

        try:
            _ = self.vehicle_class_id_name_map
        except ValueError as exc:
            errors.append(str(exc))

        try:
            _ = self.plate_class_id_name_map
        except ValueError as exc:
            errors.append(str(exc))

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")

    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "triton_server_url": self.triton_server_url,
            "vehicle_model_name": self.vehicle_model_name,
            "plate_model_name": self.plate_model_name,
            "batch_size": self.batch_size,
            "data_input_dir": str(self.data_input_dir),
            "data_output_dir": str(self.data_output_dir),
            "ocr_languages": self.ocr_languages,
            "ocr_timeout": self.ocr_timeout,
            "ocr_cleanup_regex": self.ocr_cleanup_regex,
            "plate_blur_threshold": self.plate_blur_threshold,
            "x_padding": self.x_padding,
            "y_padding": self.y_padding,
            "plate_candidate_vehicle_classes": self.plate_candidate_vehicle_classes,
            "plate_candidate_vehicle_classes_list": self.plate_candidate_vehicle_classes_list,
            "vehicle_class_id_name_map": self.vehicle_class_id_name_map,
            "plate_class_id_name_map": self.plate_class_id_name_map,
        }


# Initialize settings instance
settings = Settings()

# Create directories and validate on module import
try:
    settings.create_directories()
    settings.validate_config()
    logger.info("Settings initialized successfully", config=settings.get_summary())
except Exception as e:
    logger.error(f"Failed to initialize settings: {str(e)}")
    raise
