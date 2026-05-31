import uuid

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from triton_server.app import AnalyticsUtils


class DetectionResponseItem(BaseModel):
    frame_id: int = 0
    ts_ms: Optional[int] = None
    track_id: str = ""
    cls: int = -1
    name: str = ""
    conf: float = 0.0
    bbox_xyxy: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    polygon: Optional[List[List[float]]] = None
    area_px: int = 0
    center: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    velocity: List[str] = Field(default_factory=lambda: ["0 km/h"])
    direction: str = "stationary"
    orientation: str = "unknown"
    sources: List[str] = Field(default_factory=list)
    blur_score: float = 0.0
    ocr_confidence: float = 0.0
    ocr_text: str = ""
    color: str = ""
    camera_id: str = ""
    global_id: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    plate_bbox_xyxy: Optional[List[float]] = None
    plate_color: str = ""
    plate_conf: float = 0.0
    plate_area_px: int = 0
    direction_vector: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    spatial_state: Dict[str, Any] = Field(default_factory=dict)
    behavior_state: Dict[str, Any] = Field(default_factory=dict)


class Zone(BaseModel):
    zone_id: str
    zone_type: str
    # Tuple enforces exactly 2 values (x, y) per point.
    coordinates: List[Tuple[float, float]] = Field(
        ...,
        description="Polygon coordinates as list of (x, y) pairs, normalized 0-1",
        examples=[
            "[[0.0615, 0.2972], [0.551, 0.2972], [0.551, 0.6083], [0.0615, 0.6083]]"
        ],
    )


class BehaviorConfig(BaseModel):
    repeat_visit_threshold: int = Field(3, ge=1)
    linger_threshold_ms: int = Field(30000, ge=0)
    sensitive_zone_types: List[str] = Field(
        default_factory=AnalyticsUtils.get_default_sensitive_zone_types
    )
    min_behavior_score: float = Field(0.6, ge=0.0, le=1.0)


class SourceMetadata(BaseModel):
    url: HttpUrl = Field(
        ...,
        examples=["http://172.50.32.89:9004/videos/Location_4_Video1.mp4"],
        description="Remote media URL",
        min_length=1,
    )
    lat: Optional[float] = Field(
        None, ge=-90.0, le=90.0, examples=[89.0], description="latitude"
    )
    lon: Optional[float] = Field(
        None, ge=-180.0, le=180.0, examples=[180.0], description="longitude"
    )
    pixels_per_meter: Optional[float] = Field(
        25.0,
        gt=0.0,
        examples=[15.0],
        description="pixels per meter",
    )
    zones: Optional[List[Zone]] = None
    behavior_config: Optional[BehaviorConfig] = None
    camera_id: Optional[Any] = None

    @field_validator("camera_id", mode="before")
    @classmethod
    def apply_lambda(cls, v):
        return (
            lambda cid: (str(cid).strip() if cid is not None else "")
            or f"cam_{uuid.uuid4().hex[:8]}"
        )(v)

    @field_validator("lat", mode="before")
    @classmethod
    def validate_lat(cls, value: Any) -> Any:
        if value == "":
            raise ValueError(
                "lat must be a valid number or null; empty string is not allowed"
            )
        return value

    @field_validator("lon", mode="before")
    @classmethod
    def validate_lon(cls, value: Any) -> Any:
        if value == "":
            raise ValueError(
                "lon must be a valid number or null; empty string is not allowed"
            )
        return value

    @field_validator("pixels_per_meter", mode="before")
    @classmethod
    def validate_pixels_per_meter(cls, value: Any) -> Any:
        if value == "":
            raise ValueError(
                "pixels_per_meter must be a valid number or null; empty string is not allowed"
            )
        return value


class VisionProcessingConfig(BaseModel):
    confidence_threshold: Optional[float] = Field(
        0.2, ge=0.2, le=1.0, description="Minimum detection confidence threshold"
    )
    ocr_confidence_threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Minimum OCR confidence required to keep OCR text"
    )
    ocr_match_confidence: Optional[float] = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy OCR plate similarity required to trust OCR for global ID matching",
    )
    global_id_match_score: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum overall similarity score required to reuse an existing global ID",
    )
    frames_per_second: Optional[int] = Field(
        10, ge=10, le=30, description="Video frame sampling rate"
    )
    nms_threshold: Optional[float] = Field(
        None, description="Non-maximum suppression threshold"
    )
    similarity_threshold: Optional[float] = Field(
        None, description="Similarity threshold for identity matching"
    )
    spatial_threshold: Optional[float] = Field(
        None, description="Spatial distance threshold for correlation"
    )
    max_disappeared: Optional[int] = Field(
        None, description="Maximum disappeared frames before dropping a track"
    )
    confirmation_frames: Optional[int] = Field(
        None, description="Frames required before confirming a track"
    )
    save_cropped_faces: Optional[bool] = Field(
        None, description="Whether to save cropped face images"
    )
    generate_embeddings: Optional[bool] = Field(
        None, description="Whether to generate face or object embeddings"
    )
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    embedding_detector_backend: Optional[str] = Field(
        None, description="Detector backend used before embedding extraction"
    )
    min_face_size: Optional[int] = Field(
        None, description="Minimum face size for face recognition pipelines"
    )
    enable_face_alignment: Optional[bool] = Field(
        None, description="Whether to enable face alignment before recognition"
    )
    custom_output_fps: Optional[float] = Field(
        None, description="Custom output FPS for video output"
    )
    enable_cross_camera_reid: Optional[bool] = Field(
        None, description="Whether to enable cross-camera re-identification"
    )
    cross_camera_threshold: Optional[float] = Field(
        None, description="Threshold for cross-camera re-identification"
    )
    platform: Literal["face_recognition", "object_detection", "anpr_object"] = Field(
        None, description="Target vision platform such as ANPR or face recognition"
    )
    is_ocr_enabled: bool = Field(
        ..., description="Enable OCR processing"
    )
    ocr_plate_text_mode: Literal["balanced", "strict"] = Field(
        "balanced",
        description="OCR plate text validation mode: 'strict' (regex validation) or 'balanced' (no validation)"
    )

    @field_validator("ocr_plate_text_mode", mode="before")
    @classmethod
    def normalize_ocr_plate_text_mode(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value
        
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional platform-specific processing configuration",
    )


class VisionInputOptions(BaseModel):
    uri: Optional[str] = Field(None, description="Path or URL to the input resource")
    content_base64: Optional[str] = Field(
        None, description="Inline base64 encoded content for direct API inputs"
    )
    camera_id: Optional[str] = Field(
        None, description="Camera identifier associated with this input"
    )
    lat: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Latitude associated with this input"
    )
    lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Longitude associated with this input",
    )
    pixels_per_meter: Optional[float] = Field(
        25.0,
        gt=0.0,
        examples=[15.0],
        description="Pixels-per-meter calibration value associated with this input",
    )
    frames: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Inline frame collection for image-sequence style inputs",
    )
    zones: List[Zone] = Field(
        default_factory=list,
        description="Optional zones associated with this specific input",
    )
    behavior_config: Optional[BehaviorConfig] = Field(
        None,
        description="Optional behavioral analytics config associated with this input",
    )
    enabled: Optional[bool] = Field(
        None, description="Optional execution flag for this input"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional input-specific dynamic options",
    )

    @model_validator(mode="after")
    def validate_input_payload(self) -> "VisionInputOptions":
        if not self.uri and not self.content_base64 and not self.frames:
            raise ValueError(
                "At least one of 'uri', 'content_base64', or 'frames' must be provided in options."
            )
        return self


class VisionInputItem(BaseModel):
    id: str = Field(..., min_length=1, description="Unique input identifier")
    input_type: Literal[
        "video_file",
        "image_file",
        "image_sequence",
        "manifest",
        "video_url",
        "image_url",
        "remote_url",
        "inline_base64",
        "json_ceph_urls",
        "json_frame_bytes",
        "video_bytes",
        "frame_bytes_direct",
    ] = Field(..., description="Common input type for all vision platforms")
    options: VisionInputOptions = Field(default_factory=VisionInputOptions)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic metadata associated with the input",
    )


class CommonVisionInputRequest(BaseModel):
    processing_config: VisionProcessingConfig = Field(
        default_factory=VisionProcessingConfig,
        description="Global processing configuration for the request",
    )
    inputs: List[VisionInputItem] = Field(
        ...,
        min_length=1,
        description="List of inputs to process across vision platforms",
    )
