import asyncio
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import structlog
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils.ops import clip_boxes, xywhn2xyxy
from ultralytics.utils.plotting import Annotator, colors

from triton_server.app import (
    BehavioralPatternService,
    DetectionResponseItem,
    GlobalTrackingService,
    ImageAnalysisUtils,
    ImageSourceUtils,
    LiveVideoSourceProcessor,
    PaddleOCREngine,
    RequestTraceUtils,
    SpatiotemporalCorrelationService,
    TritonClient,
    VisionProcessingConfig,
    output_serializers,
    settings,
)

logger = structlog.get_logger(__name__)

_EMPTY_DETECTIONS: tuple[np.ndarray, np.ndarray, np.ndarray] = (
    np.empty((0, 4), dtype=np.float32),
    np.empty((0,), dtype=np.float32),
    np.empty((0,), dtype=int),
)

_EMPTY_TRACKED = (
    np.empty((0, 4), dtype=np.float32),
    np.empty((0,), dtype=int),
    np.empty((0,), dtype=np.float32),
    np.empty((0,), dtype=int),
    np.array([], dtype=str),
)

_TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


class ANPRService:
    def __init__(
        self,
        *,
        ocr_service: PaddleOCREngine,
        vehicle_triton_client: TritonClient,
        plate_triton_client: TritonClient,
        object_triton_client: TritonClient | None = None,
        global_tracking_service: GlobalTrackingService = None,
        spatial_correlation_service: SpatiotemporalCorrelationService | None = None,
        behavioral_pattern_service: BehavioralPatternService | None = None,
    ) -> None:

        self.vehicle_triton_client = vehicle_triton_client
        self.plate_triton_client = plate_triton_client
        self.object_triton_client = object_triton_client
        self.ocr_service = ocr_service
        self.global_tracking_service = global_tracking_service
        self.tracker = BYTETracker(_TRACKER_ARGS, frame_rate=30)
        self.video_source_processor = LiveVideoSourceProcessor(
            self,
            spatial_service=spatial_correlation_service,
            behavioral_service=behavioral_pattern_service,
        )
        self.vehicle_class_id_name_map = settings.vehicle_class_id_name_map
        self.plate_class_id_name_map = settings.plate_class_id_name_map
        self.object_class_id_name_map = settings.object_class_id_name_map
        self.vehicle_class_names = self._build_class_names_from_id_map(
            self.vehicle_class_id_name_map,
            fallback_name="vehicle",
        )
        self.plate_class_names = self._build_class_names_from_id_map(
            self.plate_class_id_name_map,
            fallback_name="number_plate",
        )
        self.object_class_names = self._build_class_names_from_id_map(
            self.object_class_id_name_map,
            fallback_name="object",
        )
        self.plate_class_offset = max(self.vehicle_class_id_name_map.keys(), default=-1) + 1
        self.class_names = self.vehicle_class_names + self.plate_class_names
        self.plate_candidate_vehicle_classes = set(settings.plate_candidate_vehicle_classes_list)
        self.ocr_class_ids = {self.plate_class_offset + class_id for class_id in self.plate_class_id_name_map}

        logger.info(
            "anpr_service_initialized",
            class_names=self.class_names,
            ocr_class_ids=list(self.ocr_class_ids),
            plate_candidate_vehicle_classes=sorted(self.plate_candidate_vehicle_classes),
        )

    @staticmethod
    def _resolved_processing_config(
        processing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = VisionProcessingConfig(**(processing_config or {})).model_dump()
        return resolved

    @staticmethod
    def _build_class_names_from_id_map(class_id_name_map: dict[int, str], *, fallback_name: str) -> list[str]:
        if not class_id_name_map:
            return [fallback_name]
        max_id = max(class_id_name_map.keys())
        return [class_id_name_map.get(i, f"class_{i}") for i in range(max_id + 1)]

    async def infer_frame_payloads(
        self,
        frame: np.ndarray,
        frame_idx: int,
        request_id: str | None = None,
        source_name: str | None = None,
        *,
        ocr_confidence_threshold: float,
        is_ocr_enabled: bool,
        platform: str,
    ) -> list[dict[str, Any]]:
        """
        Sequential detection flow to reduce Triton server calls:
        1. Detect vehicles/objects based on platform field
        2. For anpr_object platform: filter by PLATE_CANDIDATE_VEHICLE_CLASSES and optionally run plate detection
        3. For object_detection platform: skip plate detection entirely
        """
        triton_request_id = RequestTraceUtils.build_triton_request_id(request_id, source_name, frame_idx)

        # Step 1: Detect vehicles or objects based on platform field
        # When platform is "anpr_object", use vehicle detection model
        # When platform is "object_detection", use object detection model
        if platform == "anpr_object":
            logger.info(
                "using_vehicle_detection_model_for_anpr",
                frame_idx=frame_idx,
                request_id=request_id,
                model_name=settings.vehicle_model_name,
                is_ocr_enabled=is_ocr_enabled,
            )
            vehicle_result = await self.vehicle_triton_client.infer(
                image=frame, frame_idx=frame_idx, request_id=triton_request_id
            )
            vehicle_payload = {
                "source": "vehicle",
                "detections_raw": vehicle_result[0],
                "preprocess_meta": vehicle_result[1],
                "class_names": self.vehicle_class_names,
                "class_id_offset": 0,
            }
        elif platform == "object_detection":
            logger.info(
                "using_object_detection_model_skip_ocr",
                frame_idx=frame_idx,
                request_id=request_id,
                model_name=settings.object_model_name,
            )
            vehicle_result = await self.object_triton_client.infer(
                image=frame, frame_idx=frame_idx, request_id=triton_request_id
            )
            vehicle_payload = {
                "source": "object",
                "detections_raw": vehicle_result[0],
                "preprocess_meta": vehicle_result[1],
                "class_names": self.object_class_names,
                "class_id_offset": 0,
            }
            # For object_detection platform, always skip plate detection
            return [vehicle_payload]
        else:
            logger.warning(
                "unknown_platform_using_vehicle_model",
                frame_idx=frame_idx,
                request_id=request_id,
                platform=platform,
            )
            vehicle_result = await self.vehicle_triton_client.infer(
                image=frame, frame_idx=frame_idx, request_id=triton_request_id
            )
            vehicle_payload = {
                "source": "vehicle",
                "detections_raw": vehicle_result[0],
                "preprocess_meta": vehicle_result[1],
                "class_names": self.vehicle_class_names,
                "class_id_offset": 0,
            }

        # Step 2: For anpr_object platform, check is_ocr_enabled to determine if plate detection should run
        if platform == "anpr_object" and not is_ocr_enabled:
            logger.info(
                "ocr_disabled_skipping_plate_detection",
                frame_idx=frame_idx,
                request_id=request_id,
            )
            return [vehicle_payload]

        # Step 2: Filter to valid vehicle classes (use low threshold, NMS will filter later)
        valid_vehicle_boxes = self._filter_valid_vehicle_boxes(
            frame,
            vehicle_payload,
            confidence_threshold=ocr_confidence_threshold,
        )

        if len(valid_vehicle_boxes) == 0:
            logger.debug(
                "no_valid_vehicles_skipping_plate_detection",
                frame_idx=frame_idx,
                request_id=request_id,
            )
            # Return empty plate payload
            return [
                vehicle_payload,
                {
                    "source": "plate",
                    "detections_raw": np.empty((1, 0, 6), dtype=np.float32),
                    "preprocess_meta": vehicle_result[1],
                    "class_names": self.plate_class_names,
                    "class_id_offset": self.plate_class_offset,
                },
            ]

        # Step 3: Run plate detection only on valid vehicle crops
        logger.debug(
            "running_plate_detection_on_valid_vehicles",
            valid_vehicle_count=len(valid_vehicle_boxes),
            frame_idx=frame_idx,
        )

        plate_payload = await self._detect_plates_in_vehicles(
            frame,
            valid_vehicle_boxes,
            frame_idx,
            triton_request_id,
            vehicle_result[1],
            confidence_threshold=ocr_confidence_threshold,
        )

        return [vehicle_payload, plate_payload]

    def _filter_valid_vehicle_boxes(
        self,
        frame: np.ndarray,
        vehicle_payload: dict[str, Any],
        *,
        confidence_threshold: float,
    ) -> list[np.ndarray]:
        """Filter vehicle detections to PLATE_CANDIDATE_VEHICLE_CLASSES."""
        boxes, _, class_ids = self._decode_inference_payload(
            frame,
            vehicle_payload["detections_raw"],
            vehicle_payload["preprocess_meta"],
            class_names=vehicle_payload["class_names"],
            class_id_offset=vehicle_payload["class_id_offset"],
            confidence_threshold=confidence_threshold,
        )

        valid_boxes = [
            box
            for box, cls_id in zip(boxes, class_ids, strict=False)
            if self.vehicle_class_names[cls_id].lower() in self.plate_candidate_vehicle_classes
        ]

        logger.debug("vehicles_filtered", total=len(boxes), valid=len(valid_boxes))
        return valid_boxes

    async def _detect_plates_in_vehicles(
        self,
        frame: np.ndarray,
        vehicle_boxes: list[np.ndarray],
        frame_idx: int,
        request_id: str,
        base_preprocess_meta: dict,
        *,
        confidence_threshold: float,
    ) -> dict[str, Any]:
        """Run plate detection on valid vehicle crops, transform coords to full image."""
        img_h, img_w = frame.shape[:2]
        crops_info = []

        for box in vehicle_boxes:
            x1, y1, x2, y2 = np.clip(box.astype(int), [0, 0, 0, 0], [img_w, img_h, img_w, img_h])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops_info.append((crop, x1, y1))

        if not crops_info:
            return self._empty_plate_payload(base_preprocess_meta)

        results = await asyncio.gather(
            *[
                self.plate_triton_client.infer(image=crop, frame_idx=frame_idx, request_id=request_id)
                for crop, _, _ in crops_info
            ]
        )

        all_boxes, all_confs, all_cls = [], [], []
        for (det_raw, meta), (crop, x_off, y_off) in zip(results, crops_info, strict=False):
            boxes, confs, cls_ids = self._decode_inference_payload(
                crop,
                det_raw,
                meta,
                class_names=self.plate_class_names,
                class_id_offset=self.plate_class_offset,
                confidence_threshold=confidence_threshold,
            )
            for box in boxes:
                box[[0, 2]] += x_off
                box[[1, 3]] += y_off
            all_boxes.append(boxes)
            all_confs.append(confs)
            all_cls.append(cls_ids)

        if not any(len(b) for b in all_boxes):
            return self._empty_plate_payload(base_preprocess_meta)

        combined_boxes = np.vstack(all_boxes) if all_boxes else np.empty((0, 4))
        combined_confs = np.hstack(all_confs) if all_confs else np.empty((0,))
        combined_cls = np.hstack(all_cls) if all_cls else np.empty((0,), dtype=int)

        logger.debug("plates_detected_in_vehicles", count=len(combined_boxes))

        return {
            "source": "plate",
            "decoded_boxes": combined_boxes,
            "decoded_confs": combined_confs,
            "decoded_cls": combined_cls,
            "preprocess_meta": base_preprocess_meta,
            "class_names": self.plate_class_names,
            "class_id_offset": self.plate_class_offset,
        }

    def _empty_plate_payload(self, preprocess_meta: dict) -> dict[str, Any]:
        return {
            "source": "plate",
            "decoded_boxes": np.empty((0, 4), dtype=np.float32),
            "decoded_confs": np.empty((0,), dtype=np.float32),
            "decoded_cls": np.empty((0,), dtype=int),
            "preprocess_meta": preprocess_meta,
            "class_names": self.plate_class_names,
            "class_id_offset": self.plate_class_offset,
        }

    @staticmethod
    def _decode_inference_payload(
        frame: np.ndarray,
        detections_raw: np.ndarray,
        preprocess_meta: dict,
        *,
        class_names: list[str],
        class_id_offset: int,
        confidence_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if detections_raw.ndim == 3 and detections_raw.shape[0] == 1:
            detections_raw = detections_raw[0]  # (1, features, anchors) → (features, anchors)

        if detections_raw.ndim != 2:
            logger.error(
                "model_output_shape_invalid",
                shape=getattr(detections_raw, "shape", None),
            )
            return _EMPTY_DETECTIONS

        # Model outputs BCN: (features, anchors) — transpose to (anchors, features)
        if detections_raw.shape[0] < detections_raw.shape[1]:
            detections_raw = detections_raw.T  # (features, anchors) → (anchors, features)

        if detections_raw.shape[1] < 5:
            logger.error(
                "model_output_shape_invalid",
                shape=getattr(detections_raw, "shape", None),
            )
            return _EMPTY_DETECTIONS

        # Model outputs: (anchors, 4+nc) where first 4 cols are normalized cx,cy,w,h
        # followed by either a single confidence score or per-class scores
        if detections_raw.shape[1] == 5:
            confidences = detections_raw[:, 4]
            class_ids = np.zeros(len(detections_raw), dtype=int)
        else:
            class_scores = detections_raw[:, 4:]
            class_ids = np.argmax(class_scores, axis=1).astype(int)
            confidences = class_scores[np.arange(len(class_scores)), class_ids]

        conf_mask = confidences >= confidence_threshold
        raw_boxes = detections_raw[conf_mask, :4].astype(np.float32)
        conf_filtered = confidences[conf_mask].astype(np.float32)
        cls_filtered = class_ids[conf_mask]

        if len(raw_boxes) == 0:
            return _EMPTY_DETECTIONS

        # Convert normalized cxcywh → absolute xyxy using ultralytics ops
        img_h, img_w = frame.shape[:2]
        input_h, input_w = preprocess_meta.get("input_size", (img_h, img_w))
        x_offset = preprocess_meta.get("x_offset", 0)
        y_offset = preprocess_meta.get("y_offset", 0)
        scale = max(preprocess_meta.get("scale", 1.0), 1e-6)

        # Ultralytics: normalized xywh → absolute xyxy in input space
        boxes_input = xywhn2xyxy(raw_boxes[:, :4], w=input_w, h=input_h)

        # Unscale from letterboxed input space → original frame space
        boxes_input[:, [0, 2]] = (boxes_input[:, [0, 2]] - x_offset) / scale
        boxes_input[:, [1, 3]] = (boxes_input[:, [1, 3]] - y_offset) / scale

        # Ultralytics: clip boxes to frame bounds
        boxes_abs = clip_boxes(boxes_input, (img_h, img_w)).astype(np.float32)

        valid = (boxes_abs[:, 2] > boxes_abs[:, 0]) & (boxes_abs[:, 3] > boxes_abs[:, 1])
        boxes_abs, conf_filtered, cls_filtered = (
            boxes_abs[valid],
            conf_filtered[valid],
            cls_filtered[valid],
        )

        if len(boxes_abs) == 0:
            return _EMPTY_DETECTIONS

        max_class_id = class_id_offset + max(len(class_names) - 1, 0)
        cls_filtered = np.clip(cls_filtered + class_id_offset, class_id_offset, max_class_id)
        return boxes_abs, conf_filtered, cls_filtered.astype(int)

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """NMS via OpenCV. Returns kept indices."""
        if len(boxes_xyxy) == 0:
            return np.empty((0,), dtype=int)
        # cv2.dnn.NMSBoxes expects [x, y, w, h] lists
        xywh = np.column_stack(
            [
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
            ]
        )
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        return indices.flatten() if len(indices) else np.empty((0,), dtype=int)

    def _collect_model_detections(
        self,
        frame: np.ndarray,
        inference_payloads: list[dict[str, Any]],
        confidence_threshold: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
        all_xyxy: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []
        all_sources: list[str] = []

        for payload in inference_payloads:
            # Check if payload has pre-decoded boxes (from sequential plate detection)
            if "decoded_boxes" in payload:
                xyxy = payload["decoded_boxes"]
                confidences_filtered = payload["decoded_confs"]
                class_ids = payload["decoded_cls"]
            else:
                xyxy, confidences_filtered, class_ids = self._decode_inference_payload(
                    frame,
                    payload["detections_raw"],
                    payload["preprocess_meta"],
                    class_names=payload["class_names"],
                    class_id_offset=payload["class_id_offset"],
                    confidence_threshold=confidence_threshold,
                )

            if len(xyxy):
                all_xyxy.append(xyxy)
                all_confidences.append(confidences_filtered)
                all_class_ids.append(class_ids)
                if payload["source"] == "vehicle":
                    model_name = settings.vehicle_model_name
                elif payload["source"] == "object":
                    model_name = settings.object_model_name
                else:
                    model_name = settings.plate_model_name
                all_sources.extend([model_name] * len(xyxy))
        return all_xyxy, all_confidences, all_class_ids, all_sources

    def _merge_detections(
        self,
        all_xyxy: list[np.ndarray],
        all_confidences: list[np.ndarray],
        all_class_ids: list[np.ndarray],
        all_sources: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if all_xyxy:
            merged_xyxy = np.concatenate(all_xyxy, axis=0)
            merged_conf = np.concatenate(all_confidences, axis=0)
            merged_cls = np.concatenate(all_class_ids, axis=0)
            merged_sources = np.array(all_sources)
            keep = self._nms(merged_xyxy, merged_conf, settings.iou_threshold)
            return (
                merged_xyxy[keep],
                merged_conf[keep],
                merged_cls[keep],
                merged_sources[keep],
            )
        return (
            _EMPTY_DETECTIONS[0],
            _EMPTY_DETECTIONS[1],
            _EMPTY_DETECTIONS[2],
            np.array([], dtype=str),
        )

    def _track_detections(
        self,
        frame: np.ndarray,
        merged_xyxy: np.ndarray,
        merged_conf: np.ndarray,
        merged_cls: np.ndarray,
        merged_sources: np.ndarray,
        tracker: BYTETracker | None,
        enable_tracking: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_det = len(merged_xyxy)
        active_tracker = tracker if tracker is not None else self.tracker
        if enable_tracking and active_tracker is not None and n_det > 0:
            img_shape = frame.shape[:2]
            det_data = np.hstack(
                [
                    merged_xyxy,
                    merged_conf[:, None],
                    merged_cls[:, None].astype(np.float32),
                ]
            )
            det_boxes = Boxes(det_data, orig_shape=img_shape)
            tracked = active_tracker.update(det_boxes)
            if len(tracked):
                t_xyxy, t_ids, t_conf, t_cls = (
                    tracked[:, :4],
                    tracked[:, 4].astype(int),
                    tracked[:, 5],
                    tracked[:, 6].astype(int),
                )
                t_idx = tracked[:, 7].astype(int)
                t_sources = np.array([merged_sources[i] if i < len(merged_sources) else "triton" for i in t_idx])
                return t_xyxy, t_ids, t_conf, t_cls, t_sources
            return _EMPTY_TRACKED
        return (
            merged_xyxy,
            np.arange(n_det, dtype=int),
            merged_conf,
            merged_cls.astype(int),
            merged_sources,
        )

    def _build_detection_record(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        class_id: int,
        tracker_id: int,
        confidence: float,
        model_source: str,
        order_idx: int,
        camera_id: str,
    ) -> dict[str, Any]:
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
        base_label = f"#{tracker_id} {class_name}"
        clipped = clip_boxes(box[None], frame.shape[:2])[0].astype(int)
        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        blur = ImageAnalysisUtils.calculate_blur(crop) if crop.size > 0 else 0.0
        is_vehicle = class_id < self.plate_class_offset
        color = ImageAnalysisUtils.extract_dominant_color(crop) if (is_vehicle and crop.size > 0) else ""
        plate_color = ImageAnalysisUtils.extract_plate_color(crop) if ((not is_vehicle) and crop.size > 0) else ""
        width, height = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
        orientation = "horizontal" if width > height else "vertical" if height > width else "square"
        result_item = output_serializers.build_detection_response_item(
            DetectionResponseItem(
                frame_id=0,
                ts_ms=None,
                track_id=str(tracker_id),
                cls=class_id,
                name=class_name,
                conf=float(confidence),
                bbox_xyxy=[float(x) for x in box],
                polygon=None,
                area_px=int(width * height),
                center=[float(x1) + (width / 2.0), float(y1) + (height / 2.0)],
                velocity=["0 km/h"],
                direction="stationary",
                orientation=orientation,
                sources=[str(model_source)],
                blur_score=float(blur),
                ocr_confidence=0.0,
                ocr_text="",
                color=color,
                camera_id=camera_id,
                plate_color=plate_color,
            )
        )
        return {
            "order": order_idx,
            "base_label": base_label,
            "label": base_label,
            "box": box,
            "confidence": confidence,
            "tracker_id": tracker_id,
            "blur_score": blur,
            "result_item": result_item,
            "crop": crop,
            "ocr_confidence": 0.0,
        }

    def _build_detection_records(
        self,
        frame: np.ndarray,
        tracked_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        camera_id: str,
    ) -> list[dict[str, Any]]:
        t_xyxy, t_ids, t_conf, t_cls, t_sources = tracked_data
        return [
            self._build_detection_record(
                frame,
                t_xyxy[i],
                int(t_cls[i]),
                int(t_ids[i]),
                float(t_conf[i]),
                str(t_sources[i]) if i < len(t_sources) else "triton",
                i,
                camera_id,
            )
            for i in range(len(t_xyxy))
        ]

    @staticmethod
    def _get_class_name(record: dict[str, Any]) -> str:
        """Extract normalized class name from detection record."""
        return str(record["result_item"].get("name", "")).strip().lower()

    def _is_plate_candidate_vehicle_record(self, record: dict[str, Any]) -> bool:
        """Check if record is an eligible vehicle class for plate association."""
        return self._get_class_name(record) in self.plate_candidate_vehicle_classes

    def _should_run_ocr_for_record(self, record: dict[str, Any], detection_records: list[dict[str, Any]]) -> bool:
        """
        Only run OCR on plate detections that are inside eligible vehicles.
        Eligible vehicles are defined by PLATE_CANDIDATE_VEHICLE_CLASSES.
        """
        plate_class_names = {str(name).strip().lower() for name in self.plate_class_id_name_map.values()}

        if self._get_class_name(record) not in plate_class_names:
            return False

        eligible_vehicles = [rec for rec in detection_records if self._is_plate_candidate_vehicle_record(rec)]
        if not eligible_vehicles:
            return False

        for vehicle_record in eligible_vehicles:
            containment_score = self._bbox_containment_score(record["box"], vehicle_record["box"])
            if containment_score >= 0.5:
                return True
        return False

    def _maybe_run_ocr(
        self,
        record: dict[str, Any],
        detection_records: list[dict[str, Any]],
        ocr_confidence_threshold: float,
        *,
        plate_text_mode: str,
    ) -> float:
        crop, confidence, blur_score = (
            record["crop"],
            record["confidence"],
            record["blur_score"],
        )
        should_run = self._should_run_ocr_for_record(record, detection_records)
        crop_has_pixels = crop.size > 0
        blur_ok = blur_score >= settings.plate_blur_threshold
        if not (should_run and crop_has_pixels and blur_ok):
            logger.warning(
                "ocr_skipped",
                reason="gate_not_satisfied",
                is_plate_contained=should_run,
                crop_size=int(crop.size),
                blur_score=float(blur_score),
                blur_threshold=float(settings.plate_blur_threshold),
                detection_confidence=float(confidence),
            )
            return 0.0

        ocr_start = time.perf_counter()
        try:
            ocr_results = self.ocr_service.recognize(crop, plate_text_mode=plate_text_mode)
            if not ocr_results:
                return time.perf_counter() - ocr_start

            best_result = max(ocr_results, key=lambda r: r["confidence"])
            normalized_text = best_result["text"]
            ocr_conf = best_result["confidence"]

            if normalized_text and float(ocr_conf) >= ocr_confidence_threshold:
                record["label"] = f"{record['base_label']} {normalized_text}"
                record["result_item"]["ocr_text"] = normalized_text
                record["result_item"]["ocr_confidence"] = ocr_conf
                record["ocr_confidence"] = ocr_conf
                logger.info(
                    "ocr_success",
                    track_id=record["tracker_id"],
                    text=normalized_text,
                    ocr_confidence=round(ocr_conf, 3),
                )
        except Exception as e:
            logger.error("ocr_error", tracker_id=record["tracker_id"], error=str(e))
        return time.perf_counter() - ocr_start

    def _run_ocr_on_records(
        self,
        detection_records: list[dict[str, Any]],
        *,
        ocr_confidence_threshold: float,
        plate_text_mode: str,
    ) -> float:
        ocr_time_total = 0.0
        sorted_records = sorted(
            detection_records,
            key=lambda rec: (rec["blur_score"], rec["confidence"]),
            reverse=True,
        )
        for record in sorted_records:
            ocr_time_total += self._maybe_run_ocr(
                record,
                detection_records,
                ocr_confidence_threshold,
                plate_text_mode=plate_text_mode,
            )
        detection_records.sort(key=lambda rec: rec["order"])
        return ocr_time_total

    @staticmethod
    def _bbox_containment_score(plate_box: np.ndarray, vehicle_box: np.ndarray) -> float:
        """Fraction of the plate bbox area inside the vehicle bbox (0.0–1.0)."""
        px1, py1, px2, py2 = plate_box[:4]
        vx1, vy1, vx2, vy2 = vehicle_box[:4]
        inter = max(0.0, min(px2, vx2) - max(px1, vx1)) * max(0.0, min(py2, vy2) - max(py1, vy1))
        return inter / max((px2 - px1) * (py2 - py1), 1e-6)

    def _associate_plates_to_vehicles(
        self,
        detection_records: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Match plates to enclosing vehicles; merge plate info into vehicle record."""
        vehicles, plates = [], []
        plate_class_names = {str(name).strip().lower() for name in self.plate_class_id_name_map.values()}

        for rec in detection_records:
            if self._get_class_name(rec) in plate_class_names:
                plates.append(rec)
            else:
                vehicles.append(rec)

        eligible_vehicles = [rec for rec in vehicles if self._is_plate_candidate_vehicle_record(rec)]

        matched: set = set()
        for pidx, prec in enumerate(plates):
            best_score, best_veh = 0.0, None
            for vrec in eligible_vehicles:
                s = self._bbox_containment_score(prec["box"], vrec["box"])
                if s > best_score:
                    best_score, best_veh = s, vrec
            if best_score < threshold or best_veh is None:
                continue
            matched.add(pidx)
            pi, vi = prec["result_item"], best_veh["result_item"]
            ocr_text = pi.get("ocr_text", "")
            if ocr_text and float(pi.get("ocr_confidence", 0)) >= float(vi.get("ocr_confidence", 0)):
                vi["ocr_text"] = ocr_text
                vi["ocr_confidence"] = pi["ocr_confidence"]
                best_veh["label"] = f"{best_veh['base_label']} {ocr_text}"
            vi["plate_bbox_xyxy"] = pi.get("bbox_xyxy")
            vi["plate_color"] = pi.get("plate_color", "")
            vi["plate_conf"] = float(pi.get("conf", 0.0))
            vi["plate_area_px"] = int(pi.get("area_px", 0))
            for src in pi.get("sources", []):
                if src not in vi.get("sources", []):
                    vi["sources"].append(src)

        orphans = [plates[i] for i in range(len(plates)) if i not in matched]
        logger.info(
            "plate_vehicle_association",
            vehicles=len(vehicles),
            eligible_vehicles=len(eligible_vehicles),
            plates=len(plates),
            matched=len(matched),
            orphans=len(orphans),
        )
        return vehicles + orphans

    def _annotate_detection_records(self, frame: np.ndarray, detection_records: list[dict[str, Any]]) -> np.ndarray:
        ann = Annotator(frame.copy())
        for rec in detection_records:
            ri, box = rec["result_item"], rec["box"]
            cls_name = str(ri.get("name", "")).strip().lower()
            cls_id = self.class_names.index(cls_name) if cls_name in self.class_names else 0
            clr = colors(cls_id, True)
            plate_class_map = getattr(self, "plate_class_id_name_map", {}) or {}
            plate_class_names = {str(name).strip().lower() for name in plate_class_map.values()}
            if not plate_class_names and hasattr(self, "plate_class_offset"):
                plate_class_names = {
                    str(name).strip().lower()
                    for idx, name in enumerate(getattr(self, "class_names", []))
                    if idx >= getattr(self, "plate_class_offset", 0)
                }
            if cls_name in plate_class_names:
                ocr = ri.get("ocr_text", "")
                lbl = f"{ri.get('name','')} {ri.get('conf',0):.2f}"
                ann.box_label(box, f"{lbl} {ocr}" if ocr else lbl, color=clr)
            else:
                lbl = rec["label"]
                gid = ri.get("global_id", "")
                ann.box_label(box, f"{lbl} [{gid}]" if gid else lbl, color=clr)
                pbbox = ri.get("plate_bbox_xyxy")
                if pbbox and len(pbbox) >= 4:
                    ann.box_label(
                        [int(v) for v in pbbox[:4]],
                        ri.get("ocr_text", ""),
                        color=colors(self.plate_class_offset, True),
                    )
        return ann.result()

    def process_frame_after_inference(
        self,
        frame: np.ndarray,
        inference_payloads: list[dict[str, Any]],
        *,
        tracker: BYTETracker | None = None,
        enable_tracking: bool = True,
        camera_id: str = "",
        request_id: str = "",
        confidence_threshold: float,
        ocr_confidence_threshold: float,
        ocr_match_confidence: float,
        global_id_match_score: float,
        plate_text_mode: str,
        is_ocr_enabled: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Processes a single frame using pre-computed detections."""
        try:
            all_xyxy, all_conf, all_cls, all_src = self._collect_model_detections(
                frame, inference_payloads, confidence_threshold
            )
            merged_xyxy, merged_conf, merged_cls, merged_src = self._merge_detections(
                all_xyxy, all_conf, all_cls, all_src
            )

            n = len(merged_xyxy)
            logger.info(
                "detections_after_nms",
                count=n,
                vehicle_count=(int((merged_cls < self.plate_class_offset).sum()) if n else 0),
                plate_count=(int((merged_cls >= self.plate_class_offset).sum()) if n else 0),
            )

            tracked = self._track_detections(
                frame,
                merged_xyxy,
                merged_conf,
                merged_cls,
                merged_src,
                tracker,
                enable_tracking,
            )
            records = self._build_detection_records(frame, tracked, camera_id)
            if is_ocr_enabled:
                self._run_ocr_on_records(
                    records,
                    ocr_confidence_threshold=ocr_confidence_threshold,
                    plate_text_mode=plate_text_mode,
                )
            else:
                logger.info("ocr_disabled_skipping_ocr_processing")
            records = self._associate_plates_to_vehicles(records)
            results = [r["result_item"] for r in records]
            self.global_tracking_service.resolve_detections(
                results,
                camera_id=camera_id,
                request_id=request_id,
                skip_classes=set(self.plate_class_id_name_map.values()),
                ocr_match_confidence=ocr_match_confidence,
                global_id_match_score=global_id_match_score,
                allow_association_cache=enable_tracking,
            )
            return self._annotate_detection_records(frame, records), results
        except Exception as e:
            logger.error(
                "frame_post_processing_error",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return frame, []

    async def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        *,
        tracker: BYTETracker | None = None,
        enable_tracking: bool = True,
        camera_id: str = "",
        request_id: str | None = None,
        source_name: str | None = None,
        processing_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Runs inference + post-processing on a single frame."""
        try:
            resolved_processing_config = self._resolved_processing_config(processing_config)
            payloads = await self.infer_frame_payloads(
                frame,
                frame_idx,
                request_id=request_id,
                source_name=source_name,
                ocr_confidence_threshold=float(resolved_processing_config["ocr_confidence_threshold"]),
                is_ocr_enabled=bool(resolved_processing_config["is_ocr_enabled"]),
                platform=str(resolved_processing_config.get("platform", "anpr_object")),
            )
            return self.process_frame_after_inference(
                frame,
                payloads,
                tracker=tracker,
                enable_tracking=enable_tracking,
                camera_id=camera_id,
                request_id=request_id or "",
                confidence_threshold=float(resolved_processing_config["confidence_threshold"]),
                ocr_confidence_threshold=float(resolved_processing_config["ocr_confidence_threshold"]),
                ocr_match_confidence=float(resolved_processing_config["ocr_match_confidence"]),
                global_id_match_score=float(resolved_processing_config["global_id_match_score"]),
                plate_text_mode=str(resolved_processing_config.get("ocr_plate_text_mode", "balanced")),
                is_ocr_enabled=bool(resolved_processing_config["is_ocr_enabled"]),
            )
        except Exception as e:
            logger.error("frame_processing_error", frame_idx=frame_idx, error=str(e))
            return frame, []

    async def process_image_source(
        self,
        image_path: str,
        request_id: str | None = None,
        processing_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], str | None]:
        path = ImageSourceUtils.validate_image_source_path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        annotated_image, detections = await self.process_frame(
            image,
            frame_idx=0,
            enable_tracking=False,
            request_id=request_id,
            source_name=path.name,
            processing_config=processing_config,
        )

        output_path = self._save_annotated_image(
            annotated_image=annotated_image,
            output_name=path.name,
            request_id=request_id,
            fallback_folder_name=path.stem,
            detections_count=len(detections),
        )

        return annotated_image, detections, output_path

    async def process_image_url(
        self,
        image_url: str,
        frame_idx: int = 0,
        request_id: str | None = None,
        processing_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], str | None]:
        image = await asyncio.to_thread(ImageSourceUtils.load_image_from_url, image_url)
        name = Path(urlparse(image_url).path).name or f"image_{frame_idx:04d}.jpg"

        annotated_image, detections = await self.process_frame(
            image,
            frame_idx=frame_idx,
            enable_tracking=False,
            request_id=request_id,
            source_name=name,
            processing_config=processing_config,
        )

        output_path = self._save_annotated_image(
            annotated_image=annotated_image,
            output_name=name,
            request_id=request_id,
            fallback_folder_name=Path(name).stem,
            detections_count=len(detections),
        )

        return annotated_image, detections, output_path

    def _save_annotated_image(
        self,
        *,
        annotated_image: np.ndarray,
        output_name: str,
        request_id: str | None,
        fallback_folder_name: str,
        detections_count: int,
    ) -> str | None:
        output_path: str | None = None
        try:
            output_dir = Path(settings.output_folder).resolve()
            folder_name = request_id or fallback_folder_name
            request_folder = output_dir / folder_name
            request_folder.mkdir(parents=True, exist_ok=True)

            output_filename = f"annotated_{output_name}"
            output_path = str(request_folder / output_filename)
            cv2.imwrite(output_path, annotated_image)
            logger.info(
                "image_output_saved",
                output_path=output_path,
                detections_count=detections_count,
            )
        except Exception as e:
            logger.error("image_output_save_failed", error=str(e))
        return output_path

    async def process_video_source(
        self,
        source: str,
        output_dir: str,
        save_csv: bool = True,
        request_id: str | None = None,
        camera_id: str = "",
        lat: float | None = None,
        lon: float | None = None,
        pixels_per_meter: float | None = None,
        zones: list[dict[str, Any]] | None = None,
        behavior_config: dict[str, Any] | None = None,
        processing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_processing_config = self._resolved_processing_config(processing_config)
        return await self.video_source_processor.process(
            source=source,
            output_dir=output_dir,
            save_csv=save_csv,
            request_id=request_id,
            camera_id=camera_id,
            lat=lat,
            lon=lon,
            pixels_per_meter=pixels_per_meter,
            zones=zones,
            behavior_config=behavior_config,
            frames_per_second=float(resolved_processing_config["frames_per_second"]),
            confidence_threshold=float(resolved_processing_config["confidence_threshold"]),
            ocr_confidence_threshold=float(resolved_processing_config["ocr_confidence_threshold"]),
            ocr_match_confidence=float(resolved_processing_config["ocr_match_confidence"]),
            global_id_match_score=float(resolved_processing_config["global_id_match_score"]),
            plate_text_mode=str(resolved_processing_config.get("ocr_plate_text_mode", "balanced")),
            is_ocr_enabled=resolved_processing_config.get("is_ocr_enabled"),
            platform=str(resolved_processing_config.get("platform", "anpr_object")),
        )
