"""Video source processing — frame batching, inference dispatch, and output assembly."""

from __future__ import annotations

import asyncio
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import structlog
from ultralytics.trackers.byte_tracker import BYTETracker

from triton_server.app import VideoSourceUtils, output_serializers, settings

if TYPE_CHECKING:
    from triton_server.app import (
        ANPRService,
        BehavioralPatternService,
        SpatiotemporalCorrelationService,
    )

logger = structlog.get_logger(__name__)

# Default BYTETracker args matching ultralytics bytetrack.yaml defaults
_TRACKER_ARGS_NS: dict = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def _make_tracker(fps: float) -> BYTETracker:
    from types import SimpleNamespace

    return BYTETracker(SimpleNamespace(**_TRACKER_ARGS_NS))


class LiveVideoSourceProcessor:
    """Orchestrates video-level read → batch → infer → post-process → write loop."""

    def __init__(
        self,
        service: ANPRService,
        spatial_service: SpatiotemporalCorrelationService | None = None,
        behavioral_service: BehavioralPatternService | None = None,
    ) -> None:
        self.service = service
        self.spatial_service = spatial_service
        self.behavioral_service = behavioral_service

    # ------------------------------------------------------------------
    # Video I/O helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_capture(source: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        return cap

    @staticmethod
    def _resolve_video_properties(
        capture: cv2.VideoCapture,
        first_frame: np.ndarray,
        frames_per_second: float,
    ) -> tuple[float, float, int, int, int | None]:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = float(source_fps) if source_fps and source_fps > 0 else 25.0
        target_fps = min(frames_per_second, source_fps)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        if w <= 0 or h <= 0:
            h, w = first_frame.shape[:2]
        return source_fps, target_fps, w, h, total

    @staticmethod
    def _create_writer(
        output_path: str,
        fps: float,
        w: int,
        h: int,
        capture: cv2.VideoCapture,
    ) -> cv2.VideoWriter:
        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter.fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if writer.isOpened():
                return writer
        capture.release()
        raise ValueError(f"Failed to create output video writer: {output_path}")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    async def _run_inference_batch(
        self,
        frames: list[np.ndarray],
        indices: list[int],
        request_id: str | None,
        source_name: str,
        ocr_confidence_threshold: float,
        is_ocr_enabled: bool,
        platform: str,
    ) -> list[list[dict[str, Any]]]:
        tasks = [
            self.service.infer_frame_payloads(
                f,
                idx,
                request_id=request_id,
                source_name=source_name,
                ocr_confidence_threshold=ocr_confidence_threshold,
                is_ocr_enabled=is_ocr_enabled,
                platform=platform,
            )
            for f, idx in zip(frames, indices, strict=False)
        ]
        return await asyncio.gather(*tasks)

    def _process_inference_batch(
        self,
        frames: list[np.ndarray],
        indices: list[int],
        payloads_batch: list[list[dict[str, Any]]],
        debug_dir: Path,
        tracker: BYTETracker,
        writer: cv2.VideoWriter,
        source_fps: float,
        all_detections: list[dict[str, Any]],
        request_id: str = "",
        camera_id: str = "",
        *,
        confidence_threshold: float,
        ocr_confidence_threshold: float,
        ocr_match_confidence: float,
        global_id_match_score: float,
        plate_text_mode: str,
        is_ocr_enabled: bool,
    ) -> int:
        processed = 0
        for i, (frame, payloads) in enumerate(zip(frames, payloads_batch, strict=False)):
            fidx = indices[i]
            if fidx < 5:
                cv2.imwrite(str(debug_dir / f"frame_{fidx:04d}.jpg"), frame)

            annotated, results = self.service.process_frame_after_inference(
                frame,
                payloads,
                tracker=tracker,
                enable_tracking=True,
                camera_id=camera_id,
                request_id=request_id,
                confidence_threshold=confidence_threshold,
                ocr_confidence_threshold=ocr_confidence_threshold,
                ocr_match_confidence=ocr_match_confidence,
                global_id_match_score=global_id_match_score,
                plate_text_mode=plate_text_mode,
                is_ocr_enabled=is_ocr_enabled,
            )
            for r in results:
                r["frame_id"] = fidx
                r["ts_ms"] = int((fidx / source_fps) * 1000) if source_fps > 0 else None

            all_detections.extend(results)
            processed += 1
            writer.write(annotated)
        return processed

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def process(
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
        *,
        frames_per_second: float,
        confidence_threshold: float,
        ocr_confidence_threshold: float,
        ocr_match_confidence: float,
        global_id_match_score: float,
        plate_text_mode: str,
        is_ocr_enabled: bool,
        platform: str,
    ) -> dict[str, Any]:
        logger.info("processing_video_source", source=source)
        VideoSourceUtils.validate_video_source(source)
        capture = self._open_capture(source)
        request_folder, output_path, source_name = VideoSourceUtils.build_output_paths(source, output_dir, request_id)

        success, first_frame = capture.read()
        if not success or first_frame is None:
            capture.release()
            raise ValueError(f"Unable to read frames from source: {source}")

        source_fps, target_fps, w, h, total_frames = self._resolve_video_properties(
            capture, first_frame, frames_per_second
        )
        sample_interval = max(1, int(round(source_fps / target_fps))) if target_fps > 0 else 1
        debug_dir = request_folder / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        writer = self._create_writer(output_path, source_fps, w, h, capture)

        logger.info(
            "video_source_opened",
            source=source,
            source_fps=source_fps,
            target_fps=target_fps,
            sample_interval=sample_interval,
            width=w,
            height=h,
            total_frames=total_frames,
            output_path=output_path,
            is_ocr_enabled=is_ocr_enabled,
        )

        video_tracker = _make_tracker(source_fps)
        all_detections: list[dict[str, Any]] = []
        frame_idx = 0
        frame_count = 0
        processed_count = 0
        frames_batch: list[np.ndarray] = [first_frame]
        indices_batch: list[int] = [0]
        t_start = time.perf_counter()

        try:
            while True:
                ok, frame = capture.read()
                if ok and frame is not None:
                    frame_idx += 1
                    if (frame_idx % sample_interval) == 0:
                        frames_batch.append(frame)
                        indices_batch.append(frame_idx)
                    else:
                        writer.write(frame)
                        frame_count += 1

                reached_end = not ok or frame is None

                if len(frames_batch) < settings.batch_size and not reached_end:
                    continue
                if not frames_batch:
                    if reached_end:
                        break
                    continue

                try:
                    payloads = await self._run_inference_batch(
                        frames_batch,
                        indices_batch,
                        request_id,
                        source_name,
                        ocr_confidence_threshold,
                        is_ocr_enabled,
                        platform,
                    )
                    processed_count += self._process_inference_batch(
                        frames=frames_batch,
                        indices=indices_batch,
                        payloads_batch=payloads,
                        debug_dir=debug_dir,
                        tracker=video_tracker,
                        writer=writer,
                        source_fps=source_fps,
                        all_detections=all_detections,
                        request_id=request_id or "",
                        camera_id=camera_id,
                        confidence_threshold=confidence_threshold,
                        ocr_confidence_threshold=ocr_confidence_threshold,
                        ocr_match_confidence=ocr_match_confidence,
                        global_id_match_score=global_id_match_score,
                        plate_text_mode=plate_text_mode,
                        is_ocr_enabled=is_ocr_enabled,
                    )
                except Exception as e:
                    logger.error(
                        "video_source_batch_error",
                        frames=indices_batch,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                    for f in frames_batch:
                        writer.write(f)

                frame_count += len(frames_batch)
                frames_batch.clear()
                indices_batch.clear()

                if reached_end:
                    break

            all_detections = self._enrich_detections_with_analytics(
                all_detections,
                camera_id,
                lat,
                lon,
                pixels_per_meter,
                zones or [],
                behavior_config,
                w,
                h,
                request_id or "",
            )

            csv_path, summary_csv_path = self._write_csv_outputs(output_path, all_detections, save_csv)
            return self._build_result(
                source=source,
                source_name=source_name,
                output_path=output_path,
                csv_path=csv_path,
                summary_csv_path=summary_csv_path,
                source_fps=source_fps,
                target_fps=target_fps,
                sample_interval=sample_interval,
                frame_count=frame_count,
                processed_count=processed_count,
                all_detections=all_detections,
                t_start=t_start,
                camera_id=camera_id,
                lat=lat,
                lon=lon,
                pixels_per_meter=pixels_per_meter,
                zones=zones or [],
                behavior_config=behavior_config,
                is_ocr_enabled=is_ocr_enabled,
                platform=platform,
            )
        finally:
            capture.release()
            writer.release()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_csv_outputs(
        output_path: str,
        detections: list[dict[str, Any]],
        save_csv: bool,
    ) -> tuple[str | None, str | None]:
        if not (save_csv and detections):
            return None, None
        csv_path = output_serializers.write_frame_detections_csv(
            detections=detections,
            csv_path=Path(output_path).with_suffix(".csv"),
            header=settings.CSV_FRAME_HEADER,
        )
        summary = None
        if csv_path:
            summary = output_serializers.write_track_summary_csv(
                detections=detections,
                frame_csv_path=Path(csv_path),
                header=settings.CSV_TRACK_SUMMARY_HEADER,
            )
        return csv_path, summary

    def _enrich_detections_with_analytics(
        self,
        detections: list[dict[str, Any]],
        camera_id: str,
        lat: float | None,
        lon: float | None,
        pixels_per_meter: float | None,
        zones: list[dict[str, Any]],
        behavior_config: dict[str, Any] | None,
        frame_width: int,
        frame_height: int,
        request_id: str = "",
    ) -> list[dict[str, Any]]:
        if not detections:
            return detections

        for det in detections:
            det["camera_id"] = camera_id

        enriched = detections

        if self.spatial_service:
            try:
                enriched = self.spatial_service.enrich_detections_with_spatial_state(
                    enriched, zones, pixels_per_meter, frame_width, frame_height
                )
                logger.info("spatial_analytics_applied", detection_count=len(enriched))
            except Exception as e:
                logger.error("spatial_analytics_error", error=str(e))

        if self.behavioral_service:
            try:
                enriched = self.behavioral_service.enrich_detections_with_behavior_state(
                    enriched, behavior_config, camera_id, request_id
                )
                logger.info("behavioral_analytics_applied", detection_count=len(enriched))
            except Exception as e:
                logger.error("behavioral_analytics_error", error=str(e))

        return enriched

    @staticmethod
    def _build_result(
        *,
        source: str,
        source_name: str,
        output_path: str,
        csv_path: str | None,
        summary_csv_path: str | None,
        source_fps: float,
        target_fps: float,
        sample_interval: int,
        frame_count: int,
        processed_count: int,
        all_detections: list[dict[str, Any]],
        t_start: float,
        camera_id: str = "",
        lat: float | None = None,
        lon: float | None = None,
        pixels_per_meter: float | None = None,
        zones: list[dict[str, Any]] = None,
        behavior_config: dict[str, Any] | None = None,
        is_ocr_enabled: bool,
        platform: str,
    ) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - t_start, 1e-9)
        pfps = processed_count / elapsed
        dur = (frame_count / source_fps) if source_fps > 0 else 0.0
        logger.info(
            "video_processing_complete",
            source=source,
            total_frames=frame_count,
            processed_frames=processed_count,
            detections=len(all_detections),
            processing_time_sec=round(elapsed, 2),
            processed_fps=round(pfps, 3),
        )
        result = {
            "video_path": source,
            "source_name": source_name,
            "output_path": output_path,
            "csv_path": csv_path,
            "summary_csv_path": summary_csv_path,
            "input_video_fps": round(source_fps, 2),
            "input_video_duration_sec": round(dur, 2),
            "target_fps": round(target_fps, 2),
            "sample_interval": sample_interval,
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "processed_fps": round(pfps, 3),
            "total_detections": len(all_detections),
            "detections": all_detections,
        }
        if camera_id:
            result["camera_id"] = camera_id
        if lat is not None:
            result["lat"] = lat
        if lon is not None:
            result["lon"] = lon
        if pixels_per_meter is not None:
            result["pixels_per_meter"] = pixels_per_meter
        if zones:
            result["zones"] = zones
        if behavior_config:
            result["behavior_config"] = behavior_config
        return result
