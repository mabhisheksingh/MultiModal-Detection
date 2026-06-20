from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog

from triton_server.app import DetectionResponseItem

logger = structlog.get_logger(__name__)


def build_detection_response_item(item: DetectionResponseItem) -> dict[str, Any]:
    """Serialize a validated detection response item."""
    return item.model_dump()


def aggregate_track_votes(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate OCR results by track ID to find the most common plate text per track."""
    track_votes: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "confidence_sum": 0.0, "ocr_confidence_sum": 0.0})
    )

    for det in detections or []:
        track_id_raw = det.get("track_id")
        track_id = int(track_id_raw) if track_id_raw is not None else -1
        plate_text = (det.get("ocr_text") or "").strip()
        if not plate_text:
            continue

        entry = track_votes[track_id][plate_text]
        entry["count"] += 1
        entry["confidence_sum"] += float(det.get("conf", 0.0))
        entry["ocr_confidence_sum"] += float(det.get("ocr_confidence", 0.0))

    rows: list[dict[str, Any]] = []
    for track_id, votes in track_votes.items():

        def vote_key(item):
            stats = item[1]
            count = stats["count"]
            avg_ocr = stats["ocr_confidence_sum"] / count if count else 0.0
            avg_conf = stats["confidence_sum"] / count if count else 0.0
            return (count, avg_ocr, avg_conf)

        best_text, stats = max(votes.items(), key=vote_key)
        count = stats["count"]
        avg_conf = stats["confidence_sum"] / count if count else 0.0
        avg_ocr = stats["ocr_confidence_sum"] / count if count else 0.0

        rows.append(
            {
                "track_id": track_id,
                "plate_text": best_text,
                "votes": count,
                "avg_confidence": round(avg_conf, 4),
                "avg_confidence_ocr": round(avg_ocr, 4),
            }
        )

    return rows


def build_source_summary_rows(source: str, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build summary rows with source information for consolidated outputs."""
    rows = aggregate_track_votes(detections)
    for row in rows:
        row["source"] = source
    return rows


def write_track_summary_csv(
    *,
    detections: list[dict[str, Any]],
    frame_csv_path: Path,
    header: list[str],
) -> str | None:
    """Write track-level summary CSV with aggregated OCR results."""
    rows = aggregate_track_votes(detections)
    if not rows:
        logger.info("track_summary_skipped", reason="no_ocr_results")
        return None

    summary_path = frame_csv_path.with_name(f"{frame_csv_path.stem}_track_summary.csv")

    try:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "track_summary_csv_saved",
            path=str(summary_path),
            track_count=len(rows),
        )
        return str(summary_path)
    except Exception as exc:
        logger.error("track_summary_csv_error", error=str(exc), path=str(summary_path))
        return None


def write_frame_detections_csv(
    *,
    detections: list[dict[str, Any]],
    csv_path: Path,
    header: list[str],
) -> str | None:
    """Write frame-level detection CSV for video processing."""
    if not detections:
        logger.info("frame_csv_skipped", reason="no_detections")
        return None

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for det in detections:
                writer.writerow(
                    {
                        "frame": det.get("frame_id", 0),
                        "track_id": det.get("track_id", "-1"),
                        "plate_text": det.get("ocr_text", ""),
                        "confidence": det.get("conf", 0.0),
                        "confidence_ocr": det.get("ocr_confidence", 0.0),
                    }
                )
        logger.info("frame_csv_saved", path=str(csv_path), detections=len(detections))
        return str(csv_path)
    except Exception as exc:
        logger.error("frame_csv_error", error=str(exc), path=str(csv_path))
        return None


def write_csv_rows(
    *,
    csv_path: Path,
    rows: list[dict[str, Any]],
    header: list[str],
) -> str | None:
    """Write generic CSV rows with specified header."""
    if not rows:
        logger.info("csv_write_skipped", reason="no_rows", path=str(csv_path))
        return None

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("csv_saved", path=str(csv_path), rows=len(rows))
        return str(csv_path)
    except Exception as exc:
        logger.error("csv_write_error", path=str(csv_path), error=str(exc))
        return None
