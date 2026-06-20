"""Visualization helpers for bounding boxes and detection results."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on an image."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if show_conf and "conf" in det:
            label = f"{det.get('class_name', 'obj')}: {det['conf']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return vis


def draw_normalized_detections(
    image_path: str,
    data_path: str,
    output_path: str,
    confidence_threshold: float = 0.2,
    label: str = "Plate",
):
    """Load normalized detections from JSON/txt and draw them."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img_h, img_w = image.shape[:2]

    with open(data_path) as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            content = f.read().replace("'", '"')
            raw_data = json.loads(content)

    detections = np.array(raw_data)
    valid = detections[detections[:, 4] >= confidence_threshold]
    print(f"Loaded {len(valid)} detections above threshold {confidence_threshold}.")

    for i, det in enumerate(valid):
        cx, cy, w, h, conf = det
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        color = (0, 255, 0) if conf > 0.2 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Detection {i + 1}: Conf={conf:.4f}, Box=[{x1}, {y1}, {x2}, {y2}]")

    cv2.imwrite(str(output_path), image)
    print(f"Saved marked image to {output_path}")


def main():
    base = Path(__file__).resolve().parent.parent / "assets"
    draw_normalized_detections(
        image_path=base / "frame_0000.jpg",
        data_path=base / "xyxrc.txt",
        output_path=base / "frame_0000_marked.jpg",
        confidence_threshold=0.2,
    )


if __name__ == "__main__":
    main()
