"""Plate detection client - calls plate_region_detection_rt_detr on Triton via gRPC."""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from utils.triton_client import TritonClient
from utils.image_utils import load_image


class PlateDetector:
    """License plate detector using Triton Inference Server."""

    MODEL_NAME = "plate_region_detection_rt_detr"
    INPUT_NAME = "images"
    OUTPUT_NAME = "output0"
    INPUT_SIZE = (640, 640)

    def __init__(self, triton_client: Optional[TritonClient] = None, server_url: str = "127.0.0.1:9001"):
        self.client = triton_client or TritonClient(server_url)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Letterbox preprocess for plate detection."""
        return self.client.letterbox_preprocess(image, self.INPUT_SIZE)

    def postprocess(
        self,
        raw_output: np.ndarray,
        meta: Dict[str, Any],
        original_shape: Tuple[int, int],
        conf_thresh: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Parse plate detection output to bbox list."""
        if raw_output.ndim == 3:
            raw_output = raw_output[0]
        if raw_output.size == 0:
            return []

        h_orig, w_orig = original_shape
        th, tw = meta["target_size"]
        x_off = meta["x_off"]
        y_off = meta["y_off"]
        scale = max(meta["scale"], 1e-6)

        results = []
        for row in raw_output:
            if len(row) < 5:
                continue
            cx, cy, w, h, conf = row[:5]
            if conf < conf_thresh:
                continue

            x1 = (cx - w / 2) * tw
            y1 = (cy - h / 2) * th
            x2 = (cx + w / 2) * tw
            y2 = (cy + h / 2) * th

            x1 = int(max(0, min(w_orig, (x1 - x_off) / scale)))
            y1 = int(max(0, min(h_orig, (y1 - y_off) / scale)))
            x2 = int(max(0, min(w_orig, (x2 - x_off) / scale)))
            y2 = int(max(0, min(h_orig, (y2 - y_off) / scale)))

            if x2 <= x1 or y2 <= y1:
                continue
            results.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf)})

        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def detect(self, image: np.ndarray, conf_thresh: float = 0.1) -> List[Dict[str, Any]]:
        """Run plate detection on an image."""
        blob, meta = self.preprocess(image)
        inp = self.client.create_input(blob, self.INPUT_NAME)
        out = self.client.create_output(self.OUTPUT_NAME)
        response = self.client.infer(self.MODEL_NAME, [inp], [out])
        raw = response.as_numpy(self.OUTPUT_NAME)
        return self.postprocess(raw, meta, image.shape[:2], conf_thresh)

    def detect_from_path(self, image_source: str, conf_thresh: float = 0.1) -> List[Dict[str, Any]]:
        """Load image (from assets/ if just a filename) and run detection."""
        img = load_image(image_source)
        return self.detect(img, conf_thresh)


def main():
    import sys
    detector = PlateDetector()
    img_name = "frame_0000.jpg"
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    dets = detector.detect_from_path(img_name)
    print(f"Found {len(dets)} plates.")
    for d in dets:
        print(f"  conf={d['confidence']:.2f}, bbox={d['bbox']}")


if __name__ == "__main__":
    main()
