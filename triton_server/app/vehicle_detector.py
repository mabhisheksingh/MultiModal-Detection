"""Vehicle detection client - calls vehicle_detection_rt_detr on Triton via gRPC."""

from typing import Any

import numpy as np
from utils.image_utils import load_image
from utils.triton_client import TritonClient


class VehicleDetector:
    """Vehicle detector using Triton Inference Server."""

    MODEL_NAME = "vehicle_detection_rt_detr"
    INPUT_NAME = "images"
    OUTPUT_NAME = "output0"
    INPUT_SIZE = (640, 640)

    CLASS_MAP = {
        0: "animal",
        1: "autorickshaw",
        2: "bicycle",
        3: "bus",
        4: "car",
        5: "caravan",
        6: "motorcycle",
        7: "person",
        8: "rider",
        9: "traffic light",
        10: "traffic sign",
        11: "trailer",
        12: "train",
        13: "truck",
        14: "vehicle fallback",
    }

    def __init__(self, triton_client: TritonClient | None = None, server_url: str = "127.0.0.1:9001"):
        self.client = triton_client or TritonClient(server_url)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Letterbox preprocess for RT-DETR."""
        return self.client.letterbox_preprocess(image, self.INPUT_SIZE)

    def postprocess(
        self,
        raw_output: np.ndarray,
        meta: dict[str, Any],
        original_shape: tuple[int, int],
        conf_thresh: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Parse RT-DETR [1, 300, N] output to detection list."""
        detections = raw_output[0]
        results = []
        h_orig, w_orig = original_shape

        for row in detections:
            scores = row[4:]
            conf = float(np.max(scores))
            if conf < conf_thresh:
                continue
            class_id = int(np.argmax(scores))
            cx, cy, w, h = row[:4]

            x1 = ((cx - w / 2) * 640 - meta["x_off"]) / meta["scale"]
            y1 = ((cy - h / 2) * 640 - meta["y_off"]) / meta["scale"]
            x2 = ((cx + w / 2) * 640 - meta["x_off"]) / meta["scale"]
            y2 = ((cy + h / 2) * 640 - meta["y_off"]) / meta["scale"]

            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "class_id": class_id,
                    "class_name": self.CLASS_MAP.get(class_id, "unknown"),
                }
            )
        return results

    def detect(self, image: np.ndarray, conf_thresh: float = 0.4) -> list[dict[str, Any]]:
        """Run vehicle detection on an image."""
        blob, meta = self.preprocess(image)
        inp = self.client.create_input(blob, self.INPUT_NAME)
        out = self.client.create_output(self.OUTPUT_NAME)
        response = self.client.infer(self.MODEL_NAME, [inp], [out])
        raw = response.as_numpy(self.OUTPUT_NAME)
        return self.postprocess(raw, meta, image.shape[:2], conf_thresh)

    def detect_from_path(self, image_source: str, conf_thresh: float = 0.4) -> list[dict[str, Any]]:
        """Load image (from assets/ if just a filename) and run detection."""
        img = load_image(image_source)
        return self.detect(img, conf_thresh)


def main():
    import sys

    detector = VehicleDetector()
    img_name = "frame_0000.jpg"
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    dets = detector.detect_from_path(img_name)
    print(f"Found {len(dets)} vehicles.")
    for d in dets:
        print(f"  {d['class_name']}: conf={d['conf']:.2f}, bbox={d['bbox']}")


if __name__ == "__main__":
    main()
