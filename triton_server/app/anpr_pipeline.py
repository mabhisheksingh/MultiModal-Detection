"""ANPR Pipeline - orchestrates vehicle detection -> plate detection -> OCR via Triton gRPC."""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from utils.triton_client import TritonClient
from utils.image_utils import load_image
from vehicle_detector import VehicleDetector
from plate_detector import PlateDetector
from ocr_triton import TritonOCR


class ANPRPipeline:
    """End-to-end ANPR pipeline using Triton models."""

    def __init__(self, server_url: str = "127.0.0.1:8001"):
        self.client = TritonClient(server_url)
        self.vehicle = VehicleDetector(self.client)
        self.plate = PlateDetector(self.client)
        self.ocr = TritonOCR(self.client)

    def run(self, image: np.ndarray, vehicle_conf: float = 0.4, plate_conf: float = 0.1) -> List[Dict[str, Any]]:
        """Run full ANPR pipeline. Returns list of vehicle dicts with plates and OCR text."""
        vehicles = self.vehicle.detect(image, conf_thresh=vehicle_conf)
        results = []

        for v in vehicles:
            vehicle_result = {
                "vehicle": {
                    "bbox": v["bbox"],
                    "conf": v["conf"],
                    "class": v["class_name"],
                },
                "plates": []
            }

            x1, y1, x2, y2 = v["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            vehicle_crop = image[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                results.append(vehicle_result)
                continue

            plates = self.plate.detect(vehicle_crop, conf_thresh=plate_conf)

            for p in plates:
                px1, py1, px2, py2 = p["bbox"]
                plate_bbox = [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
                plate_crop = image[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                if plate_crop.size == 0:
                    continue

                ocr_results = self.ocr.recognize(plate_crop)
                plate_text = ocr_results[0]["text"] if ocr_results else ""
                plate_conf = ocr_results[0]["confidence"] if ocr_results else 0.0

                vehicle_result["plates"].append({
                    "bbox": plate_bbox,
                    "confidence": p["confidence"],
                    "text": plate_text,
                    "text_confidence": plate_conf,
                })

            results.append(vehicle_result)

        return results

    def run_from_path(self, image_source: str, **kwargs) -> List[Dict[str, Any]]:
        """Load image (from assets/ if just a filename) and run ANPR pipeline."""
        img = load_image(image_source)
        return self.run(img, **kwargs)

    def visualize(self, image: np.ndarray, results: List[Dict[str, Any]], output_path: Optional[str] = None):
        """Draw detections on image and optionally save."""
        vis = image.copy()
        for r in results:
            v = r["vehicle"]
            vx1, vy1, vx2, vy2 = v["bbox"]
            cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(vis, f"{v['class']}: {v['conf']:.2f}", (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for p in r["plates"]:
                px1, py1, px2, py2 = p["bbox"]
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
                label = f"{p['text']} ({p['text_confidence']:.2f})"
                cv2.putText(vis, label, (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis)
            print(f"[ANPRPipeline] Saved visualization to {output_path}")
        return vis


def main():
    import sys
    pipeline = ANPRPipeline()
    img_name = "frame_0000.jpg"
    if len(sys.argv) > 1:
        img_name = sys.argv[1]

    results = pipeline.run_from_path(img_name)
    print(f"Found {len(results)} vehicles.")
    for r in results:
        v = r["vehicle"]
        print(f"  Vehicle: {v['class']} (conf={v['conf']:.2f})")
        for p in r["plates"]:
            print(f"    Plate: '{p['text']}' (conf={p['confidence']:.2f})")

    pipeline.visualize(load_image(img_name), results, str(Path(__file__).resolve().parent / "assets" / "anpr_result.jpg"))


if __name__ == "__main__":
    main()
