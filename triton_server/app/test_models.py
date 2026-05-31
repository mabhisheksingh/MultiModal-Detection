"""Single test file for all Triton models via gRPC."""

import sys
import time
from pathlib import Path
from typing import Dict

from utils.triton_client import TritonClient
from utils.image_utils import load_image
from vehicle_detector import VehicleDetector
from plate_detector import PlateDetector
from ocr_triton import TritonOCR

TRITON_URL = "localhost:9001"
TEST_IMAGE = "frame_0000.jpg"


def print_result(name: str, result: Dict):
    emoji = {"success": "✓", "error": "✗", "inference_failed": "✗", "not_tested": "-"}
    print(f"\n{emoji.get(result['status'], '?')} {name}")
    print(f"  Status: {result['status']}")
    if result.get("inference_time"):
        print(f"  Time: {result['inference_time']}")
    if result.get("output_shape"):
        print(f"  Output Shape: {result['output_shape']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")


def test_vehicle():
    result = {"status": "not_tested", "inference_time": None, "output_shape": None, "error": None}
    try:
        detector = VehicleDetector(server_url=TRITON_URL)
        start = time.time()
        dets = detector.detect_from_path(TEST_IMAGE)
        result["status"] = "success"
        result["inference_time"] = f"{time.time() - start:.3f}s"
        result["output_shape"] = f"{len(dets)} detections"
        print(f"  Detections: {[d['class_name'] for d in dets]}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    print_result("Vehicle Detection", result)
    return result


def test_plate():
    result = {"status": "not_tested", "inference_time": None, "output_shape": None, "error": None}
    try:
        detector = PlateDetector(server_url=TRITON_URL)
        start = time.time()
        dets = detector.detect_from_path(TEST_IMAGE)
        result["status"] = "success"
        result["inference_time"] = f"{time.time() - start:.3f}s"
        result["output_shape"] = f"{len(dets)} detections"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    print_result("Plate Detection", result)
    return result


def test_ocr():
    result = {"status": "not_tested", "inference_time": None, "output_shape": None, "error": None}
    try:
        ocr = TritonOCR(server_url=TRITON_URL)
        img = load_image(TEST_IMAGE)
        start = time.time()
        texts = ocr.recognize(img)
        result["status"] = "success"
        result["inference_time"] = f"{time.time() - start:.3f}s"
        result["output_shape"] = f"{len(texts)} texts"
        print(f"  Texts: {texts}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    print_result("OCR (Triton)", result)
    return result


def main():
    print("=" * 60)
    print("Triton Model Test Suite")
    print(f"Server: {TRITON_URL}")
    print(f"Image:  {TEST_IMAGE}")
    print("=" * 60)

    client = TritonClient(TRITON_URL)
    if not client.is_server_live():
        print("✗ Triton server is not live")
        sys.exit(1)
    print("✓ Server is live")

    results = [test_vehicle(), test_plate(), test_ocr()]
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] in ["error", "inference_failed"])
    print(f"\nSummary: {success} passed, {failed} failed / {len(results)} tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
