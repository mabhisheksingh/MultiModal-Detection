"""ANPR Pipeline - sends images to server-side anpr_pipeline model."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from utils.image_utils import load_image
from utils.triton_client import TritonClient

# Static configuration
DEFAULT_SERVER_URL = "127.0.0.1:8001"
DEFAULT_IMAGE_NAME = "frame_0000.jpg"
DEFAULT_OUTPUT_PATH = "assets/car1_MH46X9996.jpg"
DEFAULT_MODEL_NAME = "anpr_pipeline"


class ANPRPipeline:
    """Client for server-side ANPR pipeline model."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, model_name: str = DEFAULT_MODEL_NAME):
        self.client = TritonClient(server_url)
        self.model_name = model_name

    def run(self, image: np.ndarray) -> dict[str, Any]:
        """Run ANPR pipeline by sending image to server-side model."""
        total_start = time.perf_counter()

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Add batch dimension if not present
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Prepare input and output for Triton
        inp = self.client.create_input(image, "image")
        out = self.client.create_output("anpr_results")

        # Send image to server-side anpr_pipeline model
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[inp],
            outputs=[out],
            request_id=request_id,
        )

        total_time = (time.perf_counter() - total_start) * 1000

        # Parse JSON response
        results_json = response.as_numpy("anpr_results")[0]
        results = json.loads(results_json)

        return {
            "results": results,
            "timing": {
                "total_ms": total_time,
            },
        }

    def run_batch(self, images: list[tuple[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
        """Run ANPR pipeline on batch of images with request IDs.

        Args:
            images: List of (request_id, image) tuples

        Returns:
            Dict mapping request_id to dict with results and timing information
        """
        batch_results = {}
        for request_id, image in images:
            result = self.run(image)
            batch_results[request_id] = result
        return batch_results

    def run_from_path(self, image_source: str) -> dict[str, Any]:
        """Load image (from assets/ if just a filename) and run ANPR pipeline."""
        img = load_image(image_source)
        return self.run(img)

    def visualize(self, image: np.ndarray, result_dict: dict[str, Any], output_path: str | None = None):
        """Draw detections on image and optionally save."""
        vis = image.copy()
        results = result_dict["results"] if isinstance(result_dict, dict) else result_dict
        for r in results:
            v = r["vehicle"]
            vx1, vy1, vx2, vy2 = v["bbox"]
            cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{v['class']}: {v['conf']:.2f}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            for p in r["plates"]:
                px1, py1, px2, py2 = p["bbox"]
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
                label = f"{p['text']} ({p['text_confidence']:.2f})"
                cv2.putText(vis, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis)
            print(f"[ANPRPipeline] Saved visualization to {output_path}")
        return vis


def main():
    import sys

    pipeline = ANPRPipeline()
    img_name = DEFAULT_IMAGE_NAME
    if len(sys.argv) > 1:
        img_name = sys.argv[1]

    result_dict = pipeline.run_from_path(img_name)
    results = result_dict["results"]
    timing = result_dict["timing"]

    print(f"Found {len(results)} vehicles.")
    for r in results:
        v = r["vehicle"]
        print(f"  Vehicle: {v['class']} (conf={v['conf']:.2f})")
        for p in r["plates"]:
            print(f"    Plate: '{p['text']}' (conf={p['confidence']:.2f})")

    print("\nTiming Information:")
    print(f"  Total time: {timing['total_ms']:.2f} ms")

    pipeline.visualize(load_image(img_name), result_dict, str(Path(__file__).resolve().parent / DEFAULT_OUTPUT_PATH))


if __name__ == "__main__":
    main()
