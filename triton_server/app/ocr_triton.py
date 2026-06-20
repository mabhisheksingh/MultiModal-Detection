"""OCR client via Triton gRPC - calls paddle_ocr_gpu_batched model."""

from typing import Any

import cv2
import numpy as np
from utils.image_utils import load_image
from utils.triton_client import TritonClient


class TritonOCR:
    """OCR via Triton Inference Server (paddle_ocr_gpu_batched model)."""

    MODEL_NAME = "paddle_ocr_gpu_batched"
    INPUT_NAME = "INPUT1"
    OUTPUT_NAME = "OUTPUT0"

    def __init__(self, triton_client: TritonClient | None = None, server_url: str = "127.0.0.1:9001"):
        self.client = triton_client or TritonClient(server_url)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 HWC format for OCR."""
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def recognize(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run OCR on a single image. Returns list of {text, confidence}."""
        img = self.preprocess(image)
        inp = self.client.create_input(np.expand_dims(img, axis=0), self.INPUT_NAME)
        out = self.client.create_output(self.OUTPUT_NAME)
        response = self.client.infer(self.MODEL_NAME, [inp], [out])
        raw = response.as_numpy(self.OUTPUT_NAME)

        results = []
        for item in raw.flatten():
            text = str(item)
            if "confidence:" in text:
                parts = text.rsplit("confidence:", 1)
                conf = float(parts[1].strip().rstrip(")"))
                txt = parts[0].strip()
                results.append({"text": txt, "confidence": conf})
            else:
                results.append({"text": text, "confidence": 0.0})
        return results

    def recognize_batch(self, images: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """Run OCR on a batch of images."""
        batch = np.array([self.preprocess(img) for img in images])
        inp = self.client.create_input(batch, self.INPUT_NAME)
        out = self.client.create_output(self.OUTPUT_NAME)
        response = self.client.infer(self.MODEL_NAME, [inp], [out])
        raw = response.as_numpy(self.OUTPUT_NAME)

        all_results = []
        for item in raw:
            texts = []
            for t in item.flatten() if hasattr(item, "flatten") else [item]:
                text = str(t)
                if "confidence:" in text:
                    parts = text.rsplit("confidence:", 1)
                    conf = float(parts[1].strip().rstrip(")"))
                    txt = parts[0].strip()
                    texts.append({"text": txt, "confidence": conf})
                else:
                    texts.append({"text": text, "confidence": 0.0})
            all_results.append(texts)
        return all_results


def main():
    import sys

    ocr = TritonOCR()
    img_name = "car1_MH46X9996.jpg"
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = load_image(img_name)
    results = ocr.recognize(img)
    print(f"OCR results: {results}")


if __name__ == "__main__":
    main()
