"""Standalone OCR using local PaddleOCR (no Triton server required)."""

import os
from statistics import mean
from typing import Any

import cv2
import numpy as np
from utils.image_utils import load_image_rgb

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

OCR_CONFIG = {
    "use_textline_orientation": True,
    "lang": "en",
    "ocr_version": "PP-OCRv5",
    "enable_preprocessing": True,
    "clahe_clip_limit": 2.0,
    "det_limit_side_len": 960,
    "det_box_thresh": 0.5,
    "rec_score_thresh": 0.5,
    "det_thresh": 0.3,
    "det_unclip_ratio": 1.6,
}


class StandaloneOCR:
    """Standalone PaddleOCR processor."""

    def __init__(self, use_cpu: bool = True):
        import paddle
        from paddleocr import PaddleOCR

        self._paddle = paddle
        if use_cpu:
            self._paddle.set_device("cpu")
            print("[StandaloneOCR] Using CPU device")

        self.reader = PaddleOCR(
            use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
            lang=OCR_CONFIG["lang"],
            ocr_version=OCR_CONFIG["ocr_version"],
        )
        print(f"[StandaloneOCR] PaddleOCR initialized ({OCR_CONFIG['ocr_version']})")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR readability."""
        if not OCR_CONFIG["enable_preprocessing"]:
            return image
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=OCR_CONFIG["clahe_clip_limit"], tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def recognize(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run OCR on an image."""
        processed = self.preprocess(image)
        result = self.reader.predict(
            processed,
            text_det_limit_side_len=OCR_CONFIG["det_limit_side_len"],
            text_det_box_thresh=OCR_CONFIG["det_box_thresh"],
            text_rec_score_thresh=OCR_CONFIG["rec_score_thresh"],
            text_det_unclip_ratio=OCR_CONFIG["det_unclip_ratio"],
            text_det_thresh=OCR_CONFIG["det_thresh"],
        )

        texts = []
        for res in result:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            if rec_texts and rec_scores:
                combined_text = "".join(str(t or "") for t in rec_texts)
                combined_score = float(mean(float(s) for s in rec_scores))
                texts.append({"text": combined_text, "confidence": combined_score})
        return texts


def main():
    import sys

    ocr = StandaloneOCR(use_cpu=True)
    img_name = "car1_MH46X9996.jpg"
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = load_image_rgb(img_name)
    results = ocr.recognize(img)
    print(f"Detected {len(results)} text(s):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['text']} (confidence: {r['confidence']:.2f})")


if __name__ == "__main__":
    main()
