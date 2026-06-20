"""
OCR Engine implementations using PaddleOCR.
"""

import base64
import os
from statistics import mean
from typing import Any

import cv2
import numpy as np
import structlog

from triton_server.app import OCRUtils, settings

logger = structlog.get_logger(__name__)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def _import_paddle_dependencies() -> tuple[Any, Any]:
    import paddle
    from paddleocr import PaddleOCR

    return paddle, PaddleOCR


class PaddleOCREngine:
    """PaddleOCR wrapper for license plate text recognition."""

    def __init__(
        self,
        use_textline_orientation: bool = True,
        lang: str = "en",
    ) -> None:
        """
        Initialize PaddleOCR engine.

        Args:
            use_textline_orientation: Whether to use text line orientation
            lang: Language code for OCR
        """
        logger.debug(f"Initializing PaddleOCR engine, lang={lang}, use_textline_orientation={use_textline_orientation}")
        import logging

        # 1. Save your original structlog root log level (usually INFO or DEBUG)
        root_logger = logging.getLogger()
        original_log_level = root_logger.level

        # 2. Import Paddle (This is where Paddle secretly changes the root level to WARNING)
        self._paddle, paddle_ocr_class = _import_paddle_dependencies()

        # 3. RESTORE your log level immediately after the import!
        root_logger.setLevel(original_log_level)

        # 4. Hook into the 'ppocr' logger and force it to propagate
        pp_logger = logging.getLogger("ppocr")
        pp_logger.setLevel(original_log_level)
        pp_logger.propagate = True

        self._device = self._select_device()

        # 5. Initialize the reader
        self.reader = paddle_ocr_class(
            use_textline_orientation=use_textline_orientation,
            lang=lang,
            ocr_version="PP-OCRv5",
        )

        # 6. Restore it one more time just in case the class initialization hijacked it again
        root_logger.setLevel(original_log_level)
        print("testng")
        logger.debug("PaddleOCR engine initialized")

    def _log_prediction_image(self, image: np.ndarray, *, stage: str) -> None:
        """Log the OCR input image as base64 for debugging purposes."""

        try:
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                raise ValueError("cv2.imencode returned success=False")
            encoded = base64.b64encode(buffer).decode("ascii")
            logger.debug(
                "ocr_prediction_image",
                stage=stage,
                image_shape=image.shape,
                image_base64=encoded,
            )
        except Exception as exc:
            logger.warning(
                "ocr_prediction_image_logging_failed",
                stage=stage,
                error=str(exc),
            )

    def _select_device(self) -> str:
        if self._paddle.device.is_compiled_with_cuda():
            try:
                self._paddle.set_device("gpu:0")
                logger.info("PaddleOCR using GPU device gpu:0")
                return "gpu:0"
            except Exception as exc:
                logger.warning(f"Failed to select gpu:0 for PaddleOCR, falling back to CPU: {exc}")
        self._paddle.set_device("cpu")
        logger.info("PaddleOCR using CPU device")
        return "cpu"

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR readability.

        Args:
            image: Input image (BGR)

        Returns:
            Preprocessed image
        """

        if not settings.ocr_enable_preprocessing:
            return image

        # 1. Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. Bilateral Filter: Smooths background dirt while preserving character edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. CLAHE: Localized contrast enhancement
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=settings.ocr_clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)

        # 4. Kernel Sharpening: Makes text boundaries stark and clear
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Convert back to BGR for PaddleOCR
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return processed

    def recognize(self, image: np.ndarray, *, plate_text_mode: str) -> list[dict]:
        """
        Perform OCR on an image.

        Args:
            image: Input image (BGR or grayscale)
            plate_text_mode: OCR validation mode - "strict" (regex validation) or "balanced" (no validation)

        Returns:
            List of dicts with 'text' and 'confidence' keys
        """
        logger.info("ocr_inference_started", plate_text_mode=plate_text_mode)
        # Ensure image is in BGR format for PaddleOCR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        original_shape = tuple(image.shape)
        logger.debug(
            "ocr_inference_started",
            shape=original_shape,
            device=self._device,
            plate_text_mode=plate_text_mode,
        )

        # Preprocess for better readability
        processed_image = self._preprocess_for_ocr(image)
        self._log_prediction_image(processed_image, stage="pre_predict")

        # paddleocr 3.x API with configurable thresholds
        logger.info(
            "ocr_predict_invoked",
            shape=processed_image.shape,
            det_limit_side_len=settings.ocr_det_limit_side_len,
            det_box_thresh=settings.ocr_det_box_thresh,
            rec_score_thresh=settings.ocr_rec_score_thresh,
            det_thresh=settings.ocr_det_thresh,
            det_unclip_ratio=settings.ocr_det_unclip_ratio,
            mode=plate_text_mode,
        )
        ocr_result = self.reader.predict(
            processed_image,
            text_det_limit_side_len=settings.ocr_det_limit_side_len,
            text_det_box_thresh=settings.ocr_det_box_thresh,
            text_rec_score_thresh=settings.ocr_rec_score_thresh,
            text_det_unclip_ratio=settings.ocr_det_unclip_ratio,
            text_det_thresh=settings.ocr_det_thresh,
        )
        logger.info("ocr_predict_completed", result_count=len(ocr_result))

        results = []
        plate_text_mode = plate_text_mode.lower()

        for res in ocr_result:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])

            if texts and scores:
                logger.info(f"OCR result: {texts} with confidence {scores}")

                combined_raw_text = "".join(str(text or "") for text in texts)
                combined_text = OCRUtils.normalize_plate_text(combined_raw_text)
                combined_score = float(mean(float(score) for score in scores))
                is_valid = OCRUtils.validate_plate_text(combined_text, mode=plate_text_mode)

                logger.debug(
                    "ocr_candidate_evaluated",
                    raw_text=combined_raw_text,
                    normalized_text=combined_text,
                    confidence=combined_score,
                    mode=plate_text_mode,
                    is_valid=is_valid,
                    candidate_type="combined",
                )

                if is_valid:
                    logger.debug(
                        "ocr_candidate_selected",
                        text=combined_text,
                        confidence=combined_score,
                        mode=plate_text_mode,
                    )
                    results.append({"text": combined_text, "confidence": combined_score})
                else:
                    logger.warning(
                        "ocr_no_valid_candidate",
                        texts=texts,
                        scores=scores,
                        combined_text=combined_text,
                        combined_score=combined_score,
                        image_shape=original_shape,
                        mode=plate_text_mode,
                        reason=f"All candidates rejected (mode={plate_text_mode})",
                    )

        if not results:
            logger.warning(
                "ocr_inference_no_results",
                image_shape=original_shape,
                ocr_result_count=len(ocr_result),
                reason="No text detected or all candidates rejected",
            )

        logger.debug("ocr_inference_completed", result_count=len(results))
        return results
