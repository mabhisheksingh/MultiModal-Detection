import numpy as np
import cv2
import os
import triton_python_backend_utils as pb_utils
from statistics import mean

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# OCR Configuration Map - using environment variables
OCR_CONFIG = {
    "use_textline_orientation": os.getenv("OCR_USE_TEXTLINE_ORIENTATION", "true").lower() == "true",
    "lang": os.getenv("OCR_LANG", "en"),
    "ocr_version": os.getenv("OCR_VERSION", "PP-OCRv5"),
    "enable_preprocessing": os.getenv("OCR_ENABLE_PREPROCESSING", "true").lower() == "true",
    "clahe_clip_limit": float(os.getenv("OCR_CLAHE_CLIP_LIMIT", "2.0")),
    "det_limit_side_len": int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "960")),
    "det_box_thresh": float(os.getenv("OCR_DET_BOX_THRESH", "0.5")),
    "rec_score_thresh": float(os.getenv("OCR_REC_SCORE_THRESH", "0.5")),
    "det_thresh": float(os.getenv("OCR_DET_THRESH", "0.3")),
    "det_unclip_ratio": float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.6")),
}

class TritonPythonModel:
    """Simple string echo model for testing Triton Python backend."""

    def initialize(self, args):
        pb_utils.Logger.log_info("TritonPythonModel initialize")
        
        # Import paddle dependencies
        import paddle
        from paddleocr import PaddleOCR
        
        self._paddle = paddle
        self._device = self._select_device()
        
        # Initialize PaddleOCR reader
        self.reader = PaddleOCR(
            use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
            lang=OCR_CONFIG["lang"],
            ocr_version=OCR_CONFIG["ocr_version"],
        )
        
        pb_utils.Logger.log_info(f"PaddleOCR initialized on device: {self._device}")


    def execute(self, requests):
        pb_utils.Logger.log_info(f"\n{10 * '*' }TritonPythonModel execution started {10 * '*' }\n")
        responses = []
        for request in requests:
            # Check which input is provided
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            if in_0 is not None:
                input_array = in_0.as_numpy()
                pb_utils.Logger.log_info(f"INPUT0 dtype={input_array.dtype}, shape={input_array.shape}")
                # Decode the first element (bytes -> str)
                raw = input_array.flat[0]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                pb_utils.Logger.log_info(f"Decoded input: {text}")
                # Build output as np.object_ array (same pattern as official QA string model)
                out_data = np.array([f"TEST SERVER: {text}"], dtype=object)
            elif in_1 is not None:
                # Handle INPUT1 (f32 image)
                input_array = in_1.as_numpy()
                pb_utils.Logger.log_info(f"INPUT1 dtype={input_array.dtype}, shape={input_array.shape}")
                process_image_response = self._process_image(input_array)
                # Process image data here
                out_data = np.array([process_image_response], dtype=object)
            else:
                pb_utils.Logger.log_info(f"No input comes in input0 and input1")
                out_data = np.array(["No input provided"], dtype=object)

            pb_utils.Logger.log_info(f"OUTPUT0 value: {out_data[0]}")

            out_tensor = pb_utils.Tensor("OUTPUT0", out_data)
            responses.append(pb_utils.InferenceResponse([out_tensor]))


        pb_utils.Logger.log_info(f"\n{10 * '*' }TritonPythonModel execution ended {10 * '*' }\n")
        return responses

    def _select_device(self) -> str:
        """Select GPU if available, otherwise fallback to CPU."""
        if self._paddle.device.is_compiled_with_cuda():
            try:
                self._paddle.set_device("gpu:0")
                pb_utils.Logger.log_info("PaddleOCR using GPU device gpu:0")
                return "gpu:0"
            except Exception as exc:
                pb_utils.Logger.log_warning(f"Failed to select gpu:0 for PaddleOCR, falling back to CPU: {exc}")
        self._paddle.set_device("cpu")
        pb_utils.Logger.log_info("PaddleOCR using CPU device")
        return "cpu"

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR readability."""
        if not OCR_CONFIG["enable_preprocessing"]:
            return image

        # Convert float32 to uint8 if needed (scale 0-255)
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral Filter: Smooths background dirt while preserving character edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE: Localized contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=OCR_CONFIG["clahe_clip_limit"],
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(blur)

        # Kernel Sharpening: Makes text boundaries stark and clear
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Convert back to BGR for PaddleOCR
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return processed

    def _process_image(self, image_array: np.ndarray) -> str:
        """Process image using PaddleOCR and return detected text."""
        pb_utils.Logger.log_info(f"Processing image with shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        try:
            # Ensure image is in BGR format for PaddleOCR
            if len(image_array.shape) == 2:
                image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                image = image_array
            
            # Preprocess for better readability
            processed_image = self._preprocess_for_ocr(image)
            
            # Perform OCR prediction using configuration from OCR_CONFIG
            ocr_result = self.reader.predict(
                processed_image,
                text_det_limit_side_len=OCR_CONFIG["det_limit_side_len"],
                text_det_box_thresh=OCR_CONFIG["det_box_thresh"],
                text_rec_score_thresh=OCR_CONFIG["rec_score_thresh"],
                text_det_unclip_ratio=OCR_CONFIG["det_unclip_ratio"],
                text_det_thresh=OCR_CONFIG["det_thresh"],
            )
            
            pb_utils.Logger.log_info(f"OCR prediction completed, result count: {len(ocr_result)}")
            
            # Extract text from OCR results
            detected_texts = []
            for res in ocr_result:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                
                if texts and scores:
                    combined_text = "".join(str(text or "") for text in texts)
                    combined_score = float(mean(float(score) for score in scores))
                    detected_texts.append(f"{combined_text} (confidence: {combined_score:.2f})")
                    pb_utils.Logger.log_info(f"Detected text: {combined_text}, confidence: {combined_score:.2f}")
            
            if detected_texts:
                result = " | ".join(detected_texts)
                pb_utils.Logger.log_info(f"OCR Result: {result}")
                return result
            else:
                pb_utils.Logger.log_warning("No text detected in image")
                return "No text detected"
                
        except Exception as exc:
            pb_utils.Logger.log_error(f"OCR processing failed: {exc}")
            return f"OCR Error: {str(exc)}"