import os
import time
from statistics import mean

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"
# Force CPU mode for environments without GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PADDLE_DISABLE_GPU"] = "1"

MIN_OCR_SIZE = 128

# Point PaddleX to local official_models directory so it finds models locally
# PaddleX looks for models in {PADDLEX_HOME}/official_models/
MODEL_DIR = "/models/paddle_ocr_gpu_batched/official_models"
os.environ["PADDLEX_HOME"] = MODEL_DIR

# ----------------------------------------------------------
# PADDLEOCR CONFIGURATION - Local PP-OCRv6 Server Tiny Models
# ----------------------------------------------------------
OCR_CONFIG = {
    # Local PP-OCRv6 server tiny models (in official_models directory)
    "text_recognition_model_name": "PP-OCRv6_tiny_rec",
    "text_detection_model_name": "PP-OCRv6_tiny_det",
    # Text Detection Parameters
    # Since your input is already a cropped plate, the detector
    # doesn't need a massive 960x960 canvas. 320 is perfect.
    "text_det_limit_side_len": 320,  # Max dimension for detection input (larger = slower but more accurate)
    "text_det_box_thresh": 0.50,  # Minimum confidence for detected boxes (higher = faster, fewer false positives)
    "text_det_thresh": 0.30,  # Detection threshold for text regions (higher = faster, may miss faint text)
    "text_det_unclip_ratio": 1.60,  # Ratio to expand detected boxes (lower = tighter boxes, faster post-processing)
    "text_rec_score_thresh": 0.55,  # Minimum confidence for recognized text (higher = fewer false recognitions)
    # --- Preprocessing Module Disabling ---
    # These modules are disabled for ANPR to improve speed:
    # - License plates are already properly oriented (no rotation needed)
    # - License plates are flat surfaces (no perspective unwarping needed)
    # - License plate text is typically horizontal (no orientation classification needed)
    "use_doc_orientation_classify": False,  # Disable document orientation model
    "use_doc_unwarping": False,  # Disable document unwarping model
    "use_textline_orientation": False,  # Disable textline orientation model
}


class TritonPythonModel:

    def initialize(self, args):
        pb_utils.Logger.log_info("[OCR] Initializing PaddleOCR with local PP-OCRv6 models")

        from paddleocr import PaddleOCR

        self._device = self._select_device()
        self.log_border = "-" * 100
        self.reader = None  # Initialize as None, set later if successful

        # Initialize PaddleOCR with local PP-OCRv6 server tiny models
        pb_utils.Logger.log_info(f"[OCR] Initializing PaddleOCR with PADDLEX_HOME={MODEL_DIR}")

        try:
            self.reader = PaddleOCR(
                text_detection_model_name=OCR_CONFIG["text_detection_model_name"],
                text_recognition_model_name=OCR_CONFIG["text_recognition_model_name"],
                lang="en",
                use_doc_orientation_classify=OCR_CONFIG["use_doc_orientation_classify"],
                use_doc_unwarping=OCR_CONFIG["use_doc_unwarping"],
                use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
                det_limit_side_len=OCR_CONFIG["text_det_limit_side_len"],
                det_db_box_thresh=OCR_CONFIG["text_det_box_thresh"],
                det_db_thresh=OCR_CONFIG["text_det_thresh"],
                det_db_unclip_ratio=OCR_CONFIG["text_det_unclip_ratio"],
                rec_batch_num=6,
            )
            pb_utils.Logger.log_info(f"[OCR] Initialized on device: {self._device}")
        except Exception as e:
            pb_utils.Logger.log_error(
                f"[OCR] Failed to initialize PaddleOCR: {str(e)}. "
                f"Model will return empty responses. "
                f"Ensure model files are compatible with the container architecture."
            )

    # ----------------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------------
    def execute(self, requests):
        # Check if initialization failed
        if self.reader is None:
            pb_utils.Logger.log_error("[OCR] Model not initialized, returning empty responses")
            return self._empty_responses(requests)

        start_time = time.perf_counter()

        # Log request start with requested format
        for req_idx, request in enumerate(requests):
            request_id = request.request_id()
            pb_utils.Logger.log_info("-" * 100)
            pb_utils.Logger.log_info(f"OCR start request id {request_id if request_id else '-'}")
            pb_utils.Logger.log_info("-" * 100)

        all_images = []
        request_map = []
        image_shapes = []

        # -------------------------------
        # 1. COLLECT INPUTS
        # -------------------------------
        for req_idx, request in enumerate(requests):
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT1")
                if in_tensor is None:
                    continue

                arr = in_tensor.as_numpy()
                if arr is None or arr.size == 0:
                    continue

                # batch handling
                if arr.ndim == 4:
                    images = list(arr)
                elif arr.ndim == 3:
                    images = [arr]
                else:
                    raise ValueError(f"Invalid shape: {arr.shape}")

                for i, img in enumerate(images):
                    if img is None or img.size == 0:
                        continue
                    image_shapes.append((img.shape, img.dtype))
                    all_images.append(img)
                    request_map.append(req_idx)

            except Exception as e:
                pb_utils.Logger.log_error(f"[OCR][REQ-{req_idx}] Parsing failed: {str(e)}")

        total_images = len(all_images)
        pb_utils.Logger.log_info(f"[OCR] Aggregated batch → total_images={total_images}")

        # -------------------------------
        # 2. EMPTY CASE
        # -------------------------------
        if total_images == 0:
            return self._empty_responses(requests)

        # -------------------------------
        # 3. PREPROCESS
        # -------------------------------
        t0 = time.perf_counter()
        processed_images = []
        for idx, img in enumerate(all_images):
            # Upscale if too small (Optional: helps mobile models read tiny crops)
            img = self._resize_if_small(img, idx)
            processed_images.append(img)

        preprocess_time = (time.perf_counter() - t0) * 1000

        # -------------------------------
        # 4. OCR (SINGLE GPU CALL)
        # -------------------------------
        pb_utils.Logger.log_info(f"[OCR] Running inference on {len(processed_images)} images")
        t1 = time.perf_counter()

        try:
            # Use predict() instead of deprecated ocr()
            # Since we are passing a list of images, it returns a list of results
            ocr_results = self.reader.predict(processed_images)
        except Exception as e:
            pb_utils.Logger.log_error(f"[OCR] Batch inference failed: {str(e)}")
            ocr_results = []

        ocr_time = (time.perf_counter() - t1) * 1000

        # -------------------------------
        # 5. POSTPROCESS
        # -------------------------------
        outputs = []
        success = 0

        # Handle empty results case
        if not ocr_results:
            ocr_results = [{}] * len(processed_images)

        for idx, res in enumerate(ocr_results):
            try:
                # PaddleX TextRecognition pipeline returns 'rec_text' and 'rec_score' directly
                # (or inside a 'res' key depending on exact sub-pipeline output formatting)
                if isinstance(res, dict):
                    texts = res.get("rec_text", res.get("rec_texts", []))
                    scores = res.get("rec_score", res.get("rec_scores", []))
                else:
                    # If it's a PaddleX result object
                    texts = getattr(res, "rec_text", getattr(res, "rec_texts", []))
                    scores = getattr(res, "rec_score", getattr(res, "rec_scores", []))

                # Normalize to lists for iteration
                if not isinstance(texts, list):
                    texts = [texts] if texts else []
                if not isinstance(scores, list):
                    scores = [scores] if scores else []

                shape, dtype = image_shapes[idx] if idx < len(image_shapes) else (None, None)
                pb_utils.Logger.log_verbose(f"[OCR][IMG-{idx}] shape={shape} | texts={texts} scores={scores}")

                success += 1

                if texts and scores and texts[0]:
                    combined_text = "".join(str(t or "") for t in texts)
                    combined_score = float(mean(float(s) for s in scores if s is not None))
                    result_str = f"{combined_text} (confidence: {combined_score:.2f})"
                    outputs.append(result_str)
                else:
                    outputs.append("")

            except Exception as e:
                pb_utils.Logger.log_error(f"[OCR][IMG-{idx}] Postprocess failed: {str(e)}")
                outputs.append("Postprocess error")

        # -------------------------------
        # 6. MAP BACK & BUILD RESPONSES
        # -------------------------------
        per_request_outputs = [[] for _ in requests]
        for out, req_idx in zip(outputs, request_map, strict=False):
            per_request_outputs[req_idx].append(out)

        responses = []
        for req_idx, out_list in enumerate(per_request_outputs):
            if not out_list:
                pb_utils.Logger.log_warn(f"[OCR][REQ-{req_idx}] No output generated")
                out_list = ["No input"]
            out_arr = np.array(out_list, dtype=object).reshape(-1, 1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT0", out_arr)]))
        # -------------------------------
        # 8. FINAL METRICS
        # -------------------------------
        total_time = (time.perf_counter() - start_time) * 1000
        pb_utils.Logger.log_verbose(
            f"[OCR] DONE | total={total_time:.2f}ms | preprocess={preprocess_time:.2f}ms | "
            f"ocr={ocr_time:.2f}ms | images={total_images} | success={success} | "
            f"per_image={total_time / max(1, total_images):.2f}ms"
        )

        # Log request end with requested format
        for req_idx, request in enumerate(requests):
            request_id = request.request_id()
            pb_utils.Logger.log_info("-" * 100)
            pb_utils.Logger.log_info(
                f"OCR end request id {request_id if request_id else '-'} and time take {total_time:.2f}ms"
            )
            pb_utils.Logger.log_info("-" * 100)

        return responses

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    def _resize_if_small(self, image, idx):
        h, w = image.shape[:2]
        if h < MIN_OCR_SIZE or w < MIN_OCR_SIZE:
            scale = max(MIN_OCR_SIZE / h, MIN_OCR_SIZE / w)
            new_size = (int(w * scale), int(h * scale))
            pb_utils.Logger.log_verbose(f"[OCR][IMG-{idx}] Upscaling {image.shape} → {new_size}")
            image = cv2.resize(image, new_size)
        return image

    def _empty_responses(self, requests):
        responses = []
        for _ in requests:
            out = np.array(["No input"], dtype=object).reshape(-1, 1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT0", out)]))
        return responses

    def _select_device(self):
        # Force CPU mode for Triton server compatibility
        pb_utils.Logger.log_info("[OCR] Using CPU mode for ONNX engine")
        return "cpu"
