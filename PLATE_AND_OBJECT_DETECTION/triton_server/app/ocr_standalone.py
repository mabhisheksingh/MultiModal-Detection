#!/usr/bin/env python3
"""
Standalone OCR script using PaddleOCR.
Similar to the paddle_ocr_gpu.disabled_not_working in mac Triton model but runs directly without Triton server.
"""

import numpy as np
import cv2
import os
import sys
from statistics import mean
from pathlib import Path

# Disable model source check to avoid download prompts
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# OCR Configuration
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


class OCRProcessor:
    """OCR processor using PaddleOCR."""
    
    def __init__(self, use_cpu=True):
        """Initialize OCR processor."""
        import paddle
        from paddleocr import PaddleOCR
        
        self._paddle = paddle
        
        # Force CPU device
        self._paddle.set_device("cpu")
        print(f"PaddleOCR using CPU device (forced)")
        
        # Initialize PaddleOCR reader
        self.reader = PaddleOCR(
            use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
            lang=OCR_CONFIG["lang"],
            ocr_version=OCR_CONFIG["ocr_version"],
        )
        
        print(f"PaddleOCR initialized with version: {OCR_CONFIG['ocr_version']}")
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
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
    
    def process_image(self, image_path: str) -> dict:
        """Process image using PaddleOCR and return detected text."""
        print(f"Processing image: {image_path}")
        
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return {"error": "Image not found"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Failed to load image: {image_path}")
                return {"error": "Failed to load image"}
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded with shape: {image.shape}, dtype: {image.dtype}")
            
            # Ensure image is in BGR format for PaddleOCR
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = image
            
            # Preprocess for better readability
            processed_image = self.preprocess_for_ocr(image)
            
            # Perform OCR prediction using configuration from OCR_CONFIG
            ocr_result = self.reader.predict(
                processed_image,
                text_det_limit_side_len=OCR_CONFIG["det_limit_side_len"],
                text_det_box_thresh=OCR_CONFIG["det_box_thresh"],
                text_rec_score_thresh=OCR_CONFIG["rec_score_thresh"],
                text_det_unclip_ratio=OCR_CONFIG["det_unclip_ratio"],
                text_det_thresh=OCR_CONFIG["det_thresh"],
            )
            
            print(f"OCR prediction completed, result count: {len(ocr_result)}")
            
            # Extract text from OCR results
            detected_texts = []
            for res in ocr_result:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                
                if texts and scores:
                    combined_text = "".join(str(text or "") for text in texts)
                    combined_score = float(mean(float(score) for score in scores))
                    detected_texts.append({
                        "text": combined_text,
                        "confidence": combined_score
                    })
                    print(f"Detected text: {combined_text}, confidence: {combined_score:.2f}")
            
            if detected_texts:
                result = {
                    "success": True,
                    "texts": detected_texts,
                    "combined_text": " | ".join([f"{t['text']} (confidence: {t['confidence']:.2f})" for t in detected_texts])
                }
                print(f"OCR Result: {result['combined_text']}")
                return result
            else:
                print("No text detected in image")
                return {"success": False, "error": "No text detected"}
                
        except Exception as exc:
            print(f"OCR processing failed: {exc}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(exc)}


def main():
    """Main function to run OCR on an image."""
    if len(sys.argv) < 2:
        print("Usage: python ocr_standalone.py <image_path>")
        print("Example: python ocr_standalone.py /path/to/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize OCR processor
    print("Initializing OCR processor...")
    ocr = OCRProcessor(use_cpu=True)
    
    # Process image
    result = ocr.process_image(image_path)
    
    # Print result
    print("\n" + "=" * 60)
    print("OCR RESULT")
    print("=" * 60)
    if result.get("success"):
        print(f"Detected {len(result['texts'])} text(s):")
        for i, text_info in enumerate(result['texts'], 1):
            print(f"  {i}. {text_info['text']} (confidence: {text_info['confidence']:.2f})")
        print(f"\nCombined: {result['combined_text']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("=" * 60)
    
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
