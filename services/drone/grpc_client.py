#!/usr/bin/env python3
"""
gRPC Client for Drone Object Detection on Triton Server.

This client sends images/videos to Triton server for object detection,
processes the results, and saves:
- Processed images with bounding boxes (output/)
- JSON detection results (output/)
- Original input images (input/)
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tritonclient.grpc as grpc_client


# Configuration
TRITON_SERVER_URL = "localhost:8001"  # gRPC port
MODEL_NAME = "object_detection"  # Change this to your drone detection model
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
CONFIDENCE_THRESHOLD = 0.5

# Paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = BASE_DIR / "dataset"
TEST_DATA_DIR = BASE_DIR / "test_data"


class TritonClient:
    """Triton Inference Server client (gRPC)"""

    def __init__(self, url: str = TRITON_SERVER_URL):
        self.url = url
        self.client = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to Triton server via gRPC"""
        try:
            self.client = grpc_client.InferenceServerClient(url=self.url, verbose=False)
            self.connected = True
            print(f"✓ Connected to Triton server at {self.url}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Triton server: {e}")
            self.connected = False
            return False

    def check_model_ready(self, model_name: str = MODEL_NAME) -> bool:
        """Check if model is ready"""
        if not self.connected:
            return False
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            print(f"✗ Model readiness check failed: {e}")
            return False

    def infer(self, model_name: str, input_tensor: np.ndarray) -> Optional[np.ndarray]:
        """Run inference on a model"""
        if not self.connected:
            return None
        try:
            # Create input
            input_tensor_obj = grpc_client.InferInput(
                INPUT_NAME, input_tensor.shape, "FP32"
            )
            input_tensor_obj.set_data_from_numpy(input_tensor)

            # Create output
            output = grpc_client.InferRequestedOutput(OUTPUT_NAME)

            # Run inference
            response = self.client.infer(model_name, [input_tensor_obj], outputs=[output])
            output_data = response.as_numpy(OUTPUT_NAME)
            return output_data
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return None


class ImageProcessor:
    """Image preprocessing for drone object detection"""

    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize
        resized = cv2.resize(image, target_size)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict],
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """Draw bounding boxes on image"""
        result = image.copy()

        for det in detections:
            if det["confidence"] < confidence_threshold:
                continue

            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            label = det.get("label", "object")
            conf = det["confidence"]

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label_text = f"{label}: {conf:.2f}"
            cv2.putText(
                result, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return result

    @staticmethod
    def save_image(image: np.ndarray, output_path: Path) -> None:
        """Save image to path"""
        cv2.imwrite(str(output_path), image)


class DroneDetectionClient:
    """Main client for drone object detection"""

    def __init__(self):
        self.triton_client = TritonClient()
        self.image_processor = ImageProcessor()

    def process_single_image(
        self,
        image_path: Path,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> Dict:
        """Process a single image"""
        print(f"\nProcessing: {image_path.name}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"✗ Failed to load image: {image_path}")
            return {"error": "Failed to load image"}

        # Save original to input folder
        input_path = INPUT_DIR / image_path.name
        ImageProcessor.save_image(image, input_path)

        # Preprocess
        preprocessed = self.image_processor.preprocess_image(image)

        # Infer
        start_time = time.time()
        output = self.triton_client.infer(model_name, preprocessed)
        inference_time = time.time() - start_time

        if output is None:
            return {"error": "Inference failed"}

        # Parse detections (this depends on your model output format)
        detections = self._parse_detections(output, image.shape)

        # Draw detections
        processed_image = self.image_processor.draw_detections(image, detections, confidence_threshold)

        # Save processed image
        output_path = OUTPUT_DIR / f"processed_{image_path.name}"
        ImageProcessor.save_image(processed_image, output_path)

        # Save JSON results
        json_path = OUTPUT_DIR / f"{image_path.stem}_detections.json"
        result = {
            "image_name": image_path.name,
            "image_shape": list(image.shape),
            "inference_time": inference_time,
            "detections": detections,
            "num_detections": len(detections)
        }

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✓ Processed in {inference_time:.3f}s")
        print(f"✓ Found {len(detections)} detections")
        print(f"✓ Saved to {output_path}")
        print(f"✓ JSON saved to {json_path}")

        return result

    def process_directory(
        self,
        input_dir: Path,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> List[Dict]:
        """Process all images in a directory"""
        if not input_dir.exists():
            print(f"✗ Input directory not found: {input_dir}")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"✗ No images found in {input_dir}")
            return []

        print(f"Found {len(image_files)} images to process")

        results = []
        for image_file in sorted(image_files):
            result = self.process_single_image(image_file, model_name, confidence_threshold)
            results.append(result)

        return results

    def process_video(
        self,
        video_path: Path,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        frame_skip: int = 1
    ) -> Dict:
        """Process video frame by frame"""
        print(f"\nProcessing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return {"error": "Failed to open video"}

        frame_count = 0
        all_detections = []

        output_video_path = OUTPUT_DIR / f"processed_{video_path.name}"
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Preprocess
            preprocessed = self.image_processor.preprocess_image(frame)

            # Infer
            output = self.triton_client.infer(model_name, preprocessed)

            if output is not None:
                detections = self._parse_detections(output, frame.shape)
                all_detections.extend(detections)

                # Draw detections
                processed_frame = self.image_processor.draw_detections(frame, detections, confidence_threshold)
            else:
                processed_frame = frame

            out.write(processed_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()

        # Save JSON results
        json_path = OUTPUT_DIR / f"{video_path.stem}_detections.json"
        result = {
            "video_name": video_path.name,
            "total_frames": frame_count,
            "detections": all_detections,
            "num_detections": len(all_detections)
        }

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✓ Processed {frame_count} frames")
        print(f"✓ Total detections: {len(all_detections)}")
        print(f"✓ Video saved to {output_video_path}")
        print(f"✓ JSON saved to {json_path}")

        return result

    def _parse_detections(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> List[Dict]:
        """Parse model output to detection format.
        
        This is a placeholder - adapt this to your specific model output format.
        """
        detections = []

        # Example parsing for YOLO-style output
        # Adjust based on your actual model output format
        try:
            # Assuming output shape is (1, num_detections, num_classes + 4)
            detections_raw = output[0]  # Remove batch dimension

            for i in range(detections_raw.shape[0]):
                # Extract bbox and confidence
                # This is a simplified example - adapt to your model
                bbox = detections_raw[i, :4]
                confidence = float(detections_raw[i, 4])

                if confidence > CONFIDENCE_THRESHOLD:
                    detections.append({
                        "bbox": bbox.tolist(),
                        "confidence": confidence,
                        "label": "object"
                    })
        except Exception as e:
            print(f"Warning: Failed to parse detections: {e}")

        return detections


def main():
    """Main entry point"""
    print("=" * 60)
    print("Drone Object Detection - Triton gRPC Client")
    print("=" * 60)

    # Create directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = DroneDetectionClient()

    # Connect to Triton server
    if not client.triton_client.connect():
        print("Please ensure Triton server is running:")
        print("  cd ../PLATE_AND_OBJECT_DETECTION/triton_server")
        print("  docker-compose up -d")
        sys.exit(1)

    # Check model readiness
    if not client.triton_client.check_model_ready(MODEL_NAME):
        print(f"✗ Model '{MODEL_NAME}' is not ready")
        print("Available models:")
        try:
            model_repository = client.triton_client.get_model_repository()
            for model in model_repository:
                print(f"  - {model.name}")
        except:
            pass
        sys.exit(1)

    print(f"✓ Model '{MODEL_NAME}' is ready")

    # Process based on command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python grpc_client.py <input_path>")
        print("  python grpc_client.py --dir <directory>")
        print("  python grpc_client.py --video <video_path>")
        print("\nExamples:")
        print("  python grpc_client.py ../dataset/drone_image.jpg")
        print("  python grpc_client.py --dir ../dataset/")
        print("  python grpc_client.py --video ../dataset/drone_footage.mp4")
        sys.exit(0)

    input_arg = sys.argv[1]

    if input_arg == "--dir" and len(sys.argv) >= 3:
        directory = Path(sys.argv[2])
        client.process_directory(directory)
    elif input_arg == "--video" and len(sys.argv) >= 3:
        video_path = Path(sys.argv[2])
        client.process_video(video_path)
    else:
        image_path = Path(input_arg)
        if image_path.is_dir():
            client.process_directory(image_path)
        elif image_path.is_file():
            client.process_single_image(image_path)
        else:
            print(f"✗ Invalid input: {input_arg}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
