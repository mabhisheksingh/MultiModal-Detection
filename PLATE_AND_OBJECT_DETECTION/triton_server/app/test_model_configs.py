#!/usr/bin/env python3
"""
Script to test all Triton models by connecting to server and running inference.
Enable/disable models by commenting/uncommenting the test function calls below.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tritonclient.grpc as grpc_client


# Triton server configuration
TRITON_SERVER_URL = "localhost:8001"  # gRPC port

# Model repository path
MODEL_REPO_PATH = Path("/Users/abhishek/PycharmProjects/TagTrack-AI_v2/PLATE_AND_OBJECT_DETECTION/triton_server/model_repository")

# Test image path (modify as needed)
TEST_IMAGE_PATH = "/Users/abhishek/PycharmProjects/TagTrack-AI_v2/PLATE_AND_OBJECT_DETECTION/triton_server/app/Cars.jpg"


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
            return True
        except Exception as e:
            print(f"Failed to connect to Triton server: {e}")
            self.connected = False
            return False
    
    def check_server_live(self) -> bool:
        """Check if server is live"""
        if not self.connected:
            return False
        try:
            return self.client.is_server_live()
        except Exception as e:
            print(f"Server live check failed: {e}")
            return False
    
    def check_server_ready(self) -> bool:
        """Check if server is ready"""
        if not self.connected:
            return False
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"Server ready check failed: {e}")
            return False
    
    def get_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready"""
        if not self.connected:
            return False
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            print(f"Model {model_name} ready check failed: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get model metadata"""
        if not self.connected:
            return None
        try:
            metadata = self.client.get_model_metadata(model_name)
            return metadata
        except Exception as e:
            print(f"Failed to get metadata for {model_name}: {e}")
            return None
    
    def infer(self, model_name: str, inputs: List, outputs: List) -> Optional[Dict]:
        """Run inference on a model via gRPC"""
        if not self.connected:
            return None
        try:
            response = self.client.infer(model_name, inputs, outputs=outputs)
            return response
        except Exception as e:
            print(f"Inference failed for {model_name}: {e}")
            return None


class ImagePreprocessor:
    """Image preprocessing for different models"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from path"""
        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return None
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def preprocess_yolo(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess for YOLO models"""
        # Resize
        resized = cv2.resize(image, target_size)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)  # FP32 for YOLO
    
    @staticmethod
    def preprocess_general(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess for general object detection"""
        resized = cv2.resize(image, target_size)
        normalized = resized.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        return batched.astype(np.float32)
    
    @staticmethod
    def preprocess_ocr(image: np.ndarray) -> np.ndarray:
        """Preprocess for OCR models"""
        # Keep original size, just normalize
        normalized = image.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched.astype(np.float32)


def test_yolo_object_detection(client: TritonClient, image: np.ndarray) -> Dict:
    """Test YOLO object detection model"""
    result = {
        "model_name": "yolo_object_detection",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output_shape": None,
        "output": None
    }
    
    try:
        # Preprocess
        preprocessed = ImagePreprocessor.preprocess_yolo(image)
        
        # Create input
        input_tensor = grpc_client.InferInput("images", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output
        output = grpc_client.InferRequestedOutput("output0")
        
        # Run inference
        start_time = time.time()
        response = client.infer("yolo_object_detection", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("output0")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output_shape"] = str(output_data.shape)
            result["output"] = output_data
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_object_detection(client: TritonClient, image: np.ndarray) -> Dict:
    """Test general object detection model"""
    result = {
        "model_name": "object_detection",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output_shape": None,
        "output": None
    }
    
    try:
        # Preprocess
        preprocessed = ImagePreprocessor.preprocess_general(image)
        
        # Create input
        input_tensor = grpc_client.InferInput("images", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output
        output = grpc_client.InferRequestedOutput("output0")
        
        # Run inference
        start_time = time.time()
        response = client.infer("object_detection", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("output0")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output_shape"] = str(output_data.shape)
            result["output"] = output_data
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_paddle_ocr(client: TritonClient, image: np.ndarray) -> Dict:
    """Test Paddle OCR model"""
    result = {
        "model_name": "paddle_ocr_gpu.disabled_not_working in mac",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output": None
    }
    
    try:
        # Preprocess
        preprocessed = ImagePreprocessor.preprocess_ocr(image)
        
        # Create input (Paddle OCR uses string input for image path or numpy array)
        input_tensor = grpc_client.InferInput("INPUT1", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output
        output = grpc_client.InferRequestedOutput("OUTPUT0")
        
        # Run inference
        start_time = time.time()
        response = client.infer("paddle_ocr_gpu.disabled_not_working in mac", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("OUTPUT0")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output"] = str(output_data)
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_plate_region_detection(client: TritonClient, image: np.ndarray) -> Dict:
    """Test plate region detection RT-DETR model"""
    result = {
        "model_name": "plate_region_detection_rt_detr",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output_shape": None,
        "output": None
    }
    
    try:
        # Preprocess (similar to general detection)
        preprocessed = ImagePreprocessor.preprocess_general(image)
        
        # Create input
        input_tensor = grpc_client.InferInput("images", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output
        output = grpc_client.InferRequestedOutput("output0")
        
        # Run inference
        start_time = time.time()
        response = client.infer("plate_region_detection_rt_detr", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("output0")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output_shape"] = str(output_data.shape)
            result["output"] = output_data
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_retinaface(client: TritonClient, image: np.ndarray) -> Dict:
    """Test RetinaFace model"""
    result = {
        "model_name": "retinaface",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output_shape": None
    }
    
    try:
        # Preprocess (similar to general detection)
        preprocessed = ImagePreprocessor.preprocess_general(image)
        
        # Create input
        input_tensor = grpc_client.InferInput("RetinaFace::input_0", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output (request first output)
        output = grpc_client.InferRequestedOutput("1156")
        
        # Run inference
        start_time = time.time()
        response = client.infer("retinaface", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("1156")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output_shape"] = str(output_data.shape)
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_vehicle_detection(client: TritonClient, image: np.ndarray) -> Dict:
    """Test vehicle detection RT-DETR model"""
    result = {
        "model_name": "vehicle_detection_rt_detr",
        "status": "not_tested",
        "error": None,
        "inference_time": None,
        "output_shape": None,
        "output": None
    }
    
    try:
        # Preprocess (similar to general detection)
        preprocessed = ImagePreprocessor.preprocess_general(image)
        
        # Create input
        input_tensor = grpc_client.InferInput("images", preprocessed.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed)
        
        # Create output
        output = grpc_client.InferRequestedOutput("output0")
        
        # Run inference
        start_time = time.time()
        response = client.infer("vehicle_detection_rt_detr", [input_tensor], [output])
        inference_time = time.time() - start_time
        
        if response:
            output_data = response.as_numpy("output0")
            result["status"] = "success"
            result["inference_time"] = f"{inference_time:.3f}s"
            result["output_shape"] = str(output_data.shape)
            result["output"] = output_data
        else:
            result["status"] = "inference_failed"
            result["error"] = "No response from server"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def print_inference_result(result: Dict):
    """Print inference result for a model"""
    status_emoji = {
        "success": "✓",
        "error": "✗",
        "inference_failed": "✗",
        "not_tested": "-"
    }
    
    emoji = status_emoji.get(result["status"], "?")
    print(f"\n{emoji} {result['model_name']}")
    print(f"  Status: {result['status']}")
    
    if result["inference_time"]:
        print(f"  Inference Time: {result['inference_time']}")
    
    if result["output_shape"]:
        print(f"  Output Shape: {result['output_shape']}")
    
    if "output" in result and result["output"] is not None:
        print(f"  Output: {result['output']}")
        print(f"  Output sample: {result['output'].flatten()[:10]}...")
    
    if result["error"]:
        print(f"  Error: {result['error']}")


def main():
    """Main test function"""
    print("=" * 60)
    print("Triton Model Inference Test")
    print("=" * 60)
    print(f"Triton Server: {TRITON_SERVER_URL}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print()
    
    # Step 1: Connect to Triton server
    print("Step 1: Connecting to Triton server...")
    client = TritonClient(TRITON_SERVER_URL)
    if not client.connect():
        print(f"✗ Failed to connect to Triton server at {TRITON_SERVER_URL}")
        print("Make sure Triton server is running: docker-compose up -d")
        sys.exit(1)
    print("✓ Connected to Triton server")
    
    # Step 2: Check server status
    print("\nStep 2: Checking server status...")
    if not client.check_server_live():
        print("✗ Server is not live")
        sys.exit(1)
    print("✓ Server is live")
    
    if not client.check_server_ready():
        print("✗ Server is not ready")
        sys.exit(1)
    print("✓ Server is ready")
    
    # Step 3: Load test image
    print("\nStep 3: Loading test image...")
    image = ImagePreprocessor.load_image(TEST_IMAGE_PATH)
    if image is None:
        print(f"✗ Failed to load image from {TEST_IMAGE_PATH}")
        print("Please provide a valid image path in TEST_IMAGE_PATH variable")
        sys.exit(1)
    print(f"✓ Loaded image with shape: {image.shape}")
    
    # Step 4: Test models
    print("\nStep 4: Testing models on Triton server...")
    print("=" * 60)
    
    # ============================================
    # ENABLE/DISABLE MODELS HERE
    # Comment/uncomment the function calls below
    # ============================================
    
    results = []
    
    # Test all models (comment out individual ones to disable)
    results.append(test_yolo_object_detection(client, image))
    results.append(test_object_detection(client, image))
    # results.append(test_paddle_ocr(client, image))
    results.append(test_plate_region_detection(client, image))
    # results.append(test_retinaface(client, image))  # Input shape mismatch: expects [-1,608,640,3], got [1,3,640,640]
    results.append(test_vehicle_detection(client, image))
    
    # ============================================
    
    # Print results
    for result in results:
        print_inference_result(result)
    
    # Summary
    print("\n" + "=" * 60)
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] in ["error", "inference_failed"])
    print(f"SUMMARY: {success_count} successful, {error_count} failed out of {len(results)} models")
    print("=" * 60)
    
    if error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
