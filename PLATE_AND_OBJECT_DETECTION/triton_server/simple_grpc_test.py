#!/usr/bin/env python3

import numpy as np
import cv2
import time
import tritonclient.grpc as grpc_client

TEST_IMAGE_PATH = "/home/abhishek/Desktop/Work/Infinium/vlm-video-captioning/ANPR/triton_server/app/Cars.jpg"

def simple_grpc_test():
    try:
        # Load image
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is None:
            print("Cannot read image")
            return

        print(f"Image shape: {img.shape}")

        # Create client
        client = grpc_client.InferenceServerClient(url="localhost:9001", verbose=False)

        # Check server
        if not client.is_server_live():
            print("Server not live")
            return

        if not client.is_model_ready("paddle_ocr_gpu.disabled_not_working in mac"):
            print("Model not ready")
            return

        print("Server and model ready")

        # Convert BGR to RGB and normalize to [0,1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        print(f"Input tensor shape: {img_float.shape}, dtype: {img_float.dtype}")

        # Create input
        inputs = [grpc_client.InferInput("INPUT1", img_float.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_float)

        # Create output
        outputs = [grpc_client.InferRequestedOutput("OUTPUT0")]

        # Send request with timing
        request_id = f"test_request_{np.random.randint(1000, 9999)}"
        start_time = time.time()
        result = client.infer(model_name="paddle_ocr_gpu.disabled_not_working in mac", inputs=inputs, outputs=outputs, model_version="1", request_id=request_id)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Inference time: {inference_time:.2f} ms")

        # Debug print raw response from server
        response = result.get_response(as_json=True)
        print(f"Response from server: {response}")

        # Print request ID
        print(f"Request ID: {request_id}")

        # Get and print decoded response
        data = result.as_numpy("OUTPUT0")
        if data is not None and len(data) > 0:
            val = data[0]
            if isinstance(val, bytes):
                decoded = val.decode('utf-8')
            else:
                decoded = str(val)
            print(f"Response: {decoded}")
        else:
            print("No output data received")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_grpc_test()
