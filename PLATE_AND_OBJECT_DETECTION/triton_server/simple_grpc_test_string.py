#!/usr/bin/env python3
"""
Minimal Triton gRPC string echo test.
Matches the official pattern from:
  https://github.com/triton-inference-server/server/blob/main/qa/python_models/string/
"""

import numpy as np
import tritonclient.grpc as grpc_client


def simple_grpc_test():
    try:
        print("Starting simple_grpc_test...")

        client = grpc_client.InferenceServerClient(url="127.0.0.1:9001", verbose=False)

        if not client.is_server_live():
            print("Server not live")
            return
        if not client.is_model_ready("paddle_ocr_gpu.disabled_not_working in mac"):
            print("Model not ready")
            return

        print("Server and model ready")

        # --- Build input (must be np.object_ for BYTES / TYPE_STRING) ---
        input_data = np.array(["Hello World"], dtype=object)
        print(f"Input: dtype={input_data.dtype}, shape={input_data.shape}, value={input_data}")

        inp = grpc_client.InferInput("INPUT0", list(input_data.shape), "BYTES")
        inp.set_data_from_numpy(input_data)

        # Debug: show what serialize_byte_tensor produced
        raw = inp._get_content()
        print(f"Serialized raw_input_contents length: {len(raw)}, hex: {raw.hex()}")

        out = grpc_client.InferRequestedOutput("OUTPUT0")

        # --- Infer ---
        print("Sending request...")
        result = client.infer(
            model_name="paddle_ocr_gpu.disabled_not_working in mac",
            inputs=[inp],
            outputs=[out],
            model_version="1",
        )

        # --- Decode output ---
        output_data = result.as_numpy("OUTPUT0")
        print(f"Output: dtype={output_data.dtype}, shape={output_data.shape}")

        if output_data is not None and output_data.size > 0:
            val = output_data[0]
            decoded = val.decode("utf-8") if isinstance(val, bytes) else str(val)
            print(f"Result: {decoded}")
        else:
            print("Empty output")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    simple_grpc_test()
