"""Shared Triton gRPC client wrapper."""

from typing import Any

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class TritonClient:
    """Base Triton Inference Server client (gRPC)."""

    def __init__(self, server_url: str = "127.0.0.1:9001"):
        self.server_url = server_url
        self.client = grpcclient.InferenceServerClient(url=server_url)
        print(f"[TritonClient] Connected to Triton at {server_url}")

    def is_server_live(self) -> bool:
        return self.client.is_server_live()

    def is_server_ready(self) -> bool:
        return self.client.is_server_ready()

    def is_model_ready(self, model_name: str) -> bool:
        return self.client.is_model_ready(model_name)

    def infer(self, model_name: str, inputs: list, outputs: list, request_id: str = None):
        """Run inference on a model."""
        return self.client.infer(model_name, inputs, outputs=outputs, request_id=request_id)

    @staticmethod
    def letterbox_preprocess(
        image: np.ndarray, target_size: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resize with padding (letterbox) and normalize. Returns (blob, meta)."""
        h, w = image.shape[:2]
        th, tw = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (nw, nh))
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        x_off = (tw - nw) // 2
        y_off = (th - nh) // 2
        padded[y_off : y_off + nh, x_off : x_off + nw] = resized

        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        meta = {"scale": scale, "x_off": x_off, "y_off": y_off, "target_size": target_size}
        return blob, meta

    @staticmethod
    def create_input(tensor: np.ndarray, name: str) -> grpcclient.InferInput:
        """Create a Triton InferInput from a numpy array."""
        inp = grpcclient.InferInput(name, tensor.shape, np_to_triton_dtype(tensor.dtype))
        inp.set_data_from_numpy(tensor)
        return inp

    @staticmethod
    def create_output(name: str) -> grpcclient.InferRequestedOutput:
        """Create a Triton InferRequestedOutput."""
        return grpcclient.InferRequestedOutput(name)
