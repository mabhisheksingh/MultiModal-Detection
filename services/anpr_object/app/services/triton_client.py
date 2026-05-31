import asyncio
import numpy as np
import cv2
import structlog
import traceback
from typing import Dict, Tuple, Any, Optional

from tritonclient.utils import np_to_triton_dtype
from ultralytics.utils.triton import TritonRemoteModel


class TritonClient:
    def __init__(self, server_url: str, model_name: str, scheme: str = "grpc") -> None:
        self.logger = structlog.get_logger(__name__)
        self.model_name = model_name
        self.server_url = server_url.replace("http://", "").replace("https://", "")
        self._scheme = scheme
        self._model: TritonRemoteModel | None = None

    def _get_model(self) -> TritonRemoteModel:
        if self._model is None:
            self._model = TritonRemoteModel(
                url=self.server_url,
                endpoint=self.model_name,
                scheme=self._scheme,
            )
            self.logger.info(
                "triton_connected",
                server=self.server_url,
                model=self.model_name,
                protocol=self._scheme,
                inputs=self._model.input_names,
                outputs=self._model.output_names,
            )
        return self._model

    def preprocess_image(
        self,
        image: np.ndarray,
        frame_idx: int,
        target_size: Tuple[int, int] = (640, 640),
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        original_height, original_width = image.shape[:2]
        target_height, target_width = target_size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = cv2.resize(image, (new_width, new_height))
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized
        )
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        meta = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "input_size": target_size,
            "frame_idx": frame_idx,
        }
        return blob, meta

    def _infer_request(
        self,
        model: TritonRemoteModel,
        input_batch: np.ndarray,
        request_id: Optional[str],
    ) -> list[np.ndarray]:
        infer_inputs = []
        input_format = input_batch.dtype
        infer_input = model.InferInput(
            model.input_names[0], [*input_batch.shape], np_to_triton_dtype(input_format)
        )
        infer_input.set_data_from_numpy(input_batch)
        infer_inputs.append(infer_input)
        infer_outputs = [
            model.InferRequestedOutput(output_name)
            for output_name in model.output_names
        ]
        outputs = model.triton_client.infer(
            model_name=model.endpoint,
            inputs=infer_inputs,
            outputs=infer_outputs,
            request_id=request_id,
        )
        return [
            outputs.as_numpy(output_name).astype(input_format)
            for output_name in model.output_names
        ]

    async def infer(
        self, image: np.ndarray, frame_idx: int, request_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            model = self._get_model()
            preprocessed, preprocess_meta = self.preprocess_image(image, frame_idx)
            input_batch = np.expand_dims(preprocessed, axis=0)
            outputs = await asyncio.to_thread(
                self._infer_request, model, input_batch, request_id
            )

            return outputs[0], preprocess_meta
        except Exception as e:
            self.logger.error(
                "inference_error", error=str(e), traceback=traceback.format_exc()
            )
            raise

    def close(self):
        self._model = None
