import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from pathlib import Path
from typing import Tuple, List, Dict, Any


class TritonTestClient:
    def __init__(self, server_url: str = "127.0.0.1:9001"):
        self.server_url = server_url
        try:
            self.client = grpcclient.InferenceServerClient(url=self.server_url)
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {self.server_url} is not live")
            print(f"Connected to Triton via gRPC at {self.server_url}")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def preprocess_image(
            self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize with padding/normalization and capture letterbox metadata."""
        original_height, original_width = image.shape[:2]
        target_height, target_width = target_size

        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = cv2.resize(image, (new_width, new_height))

        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        # Metadata lets us undo the padding later when projecting detections back
        meta = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "input_size": target_size,
        }
        return blob, meta

    def process_inference_results(
            self,
            detections_raw: np.ndarray,
            img_shape: Tuple[int, int],
            preprocess_meta: Dict[str, Any],
            conf_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Filter raw detections, then undo letterbox padding to pixel coordinates."""
        img_h, img_w = img_shape
        input_h, input_w = preprocess_meta["input_size"]
        x_offset = preprocess_meta["x_offset"]
        y_offset = preprocess_meta["y_offset"]
        scale = max(preprocess_meta["scale"], 1e-6)

        if detections_raw.ndim == 3:
            detections_raw = detections_raw[0]
        if detections_raw.size == 0:
            return []

        # Filter by confidence
        if detections_raw.shape[1] < 5:
            raise ValueError("Detections must include at least 5 columns [cx,cy,w,h,conf]")
        valid_indices = np.where(detections_raw[:, 4] >= conf_threshold)[0]
        results = []

        for idx in valid_indices:
            cx, cy, w, h, conf = detections_raw[idx]

            # Convert normalized [cx, cy, w, h] to padded pixel coordinates
            x1 = (cx - w / 2) * input_w
            y1 = (cy - h / 2) * input_h
            x2 = (cx + w / 2) * input_w
            y2 = (cy + h / 2) * input_h

            # Remove padding and scale back to original resolution
            x1 = (x1 - x_offset) / scale
            y1 = (y1 - y_offset) / scale
            x2 = (x2 - x_offset) / scale
            y2 = (y2 - y_offset) / scale

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "bbox": [
                    int(max(0, min(img_w, x1))),
                    int(max(0, min(img_h, y1))),
                    int(max(0, min(img_w, x2))),
                    int(max(0, min(img_h, y2))),
                ],
                "confidence": float(conf)
            })

        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def test_inference(self, model_name: str, image_path: str, conf_threshold: float = 0.0):
        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image not found at {image_path}")
            return

        # 2. Preprocess
        preprocessed, preprocess_meta = self.preprocess_image(image)
        preprocessed = np.expand_dims(preprocessed, axis=0)

        # 3. Run Inference
        input_name = "images"
        output_name = "output0"
        inputs = [grpcclient.InferInput(input_name, preprocessed.shape, np_to_triton_dtype(preprocessed.dtype))]
        inputs[0].set_data_from_numpy(preprocessed)
        outputs = [grpcclient.InferRequestedOutput(output_name)]

        print(f"Inference on {model_name}...")
        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        detections_raw = response.as_numpy(output_name)
        if detections_raw is None:
            print("Warning: model returned no detections")
            return []

        # 4. Process Results
        results = self.process_inference_results(
            detections_raw,
            image.shape[:2],
            preprocess_meta,
            conf_threshold,
        )
        print(f"Found {len(results)} objects with confidence >= {conf_threshold}")

        # 5. Annotate and Save
        # Draw top 50 boxes to see what the model is finding
        for i, res in enumerate(results):
            if i >= 50: break
            x1, y1, x2, y2 = res['bbox']
            conf = res['confidence']

            # Use Green for high confidence, Red for low
            color = (0, 255, 0) if conf > 0.2 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        output_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_all_boxes.jpg"))
        cv2.imwrite(output_path, image)
        print(f"Processed image saved to: {output_path}")

        return results


def main():
    client = TritonTestClient(server_url="127.0.0.1:9001")
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    image_path = BASE_DIR / "frame_0000.jpg"
    # image_path ="/home/harsha/abhishek/MultiModal-Detection-master/triton-server/app/frame_0000.jpg"
    # image_path="abhishek/MultiModal-Detection-master/triton_server/app/frame_0000.jpg"
    # image_path="abhishek/MultiModal-Detection-master/triton-server/app/Cars.jpg"

    # Run test with plate detection model and very low threshold to see everything
    client.test_inference("plate_region_detection_rt_detr", image_path, conf_threshold=0.1)
    # client.test_inference("vehicle_detection_rt_detr", image_path, conf_threshold=0.1)


if __name__ == "__main__":
    main()
