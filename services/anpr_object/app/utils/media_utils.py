from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np

from triton_server.app import settings


class FileSourceUtils:
    @staticmethod
    def validate_file_path(source_path: str, allowed_extensions: set) -> Path:
        resolved = Path(source_path).expanduser()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Source does not exist: {source_path}")
        if resolved.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {source_path}")
        return resolved


class MediaSourceUtils:
    @staticmethod
    def validate_remote_media_url(
        url: str, *, source_name: str, allowed_extensions: list[str]
    ) -> str:
        normalized_url = url.strip()
        parsed = urlparse(normalized_url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"{source_name} must use http or https: {url}")
        suffix = Path(parsed.path).suffix.lower()
        if suffix and suffix not in {ext.lower() for ext in allowed_extensions}:
            raise ValueError(
                f"Unsupported remote {source_name.split('_')[0]} extension: {url}"
            )
        return normalized_url

    @staticmethod
    def resolve_camera_id(source: str, camera_id: Optional[str] = None) -> str:
        normalized_camera_id = (camera_id or "").strip()
        if normalized_camera_id:
            return normalized_camera_id
        normalized_source = (source or "").strip()
        if not normalized_source:
            return ""
        parsed = urlparse(normalized_source)
        if parsed.scheme.lower() in {"http", "https"}:
            source_name = Path(parsed.path).stem.strip()
            if source_name:
                return source_name
            return (parsed.netloc or "").replace(":", "_")
        return Path(normalized_source).expanduser().stem


class VideoSourceUtils:
    @staticmethod
    def validate_video_source(source: str) -> None:
        parsed = urlparse(source)
        if parsed.scheme.lower() in {"http", "https"}:
            suffix = Path(parsed.path).suffix.lower()
            if suffix and suffix not in {
                ext.lower() for ext in settings.video_extensions_list
            }:
                raise ValueError(
                    f"Unsupported remote video extension for source: {source}"
                )
            return
        FileSourceUtils.validate_file_path(
            source, {ext.lower() for ext in settings.video_extensions_list}
        )

    @staticmethod
    def build_output_paths(
        source: str, output_dir: str, request_id: Optional[str]
    ) -> Tuple[Path, str, str]:
        base_output_dir = (
            Path(output_dir).resolve() if output_dir else settings.data_output_dir
        )
        base_output_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(source)
        source_name = Path(parsed.path).name if parsed.path else "live_stream.mp4"
        if not source_name:
            source_name = "live_stream.mp4"
        if Path(source_name).suffix == "":
            source_name = f"{source_name}.mp4"
        folder_name = request_id or Path(source_name).stem
        request_folder = base_output_dir / folder_name
        request_folder.mkdir(parents=True, exist_ok=True)
        is_remote_source = urlparse(source).scheme.lower() in {"http", "https"}
        output_name = (
            source_name
            if not is_remote_source
            else f"annotated_{Path(source_name).stem}.mp4"
        )
        output_path = str(request_folder / output_name)
        return request_folder, output_path, source_name


class ImageSourceUtils:
    @staticmethod
    def validate_image_source_path(source_path: str) -> Path:
        return FileSourceUtils.validate_file_path(
            source_path, {ext.lower() for ext in settings.image_extensions_list}
        )

    @staticmethod
    def validate_image_source_url(image_url: str) -> None:
        parsed = urlparse(image_url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"Unsupported image URL scheme: {image_url}")
        suffix = Path(parsed.path).suffix.lower()
        if suffix and suffix not in {
            ext.lower() for ext in settings.image_extensions_list
        }:
            raise ValueError(f"Unsupported remote image extension: {image_url}")

    @staticmethod
    def load_image_from_url(image_url: str) -> np.ndarray:
        ImageSourceUtils.validate_image_source_url(image_url)
        with urlopen(image_url) as response:
            image_bytes = response.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Downloaded content is not a valid image: {image_url}")
        return image


class ImageAnalysisUtils:
    _VALID_PLATE_COLORS = GLOBAL_PLATE_COLORS = {
        "white",
        "yellow",
        "green",
        "black",
        "blue",
        "red",
        "light_blue",
        "maroon",
        "grey",
        "silver",
        "orange",
        "brown",
        "purple",
        "pink",
    }

    @staticmethod
    def calculate_blur(image: np.ndarray) -> float:
        return float(
            cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )

    @staticmethod
    def hue_to_color_name(hue: int) -> str:
        if hue <= 10 or hue >= 170:
            return "red"
        if hue <= 22:
            return "orange"
        if hue <= 38:
            return "yellow"
        if hue <= 85:
            return "green"
        if hue <= 100:
            return "cyan"
        if hue <= 130:
            return "blue"
        if hue <= 160:
            return "purple"
        return "pink"

    @staticmethod
    def extract_dominant_color(image: np.ndarray, min_area: int = 2500) -> str:
        if image is None or image.size == 0:
            return ""
        h, w = image.shape[:2]
        if (h * w) < min_area:
            return ""
        y0 = int(h * 0.22) if h > 20 else 0
        y1 = int(h * 0.88) if h > 20 else h
        x0 = int(w * 0.12) if w > 20 else 0
        x1 = int(w * 0.88) if w > 20 else w
        roi = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        if total_pixels == 0:
            return ""
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        valid_mask = val > 35
        valid_pixels = int(np.count_nonzero(valid_mask))
        if valid_pixels == 0:
            return ""
        neutral_mask = valid_mask & (sat < 40)
        neutral_pixels = int(np.count_nonzero(neutral_mask))
        if neutral_pixels >= int(valid_pixels * 0.35):
            neutral_v = val[neutral_mask]
            bright_ratio = float(np.count_nonzero(neutral_v >= 185)) / max(
                neutral_pixels, 1
            )
            dark_ratio = float(np.count_nonzero(neutral_v <= 70)) / max(
                neutral_pixels, 1
            )
            v_median = float(np.median(neutral_v))
            if bright_ratio >= 0.50 and v_median >= 185:
                return "white"
            if dark_ratio >= 0.40 and v_median <= 95:
                return "black"
            if v_median >= 150:
                return "silver"
            return "grey"
        colorful_mask = valid_mask & (sat >= 40)
        hue_vals = hsv[:, :, 0][colorful_mask]
        if len(hue_vals) == 0:
            v_median = float(np.median(val[valid_mask]))
            if v_median >= 175:
                return "white"
            if v_median <= 70:
                return "black"
            return "grey"
        hist = cv2.calcHist([hue_vals], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))
        dominant_count = int(hist[dominant_hue][0])
        if dominant_count < int(valid_pixels * 0.05):
            return ""
        return ImageAnalysisUtils.hue_to_color_name(dominant_hue)

    @classmethod
    def extract_plate_color(cls, image: np.ndarray) -> str:
        """Extract plate background color from plate bbox crop (same crop used for OCR)."""
        color = cls.extract_dominant_color(image, min_area=500)
        if color in cls._VALID_PLATE_COLORS:
            return color
        if color in {"grey", "silver"}:
            return "white"
        return ""
