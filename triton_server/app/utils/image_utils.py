"""Image loading utilities - resolves image names to assets folder paths."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

# Base assets directory relative to this file: utils/../assets
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_image(image_source: Union[str, Path]) -> np.ndarray:
    """
    Load an image from the assets folder or an absolute path.

    - If only a filename is given (e.g., "frame_0000.jpg"), it is resolved
      from the assets/ folder.
    - If an absolute or relative path is given, it is used directly.

    Args:
        image_source: Image filename (resolved from assets/) or full path.

    Returns:
        Loaded image as BGR numpy array.

    Raises:
        FileNotFoundError: If the image cannot be found.
        RuntimeError: If the image fails to load.
    """
    path = Path(image_source)
    if not path.is_absolute() and len(path.parts) == 1:
        # Just a filename — resolve from assets/
        path = _ASSETS_DIR / path

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def load_image_rgb(image_source: Union[str, Path]) -> np.ndarray:
    """Load image and convert to RGB."""
    img = load_image(image_source)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, filename: str, subdir: Optional[str] = None) -> Path:
    """
    Save image to the assets folder.

    Args:
        image: Image array (BGR).
        filename: Output filename.
        subdir: Optional subdirectory under assets/.

    Returns:
        Path to the saved file.
    """
    out_dir = _ASSETS_DIR
    if subdir:
        out_dir = out_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), image)
    return out_path


def list_assets() -> list:
    """List all files in the assets directory."""
    if not _ASSETS_DIR.exists():
        return []
    return [f.name for f in _ASSETS_DIR.iterdir() if f.is_file()]
