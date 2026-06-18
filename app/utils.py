from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resize_if_too_large(
    image: np.ndarray,
    max_dimension: int = 1280,
    threshold: int = 1600,
) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= threshold and height <= threshold:
        return image

    scale = max_dimension / max(width, height)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def save_image_high_quality(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    params: list[int] = []
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    return cv2.imwrite(str(path), image, params)
