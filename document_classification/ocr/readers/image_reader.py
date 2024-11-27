from pathlib import Path

import cv2
import numpy as np


class ImageReader:
    """Utilizes OpenCV to read images from various sources."""

    @staticmethod
    def read_image_from_path(image_path: str) -> np.ndarray:
        """Read an image from the given file path."""
        if not Path(image_path).is_file():
            msg = f"Image not found at {image_path}"
            raise FileNotFoundError(msg)
        return cv2.imread(image_path)

    @staticmethod
    def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Read an image from a byte array."""
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            msg = "Failed to decode image from bytes."
            raise ValueError(msg)
        return image

    @staticmethod
    def read_image_from_url(image_url: str) -> np.ndarray:
        """Read an image from a url."""
        raise NotImplementedError
