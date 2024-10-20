from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from ocr.exceptions.ocr_processing_error import ImageNotFoundError
from ocr.schemas.ocr_result import OcrResult


class ImageReader:
    """Read image from various sources."""

    @staticmethod
    def read_image_from_path(image_path: str) -> np.ndarray:
        """Read image from path."""
        if not Path(image_path).is_file():
            raise ImageNotFoundError
        return cv2.imread(image_path)

    @staticmethod
    def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Read image from bytes."""
        return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def read_image_from_url(image_url: str) -> np.ndarray:
        """Read image from url."""
        raise NotImplementedError


class OCRProvider(ABC):
    """Base class for OCR providers."""

    @abstractmethod
    def perform_ocr(self, image: np.ndarray) -> OcrResult:
        """Take an image and return OCR results."""
