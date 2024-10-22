from abc import ABC, abstractmethod

import numpy as np

from ocr.schemas.ocr_result import OcrResult


class OCRProvider(ABC):
    """Base class for OCR providers."""

    @abstractmethod
    def perform_ocr(self, image: np.ndarray) -> OcrResult:
        """Take an image and return OCR results."""
