"""Initialize the OCR package."""

from .closed_source import GoogleVisionOCR
from .open_source import PaddleOCR, TesseractOCR

__all__ = (
    "GoogleVisionOCR",
    "PaddleOCR",
    "TesseractOCR",
)
