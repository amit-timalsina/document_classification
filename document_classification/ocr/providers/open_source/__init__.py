"""OCRProvider implementations for Open Source OCR tools."""

from .paddle import PaddleOCR
from .tesseract import TesseractOCR

__all__ = (
    "PaddleOCR",
    "TesseractOCR",
)
