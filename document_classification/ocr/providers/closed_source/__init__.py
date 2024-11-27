"""
Initialize the closed source OCR providers.

Top providers:
- Google Vision
- AWS Tesseract
- Azure OCR
"""

from .google_vision import GoogleVisionOCR

__all__ = ("GoogleVisionOCR",)
