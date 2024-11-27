from document_classification.ocr.base import OCRProvider
from document_classification.ocr.providers.closed_source.google_vision import GoogleVisionOCR
from document_classification.ocr.providers.open_source.paddle import PaddleOCRProvider
from document_classification.ocr.providers.open_source.tesseract import TesseractOCR


class OCRFactory:
    """Factory class for creating OCR provider instances."""

    @staticmethod
    def create_ocr(provider: str) -> OCRProvider:
        """
        Create and return an OCR provider based on the given provider name.

        Args:
            provider (str): The name of the OCR provider.

        Returns:
            OCRProvider: An instance of the specified OCR provider.

        Raises:
            ValueError: If an unsupported OCR provider is specified.

        """
        if provider == "tesseract":
            return TesseractOCR()
        if provider == "paddleocr":
            return PaddleOCRProvider()
        if provider == "google_vision":
            return GoogleVisionOCR()
        msg = f"Unsupported OCR provider: {provider}"
        raise ValueError(msg)
