import numpy as np
from google.cloud import vision

from document_classification.ocr.base import OCRProvider
from document_classification.ocr.schemas.ocr_result import OcrResult

from .client import GoogleVisionClient
from .credentials import GoogleCredentialsManager
from .ocr_processor import GoogleVisionOCRProcessor


class GoogleVisionOCR(OCRProvider):
    """OCR provider for Google Vision API."""

    def __init__(self) -> None:
        """Initialize the Google Vision OCR client."""
        self.setup_credentials()
        self.client = self.create_client()
        self.processor = GoogleVisionOCRProcessor(self.client)

    def setup_credentials(self) -> None:
        """Set up the Google Vision API credentials."""
        GoogleCredentialsManager.setup_credentials()

    def create_client(self) -> vision.ImageAnnotatorClient:
        """Create a client for interacting with Google Vision API."""
        return GoogleVisionClient.create_client()

    def perform_ocr(self, image: np.ndarray) -> OcrResult:
        """
        Perform OCR on the given image and return standardized results.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            OCRResult: The standardized OCR results.

        """
        ocr_dataframe = self.processor.perform_ocr(image)
        return OcrResult(ocr_df=ocr_dataframe)
