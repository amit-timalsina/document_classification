import numpy as np
import pandas as pd
import pytesseract  # type: ignore[import-untyped]

from document_classification.ocr.base import OCRProvider
from document_classification.ocr.config import ocr_config
from document_classification.ocr.providers.mappings import standard_to_tesseract
from document_classification.ocr.schemas.ocr_result import OcrResult
from logger import logger


class TesseractOCR(OCRProvider):
    """Tesseract OCR provider."""

    def perform_ocr(self, image: np.ndarray) -> OcrResult:
        """Take an image and return OCR results."""
        ocr_response = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        logger.debug(f"OCR response columns: {ocr_response.columns}")
        return self.standardize_output(ocr_response)

    def standardize_output(self, ocr_df: pd.DataFrame) -> OcrResult:
        """Standardize the OCR response to the expected output format."""
        standard_columns = ocr_config.output_columns

        for col in standard_columns:
            if standard_to_tesseract.get(col) in ocr_df.columns:
                ocr_df[col] = ocr_df[standard_to_tesseract[col]]
            elif col not in ocr_df.columns:
                ocr_df[col] = None

        ocr_df["x2"] = ocr_df["x0"] + ocr_df["width"]
        ocr_df["y2"] = ocr_df["y0"] + ocr_df["height"]

        return OcrResult(ocr_df=ocr_df)
