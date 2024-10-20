import numpy as np
import pandas as pd
import pytesseract  # type: ignore[import-untyped]

from ocr.base import OCRProvider
from ocr.config import ocr_config
from ocr.mappings import standard_to_tesseract
from ocr.schemas.ocr_result import OCRResult


class TesseractOCR(OCRProvider):
    """Tesseract OCR provider."""

    def perform_ocr(self, image: np.ndarray) -> pd.DataFrame:
        """Take an image and return OCR results."""
        ocr_response = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        return self.standardize_output(ocr_response)

    def standardize_output(self, ocr_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the OCR response to the expected output format."""
        standard_columns = ocr_config.output_columns

        for col in standard_columns:
            if standard_to_tesseract.get(col) in ocr_df.columns:
                ocr_df[col] = ocr_df[standard_to_tesseract[col]]
            elif col not in ocr_df.columns:
                ocr_df[col] = None

        ocr_df["x2"] = ocr_df["x0"] + ocr_df["width"]
        ocr_df["y2"] = ocr_df["y0"] + ocr_df["height"]

        return OCRResult(df=ocr_df).standardize_output()
