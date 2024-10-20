# TODO (Amit): standardization is failing. Not tested

import numpy as np
import pandas as pd
from paddleocr import PaddleOCR  # type: ignore[import-untyped]

from ocr.base import OCRProvider
from ocr.schemas.ocr_result import OCRResult


class PaddleOCRProvider(OCRProvider):
    """OCR provider for Paddle OCR."""

    def __init__(self) -> None:
        """Initialize the Paddle OCR provider."""
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def perform_ocr(self, image: np.ndarray) -> pd.DataFrame:
        """Take an image and return OCR results."""
        result = self.ocr.ocr(image, cls=True)
        return self.format_result(result)

    def format_result(self, result: list) -> pd.DataFrame:
        """Format the OCR result into a pandas dataframe."""
        data = []
        for line in result:
            for word_info in line:
                bbox, (text, confidence) = word_info
                data.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "x0": bbox[0][0],
                        "y0": bbox[0][1],
                        "x2": bbox[2][0],
                        "y2": bbox[2][1],
                    },
                )
        return OCRResult(df=pd.DataFrame(data)).standardize_output()
