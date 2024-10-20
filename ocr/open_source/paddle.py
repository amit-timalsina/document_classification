import paddleocr # type: ignore[import-untyped]

from ocr.base import BaseOCR
from logger import logger
import pandas as pd

class PaddleOCR(BaseOCR):
    def __init__(self):
        super().__init__()
        self.ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

    def ocr(self, image):
        ocr_response = self.ocr_model.ocr(image, cls=True)
        logger.debug(f"OCR response: {ocr_response}")
        return self.dict_to_df(ocr_response)

    def dict_to_df(self, ocr_response) -> pd.DataFrame:
        # the schema needs to be predefined else different format will be returned
        return pd.DataFrame(ocr_response)
