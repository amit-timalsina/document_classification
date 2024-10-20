import pytesseract # type: ignore[import-untyped]
from ocr.base import BaseOCR
from logger import logger
import pandas as pd

class TesseractOCR(BaseOCR):
    def ocr(self, image):
        ocr_response = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        logger.debug(f"OCR response: {ocr_response}")
        return self.dict_to_df(ocr_response)

    def dict_to_df(self, ocr_response) -> pd.DataFrame:
        # the schema needs to be predefined else different format will be returned
        return pd.DataFrame(ocr_response)
