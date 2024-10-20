from ocr.open_source.paddle import PaddleOCR
from ocr.open_source.tesseract import TesseractOCR


class OCRFactory:
    @staticmethod
    def create_ocr(ocr_type: str, **kwargs):
        if ocr_type == "tesseract":
            return TesseractOCR(**kwargs)
        elif ocr_type == "paddle":
            return PaddleOCR(**kwargs)
        else:
            raise ValueError(f"Unknown OCR type: {ocr_type}")
