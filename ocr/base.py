from abc import ABC, abstractmethod
import pandas as pd

class BaseOCR(ABC):
    """Template for OCR implementations"""

    @abstractmethod
    def ocr(self, image) -> pd.DataFrame:
        pass

    def __call__(self, image) -> pd.DataFrame:
        return self.ocr(image)

    def get_ocr_text(self, df: pd.DataFrame) -> str:
        return " ".join(df["text"])