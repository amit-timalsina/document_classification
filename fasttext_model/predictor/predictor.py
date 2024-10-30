from pathlib import Path

import fasttext  # type: ignore[import-untyped]

from fasttext_model.text_preprocessor import TextPreprocessor
from ocr.base import OCRProvider
from ocr.readers.file_reader import FileReader


class FasttextPredictor:
    """Performs inference using a trained fastText model."""

    def __init__(self, model_path: Path, preprocessor: TextPreprocessor) -> None:
        """Initialize Infer."""
        self.model = fasttext.load_model(str(model_path))
        self.preprocessor = preprocessor

    def predict(self, text: str) -> tuple:
        """Predict the label of the provided text."""
        preprocessed_text = self.preprocessor.preprocess_text(text)
        return self.model.predict(preprocessed_text)

    def predict_from_file(self, file_path: Path, ocr_provider: OCRProvider) -> tuple:
        """Predict the label from a PDF file using OCR."""
        images = FileReader.read_file_from_path(str(file_path))
        ocr_dicts = []

        for image in images:
            ocr_result = ocr_provider.perform_ocr(image)
            ocr_dicts.extend(ocr_result.ocr_dict)

        text = " ".join([i["text"] for i in ocr_dicts])
        return self.predict(text)
