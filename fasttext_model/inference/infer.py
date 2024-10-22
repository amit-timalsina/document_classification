from pathlib import Path

import fasttext  # type: ignore[import-untyped]

from fasttext_model.text_preprocessor import TextPreprocessor


class Infer:
    """Performs inference using a trained fastText model."""

    def __init__(self, model_path: Path, preprocessor: TextPreprocessor) -> None:
        """Initialize Infer."""
        self.model = fasttext.load_model(str(model_path))
        self.preprocessor = preprocessor

    def predict(self, text: str) -> tuple:
        """Predict the label of the provided text."""
        preprocessed_text = self.preprocessor.preprocess_text(text)
        return self.model.predict(preprocessed_text)

    def predict_from_file(self, file_path: Path) -> tuple:
        """Predict the label of the text within the specified file."""
        with file_path.open("r", encoding="utf-8") as file:
            return self.predict(file.read())
