from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import torch

from common.parsers.default_parser import DefaultParser
from common.parsers.layout_preserving_formatter import LayoutPreservingFormatter
from common.utils.json_to_ocr_text import json_to_ocr_text
from logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]

    from language_model.slm_model import LanguageModel


class SLMPredictor:
    """Predictor for the SLM model."""

    def __init__(self, model: LanguageModel, processed_data_path: Path) -> None:
        """Initialize the predictor with model."""
        self.model = model
        self.label_encoder = self.load_label_encoder(processed_data_path=processed_data_path)

    @staticmethod
    def load_label_encoder(processed_data_path: Path) -> LabelEncoder:
        """Load the label encoder."""
        with processed_data_path.open("rb") as f:
            _, _, label_encoder = pickle.load(f)  # noqa: S301
        logger.warning("Using pickle to load data. Ensure the source is trusted.")
        return label_encoder

    def predict_file(
        self,
        file_path: Path,
    ) -> tuple[str, float]:
        """Predict the label for a file."""
        parser = DefaultParser()
        formatter = LayoutPreservingFormatter()

        ocr_text = json_to_ocr_text(file_path, parser, formatter)

        if not ocr_text:
            return "Error", 0.0

        inputs = self.model.tokenizer(
            ocr_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self.model.model.eval()
        with torch.no_grad():
            outputs = self.model.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][int(predicted_class_idx)].item()

        predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]

        return predicted_label, confidence
