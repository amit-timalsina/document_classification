from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import (  # type: ignore[import-untyped]
    AutoModelForSequenceClassification,
    BertTokenizer,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from document_classification.language_model.config import ModelConfig


class LanguageModel:
    """Main class for Language Model."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        """Initialize the model with configuration."""
        self.config = config
        self.device = device
        self.model, self.tokenizer = self._initialize_model()
        self.model.to(self.device)

    def _initialize_model(self) -> tuple[AutoModelForSequenceClassification, BertTokenizer]:
        """Initialize the BERT model and tokenizer."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
        )
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        return model, tokenizer

    def save(self, save_path: Path) -> None:
        """Save the model."""
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
