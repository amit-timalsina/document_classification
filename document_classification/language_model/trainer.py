from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logger import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from document_classification.language_model.schemas.slm_model import LanguageModel
    from document_classification.language_model.tokenizers.base import BaseTokenizer


class SLMModelTrainer:
    """Main class for Sequence Learning Model training."""

    def __init__(
        self,
        model: LanguageModel,
        learning_rate: float,
        tokenizer: BaseTokenizer,
    ) -> None:
        """Initialize the trainer with model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
        )

    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        self.model.model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            inputs, labels = self.tokenizer.process_batch(batch)
            outputs = self.model.model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_dataloader)

    def _validate(self, val_dataloader: DataLoader) -> tuple[float, float]:
        self.model.model.eval()
        total_loss = 0.0
        correct_predictions: int | float = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = self.tokenizer.process_batch(batch)
                outputs = self.model.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        return (
            total_loss / len(val_dataloader),
            correct_predictions / total_predictions,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Train the model."""
        for epoch in range(num_epochs):
            avg_train_loss = self._train_epoch(train_dataloader)
            avg_val_loss, accuracy = self._validate(val_dataloader)

            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Accuracy: {accuracy:.4f}")
