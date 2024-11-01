from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logger import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from language_model.slm_model import SLMModel


class SLMModelTrainer:
    """Main class for Sequence Learning Model training."""

    def __init__(self, model: SLMModel, learning_rate: float) -> None:
        """Initialize the trainer with model."""
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
        )

    def _process_batch(
        self,
        batch_texts: list[str],
        batch_labels: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Process a batch of data."""
        inputs = self.model.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs, batch_labels.to(self.model.device)

    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0.0

        for batch_texts, batch_labels in train_dataloader:
            inputs, labels = self._process_batch(batch_texts, batch_labels)
            outputs = self.model.model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_dataloader)

    def _validate(self, val_dataloader: DataLoader) -> tuple[float, float]:
        """Perform validation."""
        self.model.model.eval()
        total_loss = 0.0
        correct_predictions: int | float = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_texts, batch_labels in val_dataloader:
                inputs, labels = self._process_batch(batch_texts, batch_labels)
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
