from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from document_classification.language_model.tokenizers.base import BaseTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # type: ignore[import-untyped]


class TextWithLayoutTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, device: torch.device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.bbox_tokens = self._create_bbox_tokens()

    def _create_bbox_tokens(self) -> dict[str, int]:
        return {
            "top": self.tokenizer.convert_tokens_to_ids("[TOP]"),
            "bottom": self.tokenizer.convert_tokens_to_ids("[BOTTOM]"),
            "left": self.tokenizer.convert_tokens_to_ids("[LEFT]"),
            "right": self.tokenizer.convert_tokens_to_ids("[RIGHT]"),
        }

    def process_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # The batch is now a dictionary with 'texts', 'labels', and 'bboxes' keys
        texts = batch["texts"]
        labels = batch["labels"]
        bboxes = batch["bboxes"]

        inputs = self.tokenize_with_layout(texts, bboxes)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device)
        return inputs, labels

    def tokenize_with_layout(
        self,
        texts: list[str],
        bboxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        # bboxes is already a tensor, so we don't need to create it
        bbox_input = self._process_bbox_input(bboxes, input_ids.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox_input,
        }

    def _process_bbox_input(
        self,
        bboxes: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        batch_size, seq_length = shape
        bbox_input = torch.zeros((batch_size, seq_length, 4), dtype=torch.long)

        # Add special tokens for the first position
        bbox_input[:, 0, :] = torch.tensor(
            [
                self.bbox_tokens["top"],
                self.bbox_tokens["bottom"],
                self.bbox_tokens["left"],
                self.bbox_tokens["right"],
            ],
        )

        # Copy the bboxes to the bbox_input tensor
        max_bboxes = min(seq_length - 1, bboxes.shape[1])
        bbox_input[:, 1 : max_bboxes + 1, :] = bboxes[:, :max_bboxes, :]

        # If there are fewer bboxes than seq_length, pad with the last bbox
        if max_bboxes < seq_length - 1:
            bbox_input[:, max_bboxes + 1 :, :] = bbox_input[:, max_bboxes : max_bboxes + 1, :]

        return bbox_input
