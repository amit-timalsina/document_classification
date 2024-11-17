import torch
from transformers import PreTrainedTokenizer  # type: ignore[import-untyped]

from language_model.tokenizers.base import BaseTokenizer


class TextTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, device: torch.device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def process_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        texts = batch["texts"]
        labels = batch["labels"]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device)
        return inputs, labels
