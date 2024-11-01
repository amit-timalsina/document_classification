from __future__ import annotations

from torch.utils.data import Dataset


class OCRDataset(Dataset):
    """Extension of PyTorch's Dataset class for OCR data."""

    def __init__(self, texts: list[str], labels: list[int]) -> None:  # noqa: D107
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Return a sample from the dataset at the given index."""
        return self.texts[idx], self.labels[idx]
