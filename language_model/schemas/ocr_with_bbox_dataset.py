from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from common.schemas.bounding_box import BoundingBox


class OCRwithBBoxDataset(Dataset):
    """Extension of PyTorch's Dataset class for OCR data."""

    def __init__(  # noqa: D107
        self,
        texts: list[str],
        labels: list[int],
        bboxes: list[list[BoundingBox]],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.bboxes = bboxes

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int, list[BoundingBox]]:
        """Return a sample from the dataset at the given index."""
        return self.texts[idx], self.labels[idx], self.bboxes[idx]
