from __future__ import annotations

from pydantic import BaseModel, ValidationInfo, field_validator
from torch.utils.data import Dataset


class BoundingBox(BaseModel):
    """Represents a bounding box with minimum and maximum x and y coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        """Calculate the center point of the bounding box."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def width(self) -> float:
        """Calculate the width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Calculate the height of the bounding box."""
        return self.y_max - self.y_min

    @field_validator("x_max")
    @classmethod
    def x_max_must_be_greater_than_or_equal_to_x_min(cls, v: float, info: ValidationInfo) -> float:
        """Validate that x_max is greater than x_min."""
        if "x_min" in info.data and v < info.data["x_min"]:
            msg = "x_max must be greater than x_min"
            raise ValueError(msg)
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_must_be_greater_than_or_equal_to_y_min(cls, v: float, info: ValidationInfo) -> float:
        """Validate that y_max is greater than y_min."""
        if "y_min" in info.data and v < info.data["y_min"]:
            msg = "y_max must be greater than y_min"
            raise ValueError(msg)
        return v


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


class OCRwithBBoxDataset(Dataset):
    """Extension of PyTorch's Dataset class for OCR data."""

    def __init__(
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
