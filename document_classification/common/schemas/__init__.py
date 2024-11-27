"""Common schemas for document classification."""

from .bounding_box import BoundingBox
from .document import Document
from .line import Line
from .word import Word

__all__ = (
    "BoundingBox",
    "Document",
    "Line",
    "Word",
)
