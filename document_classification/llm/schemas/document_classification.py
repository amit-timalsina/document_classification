from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from enum import Enum


class DocumentClassification(BaseModel):
    """Schema for document classification."""

    classification: Enum
    confidence: int | None = Field(
        description="From 1 to 10. 10 being the highest confidence. Always integer",
        ge=1,
        le=10,
    )
