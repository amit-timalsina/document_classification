from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentClassification(BaseModel):
    """Schema for document classification."""

    classification: Any
    confidence: int | None = Field(
        description="From 1 to 10. 10 being the highest confidence. Always integer",
        ge=1,
        le=10,
    )
