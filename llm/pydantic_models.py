from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Classifications(BaseModel):
    """Label and description of a classification."""

    label: str
    description: str


class BaseDocumentClassification(BaseModel):
    """Base class for document classification."""

    classification: Any
    confidence: int | None = Field(
        description="From 1 to 10. 10 being the highest confidence. Always integer",
        ge=1,
        le=10,
    )


class DocumentClassificationCOT(BaseDocumentClassification):
    """Classify the document into a label using a chain of thought."""

    chain_of_thought: str = Field(
        ...,
        description="The chain of thought that led to the classification",
    )
