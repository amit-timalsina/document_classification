from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from document_classification.common.schemas.line import Line


class Document(BaseModel):
    """Represents a complete document containing multiple lines of text."""

    lines: list[Line] = Field(description="A list of Line objects that make up this document")
