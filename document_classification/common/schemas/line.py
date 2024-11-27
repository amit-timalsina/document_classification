from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from document_classification.common.schemas.word import Word


class Line(BaseModel):
    """Represents a line of text in a document containing multiple words."""

    words: list[Word] = Field(description="A list of Word objects that make up this line")
