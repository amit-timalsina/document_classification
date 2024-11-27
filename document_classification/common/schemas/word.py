from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Represents a single word in a document with its bounding box coordinates."""

    text: str = Field(description="The actual text content of the word")
    x0: float = Field(description="The left x-coordinate of the word's bounding box", ge=0)
    y0: float = Field(description="The bottom y-coordinate of the word's bounding box", ge=0)
    x2: float = Field(description="The right x-coordinate of the word's bounding box", ge=0)
    y2: float = Field(description="The top y-coordinate of the word's bounding box", ge=0)
