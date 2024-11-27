from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Represents a single word in a document with its bounding box coordinates."""

    text: str = Field(description="The actual text content of the word")
    x0: float = Field(description="The left x-coordinate of the word's bounding box")
    y0: float = Field(description="The bottom y-coordinate of the word's bounding box")
    x2: float = Field(description="The right x-coordinate of the word's bounding box")
    y2: float = Field(description="The top y-coordinate of the word's bounding box")


class Line(BaseModel):
    """Represents a line of text in a document containing multiple words."""

    words: list[Word] = Field(description="A list of Word objects that make up this line")


class Document(BaseModel):
    """Represents a complete document containing multiple lines of text."""

    lines: list[Line] = Field(description="A list of Line objects that make up this document")
