from typing import List

from pydantic import BaseModel, Field

from document_classification.common.schemas.word import Word


class Line(BaseModel):
    """Represents a line of text in a document containing multiple words."""

    words: List[Word] = Field(  # noqa: FA100
        default_factory=list,
        description="A list of Word objects that make up this line",
    )
