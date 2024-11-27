from typing import List

from pydantic import BaseModel, Field

from document_classification.common.schemas.line import Line


class Document(BaseModel):
    """Represents a complete document containing multiple lines of text."""

    lines: List[Line] = Field(  # noqa: FA100
        default_factory=list,
        description="A list of Line objects that make up this document",
    )
