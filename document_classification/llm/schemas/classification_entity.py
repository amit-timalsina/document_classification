from __future__ import annotations

from pydantic import BaseModel


class ClassificationEntity(BaseModel):
    """Label and description of a classification."""

    label: str
    description: str
